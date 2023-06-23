import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.model.deformable_transformer import DeformableTransformer
from src.model.utils import inverse_sigmoid_offset, sigmoid_offset
from src.model.pos_encoding import PositionalEncoding1D
from src.model.backbone import PyramidResNet


class FFN(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TESTR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.model.device)

        self.backbone = PyramidResNet()
        
        # fmt: off
        self.d_model                 = cfg.model.transformer.hidden_dim
        self.nhead                   = cfg.model.transformer.nheads
        self.num_encoder_layers      = cfg.model.transformer.enc_layers
        self.num_decoder_layers      = cfg.model.transformer.dec_layers
        self.dim_feedforward         = cfg.model.transformer.dim_feedforward
        self.dropout                 = cfg.model.transformer.dropout
        self.activation              = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels      = 3
        self.dec_n_points            = cfg.model.transformer.enc_n_points
        self.enc_n_points            = cfg.model.transformer.dec_n_points
        self.num_proposals           = cfg.model.transformer.num_queries
        self.pos_embed_scale         = cfg.model.transformer.position_embedding_scale
        self.num_ctrl_points         = cfg.model.transformer.num_ctrl_points
        self.num_classes             = 1
        self.max_text_len            = cfg.model.transformer.num_chars
        self.voc_size                = cfg.model.transformer.voc_size
        self.sigmoid_offset          = not cfg.model.transformer.use_polygon

        self.text_pos_embed = PositionalEncoding1D(self.d_model, normalize=True, scale=self.pos_embed_scale)
        # fmt: on
        
        self.transformer = DeformableTransformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout, activation=self.activation, return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels, dec_n_points=self.dec_n_points, 
            enc_n_points=self.enc_n_points, num_proposals=self.num_proposals,
        )
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)
        self.ctrl_point_coord = FFN(self.d_model, self.d_model, 2, 3)
        self.bbox_coord = FFN(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)
        self.text_class = nn.Linear(self.d_model, self.voc_size+1)

        # shared prior between instances (objects)
        self.ctrl_point_embed = nn.Embedding(self.num_ctrl_points, self.d_model)
        self.text_embed = nn.Embedding(self.max_text_len, self.d_model)

        strides = [8, 16, 32]
        num_channels = [512, 1024, 2048]
        num_backbone_outs = len(strides)
        input_proj_list = []
        for i in range(num_backbone_outs):
            in_channels = num_channels[i]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                nn.GroupNorm(32, self.d_model),
            ))
        self.input_proj = nn.ModuleList(input_proj_list)

        self.aux_loss = cfg.model.transformer.aux_loss

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.bbox_class.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.ctrl_point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.ctrl_point_coord.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.num_decoder_layers
        self.ctrl_point_class = nn.ModuleList(
            [self.ctrl_point_class for _ in range(num_pred)])
        self.ctrl_point_coord = nn.ModuleList(
            [self.ctrl_point_coord for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None

        nn.init.constant_(self.bbox_coord.layers[-1].bias.data[2:], 0.0)
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_embed = self.bbox_coord

        self.to(self.device)

    def forward(self, x):
        features = self.backbone(x)

        srcs = []
        masks = []
        for l, src in enumerate(features):
            srcs.append(self.input_proj[l](src))

        ctrl_point_embed = self.ctrl_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)
        text_pos_embed = self.text_pos_embed(self.text_embed.weight)[None, ...].repeat(self.num_proposals, 1, 1)
        text_embed = self.text_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)

        hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs, masks, pos, ctrl_point_embed, text_embed, text_pos_embed, text_mask=None)

        outputs_classes = []
        outputs_coords = []
        outputs_texts = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)
            outputs_class = self.ctrl_point_class[lvl](hs[lvl])
            tmp = self.ctrl_point_coord[lvl](hs[lvl])
            if reference.shape[-1] == 2:
                tmp += reference[:, :, None, :]
            else:
                assert reference.shape[-1] == 4
                tmp += reference[:, :, None, :2]
            outputs_texts.append(self.text_class(hs_text[lvl]))
            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_text = torch.stack(outputs_texts)

        out = {'pred_logits': outputs_class[-1],
               'pred_ctrl_points': outputs_coord[-1],
               'pred_texts': outputs_text[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_text)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_text):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_ctrl_points': b, 'pred_texts': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_text[:-1])]