import json
from functools import reduce

import cv2
import torch
from torch.utils.data import Dataset

from src.data.augmentations import preproc_aug, main_aug, post_aug


class CustomDataset(Dataset):
    def __init__(self, anno_path, device, apply_augs=False):
        with open(anno_path, 'r') as f:
            self.anno = json.load(f)
        self.device = device
        self.apply_augs = apply_augs

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        sample = self.anno[idx]
        im = cv2.imread(sample["image"], cv2.IMREAD_COLOR)
        bboxs = sample.get('bboxes', [])
        texts = sample.get('texts', [])

        polys = sample.get('polys', [])
        n_polys = len(polys)
        n_pts = 1
        if len(polys):
            n_pts = len(sample['polys'][0])
            polys = reduce(lambda x, y: x + y, polys)

        out = preproc_aug(image=im, keypoints=polys, bboxes=bboxs, class_labels=['text'] * len(bboxs))

        if self.apply_augs:
            out = main_aug(image=out['image'], keypoints=out['keypoints'], bboxes=out['bboxes'], class_labels=['text'] * len(bboxs))
        out = post_aug(image=out['image'], keypoints=out['keypoints'], bboxes=out['bboxes'], class_labels=['text'] * len(bboxs))

        polys = out['keypoints']
        polys = [polys[i:i+n_pts] for i in range(n_polys)]

        return (out['image'].to(self.device),
                torch.tensor(out['bboxes'], device=self.device),
                torch.tensor(polys, device=self.device),
                torch.tensor(texts, device=self.device),
                )


if __name__ == '__main__':
    from tqdm import tqdm
    anno_path = r'C:\Users\Admin\PycharmProjects\TESTR\datasets\icdar2015\train.json'
    device = 'cuda'

    d = CustomDataset(anno_path, device, apply_augs=True)
    for i in tqdm(range(300)):
        a = d[i]

    print(a)