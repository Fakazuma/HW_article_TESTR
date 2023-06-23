from transformers import AdamW
from transformers import get_scheduler

from src.model.models import TESTR
from src.data.dataloader import Dataloader

# TODO: maybe add parametres to optimizer


class Trainer:
    def __init__(self, cfg, backbone):
        self.optimizer = None
        self.model = None
        self.cfg = cfg
        self.backbone = backbone
        self.train_dataloader = Dataloader() # build dataloader

    def train(self):
        self.model = TESTR(self.cfg, backbone=self.backbone)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

        self.num_epochs = self.cfg.trainer.max_epochs

        num_training_steps = self.num_epochs * len(self.train_dataloader)
        self.lr_scheduler = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=0,
                                     num_training_steps=num_training_steps)

        for epoch in self.num_epochs:
            self.train_loop()

    def train_loop(self):
        for i, batch in enumerate(self.train_dataloader):
            self.run_step(batch)

    def run_step(self, batch):
        self.optimizer.zero_grad()

        outputs = self.model(batch)
        loss = outputs.loss
        loss.backward()

        self.lr_scheduler.step()

