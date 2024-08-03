import torch
from matplotlib import pyplot as plt

from base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        train_loader,
        valid_loader=None,
    ):
        super().__init__(model, criterion, metrics, optimizer, config)
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def _train_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        for batch_index, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        print(f"epoch: {epoch}, loss: {total_loss / len(self.train_loader)}")

        if self.valid_loader:
            self._valid_epoch(epoch)

    def _valid_epoch(self, epoch):
        self.model.eval()

        with torch.no_grad():
            for batch_index, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
