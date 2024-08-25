import torch
from matplotlib import pyplot as plt
from src.base import BaseTrainer
from src.utils import MetricTracker


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

        self.train_metrics = MetricTracker(
            "loss", *[metric.__name__ for metric in self.metrics]
        )
        self.valid_metrics = MetricTracker(
            "loss", *[metric.__name__ for metric in self.metrics]
        )

    def _train_epoch(self, epoch):
        self.model.train()

        self.train_metrics.reset()
        for batch_index, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # update metrics
            self.train_metrics.update("loss", loss.item())
            for metric in self.metrics:
                self.train_metrics.update(metric.__name__, metric(outputs, labels))

        log = self.train_metrics.result()
        print(f"Epoch: {epoch} - train metrics: ", self.train_metrics.result())

        if self.valid_loader:
            valid_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in valid_log.items()})

        return log

    def _valid_epoch(self, epoch):
        self.model.eval()

        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_index, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # update metrics
                self.valid_metrics.update("loss", loss.item())
                for metric in self.metrics:
                    self.valid_metrics.update(metric.__name__, metric(outputs, labels))

            print(f"Epoch: {epoch} - valid metrics: ", self.valid_metrics.result())
            print("#" * 30)

        log = self.valid_metrics.result()
        return log
