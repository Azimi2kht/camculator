from abc import abstractmethod

import torch


class BaseTrainer:
    def __init__(self, model, criterion, metrics, optimizer, config):
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer

        trainer_config = config["trainer"]

        self.epochs = trainer_config["epochs"]
        self.save_period = trainer_config["save_period"]

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(1, self.epochs + 1):
            result = self._train_epoch(epoch)

            # log = {"epoch": epoch}
            # log.update(result)

            # for key, value in log.items():
            #     self.logger.info("    {:15s}: {}".format(str(key), value))
