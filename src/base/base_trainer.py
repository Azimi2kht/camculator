from abc import abstractmethod

import torch
from numpy import inf


class BaseTrainer:
    def __init__(self, model, criterion, metrics, optimizer, config):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer

        trainer_config = self.config["trainer"]

        self.epochs = trainer_config["epochs"]
        self.save_period = trainer_config["save_period"]
        self.save_dir = config.save_dir

        self.best_result = inf
        self.monitor_metric = trainer_config["monitor_metric"]

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(1, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {"epoch": epoch}
            log.update(result)

            best = False
            is_improved = log[self.monitor_metric] <= self.best_result
            if is_improved:
                self.best_result = log[self.monitor_metric]
                best = True

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.best_result,
            "config": self.config,
        }
        filename = str(self.save_dir / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename)
        # self.logger.info("Saving checkpoint: {} ...".format(filename))

        if save_best:
            best_path = str(self.save_dir / "model_best.pth")
            torch.save(state, best_path)
            # self.logger.info("Saving current best: model_best.pth ...")
