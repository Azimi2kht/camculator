from abc import abstractmethod

import torch
from numpy import inf


class BaseTrainer:
    """Base class for Trainers"""

    def __init__(self, model, criterion, metrics, optimizer, config):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer

        trainer_config = self.config["trainer"]

        self.num_epochs = trainer_config["epochs"]
        self.save_period = trainer_config["save_period"]
        self.save_dir = config.save_dir

        self.best_result = inf
        self.monitor_metric = trainer_config["monitor_metric"]
        self.epoch = 1

    @abstractmethod
    def _train_epoch(self, epoch):
        """trains the model one epoch

        Args:
            epoch (int): the current epoch index

        Raises:
            NotImplementedError: throws an exception if this method
            is not implemented in the child classes
        """
        raise NotImplementedError

    def train(self):
        """the main training logic"""
        for epoch in range(self.epoch, self.num_epochs + 1):
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
        """Saves a checkpoint of architecture, epoch, model, optimizer, best_result and config,
        so you can resume training later.

        Args:
            epoch (int): current epoch index
            save_best (bool, optional): If true, the best model is saved as model_best.pth. Defaults to False.
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

        if save_best:
            best_path = str(self.save_dir / "model_best.pth")
            torch.save(state, best_path)

    def _load_checkpoint(self, checkpoint_path):
        """loads saved checkpoint

        Args:
            checkpoint_path (str): the path to checkpoint
        """
        # TODO: change weights_only to true for security concerns
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.best_result = checkpoint["monitor_best"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])
