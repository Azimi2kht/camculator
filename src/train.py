import numpy as np
import torch
from torch.optim import Adam

import model.loss as loss_module
import model.metrics as metric_module
import model.models as arch_module
from data import CamculatorDataLoader
from trainer import Trainer
from utils import ConfigParser, load_config

# set the seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = config.init_obj("arch", arch_module)
    model = model.to(device)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    metrics = [getattr(metric_module, met) for met in config["metrics"]]
    criterion = getattr(loss_module, config["loss"])
    train_loader, valid_loader = CamculatorDataLoader(config["data"]).get_train_valid()

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    trainer.train()


if __name__ == "__main__":
    config = ConfigParser(load_config("src/config.yaml"))
    main(config)
