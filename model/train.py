import src.model.loss as loss_module
import src.model.metrics as metric_module
import src.model.models as arch_module
import torch
from src.data import CamculatorDataLoader
from src.trainer import Trainer
from src.utils import ConfigParser, load_config, set_seed

set_seed(42)


def get_trainer_settings(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = config.init_obj("arch", arch_module)
    model = model.to(device)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    metrics = [getattr(metric_module, met) for met in config["metrics"]]
    criterion = getattr(loss_module, config["loss"])
    train_loader, valid_loader = CamculatorDataLoader(config["data"]).get_train_valid()

    trainer_settings = {
        "model": model,
        "criterion": criterion,
        "metrics": metrics,
        "optimizer": optimizer,
        "config": config,
        "device": device,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
    }
    return trainer_settings


def main(config):
    settings = get_trainer_settings(config)

    trainer = Trainer(**settings)

    trainer.train()


if __name__ == "__main__":
    config = ConfigParser(load_config("model/src/config.yaml"))
    main(config)
