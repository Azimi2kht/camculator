import torch
from matplotlib import pyplot as plt
from torchvision.io import ImageReadMode, read_image

import model.loss as loss_module
import model.metrics as metric_module
import model.models as arch_module
from data import CamculatorDataLoader
from trainer import Trainer
from utils import ConfigParser, load_config, set_seed

set_seed(42)


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

    # trainer.train()
    trainer._load_checkpoint("saved/models/camculator/0804_165621/model_best.pth")
    img = read_image("data/raw/slash-0002.png").to(device=device, dtype=torch.float32).unsqueeze(0)
    print(img.shape)
    output = trainer.model(img)
    pred = torch.argmax(output, dim=1)
    print(pred)


if __name__ == "__main__":
    config = ConfigParser(load_config("src/config.yaml"))
    main(config)
