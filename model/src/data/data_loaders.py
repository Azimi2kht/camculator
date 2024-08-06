from torch.utils.data import DataLoader, random_split

from .datasets import CamculatorDataset


class CamculatorDataLoader:
    def __init__(self, config):
        dataset = CamculatorDataset(config["data_dir"])

        data_size = len(dataset)
        val_size = int(data_size * config["valid_split_ratio"])
        train_size = int(data_size - val_size)

        train_data, valid_data = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_data, **config["data_loader"]["args"])
        self.valid_loader = DataLoader(valid_data, **config["data_loader"]["args"])

    def get_train_valid(self):
        return self.train_loader, self.valid_loader
