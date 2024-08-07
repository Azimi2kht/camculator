import cv2
import torch
from matplotlib import pyplot as plt

# import model.src.model.models as arch_module
from model.src.model.models import CamculatorModel

from .process import get_bounding_boxes

# from model.src.utils import ConfigParser, load_config


class Solver:
    def __init__(self):

        self.model = CamculatorModel()
        self.model.load_state_dict(
            torch.load("model/saved/models/camculator/model.pth", weights_only=True)
        )

        # config = ConfigParser(load_config("model/src/config.yaml"))
        # self.model = config.init_obj("arch", arch_module)
        # self.model.load_state_dict(
        #     torch.load(config.save_dir.parent / "final_model.pth", weights_only=False)
        # )

    def solve(self, image):
        self.chars = get_bounding_boxes(image, False)
        # predict:
        char = [c[0] for c in self.chars]
        batch = torch.tensor(char, dtype=torch.float32).unsqueeze(1)
        pred = self.model(batch)
        output = torch.argmax(pred, dim=1)
        for i, c in enumerate(char):
            print(output[i])
            plt.imshow(c, cmap="grey")
            cv2.imwrite(f"./model/data/{i}.png", c)
            # plt.show()
        print(output)
        return
