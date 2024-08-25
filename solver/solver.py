import torch

import model.src.model.models as arch_module
from model.src.utils import ConfigParser, load_config

from .process import get_bounding_boxes


class Solver:
    def __init__(self):
        config = ConfigParser(load_config("model/src/config.yaml"))
        self.model = config.init_obj("arch", arch_module)
        self.model.load_state_dict(
            torch.load(config.save_dir.parent / "final_model.pth", weights_only=False)
        )

    def solve(self, image):
        self.chars = get_bounding_boxes(image, False)

        char = [c[0] for c in self.chars]
        batch = torch.tensor(char, dtype=torch.float32).unsqueeze(1)
        output = torch.argmax(self.model(batch), dim=1)
        return output
