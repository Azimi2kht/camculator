# import torch
# from matplotlib import pyplot as plt

# # import src.model.models as arch_module
# from src.utils import ConfigParser, load_config

# from .process import get_bounding_boxes


# class Solver:
#     def __init__(self, image):
#         self.image = image
#         self.chars = get_bounding_boxes(image, False)

#         config = ConfigParser(load_config("src/config.yaml"))

#         self.model = config.init_obj("arch", arch_module)
#         self.model.load_state_dict(
#             torch.load(config.save_dir.parent / "final_model.pth", weights_only=False)
#         )

#         # predict:
#         char = [c[0] for c in self.chars]
#         batch = torch.tensor(char, dtype=torch.float32).unsqueeze(1)
#         pred = self.model(batch)
#         output = torch.argmax(pred, dim=1)
#         for i, c in enumerate(char):
#             print(output[i])
#             plt.imshow(c, cmap="grey")
#             plt.show()
#         print(output)
