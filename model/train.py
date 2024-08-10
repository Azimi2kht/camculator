import src.model.loss as loss_module
import src.model.metrics as metric_module
import src.model.models as arch_module
import torch
from matplotlib import pyplot as plt
from src.data import CamculatorDataLoader
from src.trainer import Trainer
from src.utils import ConfigParser, load_config, set_seed
from torchvision.io import read_image

set_seed(42)


#################33 to be removed
import cv2
import imutils.contours
from matplotlib import pyplot as plt

IMG_SIZE = 200
MODEL_INPUT_SIZE = 28


def pad_image(image, target_size):
    th, tw = image.shape
    if tw > th:
        image = imutils.resize(image, width=MODEL_INPUT_SIZE)
    else:
        image = imutils.resize(image, height=MODEL_INPUT_SIZE)

    th, tw = image.shape

    pad_x = int(max(0, MODEL_INPUT_SIZE - tw) / 2.0)
    pad_y = int(max(0, MODEL_INPUT_SIZE - th) / 2.0)

    padded = cv2.copyMakeBorder(
        image,
        top=pad_y,
        bottom=pad_y,
        left=pad_x,
        right=pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    padded = cv2.resize(padded, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))

    return padded


# TODO: chage this so that steps are read from the config.yaml file
def preprocess(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    img_processed = cv2.GaussianBlur(image, (3, 3), 0)
    img_processed = cv2.Canny(img_processed, 20, 150)

    return img_processed


def get_bounding_boxes(image, display=False):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    img_processed = preprocess(image)

    contours = cv2.findContours(
        img_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)
    _, bounding_boxes = imutils.contours.sort_contours(contours, method="left-to-right")

    chars = []
    for x, y, w, h in bounding_boxes:
        if (w >= 2 and w <= 150) and (h >= 2 and h <= 120):
            roi = image[y : y + h, x : x + w]

            _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            padded = pad_image(thresh, MODEL_INPUT_SIZE)
            chars.append((padded, (x, y, w, h)))

    if display:
        boxes = [bounding_box[1] for bounding_box in chars]
        for x, y, w, h in boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        plt.imshow(image, cmap="grey")
        plt.show()

    return chars


#############3 to be removed


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

    ################ to be removed
    # trainer._load_checkpoint(
    # )
    # model = trainer.model

    # image = cv2.imread("./model/data/IMG_2007.png", cv2.IMREAD_GRAYSCALE)
    # chars = get_bounding_boxes(image, False)
    # # predict:
    # char = [c[0] for c in chars]
    # batch = torch.tensor(char, dtype=torch.float32, device="cuda").unsqueeze(1)
    # print(batch.shape, batch.dtype)
    # pred = model(batch)
    # output = torch.argmax(pred, dim=1)
    # # print(output)
    # for i, c in enumerate(char):
    #     print(output[i])
    #     plt.imshow(c, cmap="grey")
    #     plt.show()
    # # print(output)

    ######### remove


if __name__ == "__main__":
    config = ConfigParser(load_config("model/src/config.yaml"))
    main(config)
