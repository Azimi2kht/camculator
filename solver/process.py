import cv2
import imutils.contours
from matplotlib import pyplot as plt

IMG_SIZE = 200
MODEL_INPUT_SIZE = 28
OFFSET = 5


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
            roi = image[y - OFFSET: y + h + OFFSET, x - OFFSET : x + w + OFFSET]

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


if __name__ == "__main__":
    image = cv2.imread("./data/IMG_2007.png", cv2.IMREAD_GRAYSCALE)
    get_bounding_boxes(image, True)
