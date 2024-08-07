import cv2

from solver import Solver

if __name__ == "__main__":
    solver = Solver()

    image = cv2.imread("./model/data/IMG_2007.png", cv2.IMREAD_GRAYSCALE)
    solver.solve(image)
