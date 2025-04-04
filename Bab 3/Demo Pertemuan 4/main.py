import cv2
import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def main(image_path):
    img = cv2.imread(Path(image_path))
    if img is None:
        sys.exit("File not found")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_image_negative = np.apply_along_axis(image_negative, 1, img, np.max(img))
    img_log_transformations = np.apply_along_axis(log_transformations, 1, img)

    fig, axs = plt.subplots(3, 3, figsize=(15, 4))

    axs[0][0].imshow(img, cmap="gray")
    axs[0][0].set_title("Grayscale Image")
    axs[0][1].imshow(img_image_negative, cmap="gray")
    axs[0][1].set_title("Image Negative")
    axs[0][2].imshow(img_log_transformations, cmap="gray")
    axs[0][2].set_title("Log Transformations")

    for ax in axs:
        for a in ax:
            a.axis("off")

    plt.tight_layout()
    plt.show()


def log_transformations(n):
    return 1 * np.log(1 + n)


def image_negative(n, max):
    return max - n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    main(args.path)