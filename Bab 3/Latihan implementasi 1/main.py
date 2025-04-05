import sys
import cv2
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def main(args):
    img = cv2.imread(str(Path(args.path)), cv2.COLOR_BGR2RGB)
    if img is None:
        sys.exit("File not found")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = {
        "Original": img_rgb,
        "Light Method": light_method(img),
        "Average Method": avg_method(img),
        "Luminosity Method": lum_method(img),
    }

    fig, axs = plt.subplots(2, 2, figsize=(15, 4))
    axes = axs.ravel()

    for (title, img), ax in zip(results.items(), axes):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def avg_method(img):
    return np.round(np.mean(img, axis=2)).astype(np.uint8)


def light_method(img):
    max_val = np.max(img, axis=2)
    min_val = np.min(img, axis=2)
    return np.round((max_val + min_val) / 2).astype(np.uint8)


def lum_method(img):
    return np.round(0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    main(args)