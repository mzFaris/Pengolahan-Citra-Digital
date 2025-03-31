import sys
import cv2
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def main(image_path):
    img = cv2.imread(Path(image_path))
    if img is None:
        sys.exit("File not found")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width = img.shape[:2]
    
    light_img = np.apply_along_axis(light_method, 2, img)
    avg_img = np.apply_along_axis(avg_method, 2, img)
    lum_img = np.apply_along_axis(lum_method, 2, img)

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))

    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[1].imshow(light_img, cmap="gray")
    axs[1].set_title("Lightness Method")
    axs[2].imshow(avg_img, cmap="gray")
    axs[2].set_title("Average Method")
    axs[3].imshow(lum_img, cmap="gray")
    axs[3].set_title("Luminosity Method")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def avg_method(n):
    return np.round(np.sum(n) / 3).astype(np.uint8)


def light_method(n):
    return np.round((np.max(n) + np.min(n)) / 2).astype(np.uint8)

def lum_method(n):
    return np.round(0.21 * n[0] + 0.71 * n[1] + 0.07 * n[2]).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    main(args.path)