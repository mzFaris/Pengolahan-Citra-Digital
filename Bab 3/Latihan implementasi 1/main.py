import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import sys
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="image from path that will be processed")
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        sys.exit("File not found!")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = {
        "Original Image": img_rgb,
        "Light Method": light_method(img),
        "Average Method": avg_method(img),
        "Luminosity Method": lum_method(img),
    }

    row = 2
    col = math.ceil(len(results) / row)

    fig, axes = plt.subplots(row, col, figsize=(15, 5))
    axes = axes.flatten()

    for i, (title, image) in enumerate(results.items()):
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def avg_method(img):
    b, g, r = cv2.split(img)
    return (r.astype(float) + g.astype(float) + b.astype(float) / 3).astype(np.uint8)


def light_method(img):
    b, g, r = cv2.split(img)
    max_val = np.maximum(np.maximum(r, g), b).astype(float)
    min_val = np.minimum(np.minimum(r, g), b).astype(float)
    return ((max_val + min_val) / 2).astype(np.uint8)


def lum_method(img):
    b, g, r = cv2.split(img)
    return (0.21 * r + 0.71 * g + 0.07 * b).astype(np.uint8)


if __name__ == "__main__":
    main()