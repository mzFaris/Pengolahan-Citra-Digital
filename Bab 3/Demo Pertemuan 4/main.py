import cv2
import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def main(args):
    img_1 = cv2.imread(str(Path(args.image1)), cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread(str(Path(args.image2)), cv2.IMREAD_GRAYSCALE)
    
    if img_1 is None or img_2 is None:
        sys.exit("Error loading image files")

    mask = np.zeros_like(img_1)
    cv2.rectangle(mask, (900, 1100), (2600, 2400), 255, -1)


    results = {
        "Original": img_1,
        "Negative": image_negative(img_1),
        "Log Transform": log_transformations(img_1),
        "Power-Law": power_law_transformations(img_1),
        "Contrast Stretch": contrast_stretching(img_1),
        "Bit-Plane": bit_plane_slicing(img_1),
        "Subtraction": image_subtraction(img_1, img_2),
        "AND Operation": logic_operations_and(img_1, mask),
        "OR Operation": logic_operations_or(img_1, mask),
        "XOR Operation": logic_operations_xor(img_1, mask),
    }

    fig, axs = plt.subplots(3, 4, figsize=(15, 6))
    axes = axs.ravel()
    
    for (title, img), ax in zip(results.items(), axes):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def bit_plane_slicing(img, bit_plane=8):
    return ((img >> (bit_plane - 1)) & 1) * 255


def contrast_stretching(img, r1=55, s1=40, r2=140, s2=200):
    img = img.astype(np.float32)
    p = np.clip(np.select(
        [img <= r1, (img > r1) & (img <= r2), img > r2],
        [(s1/r1)*img, 
         ((s2-s1)/(r2-r1))*(img-r1)+s1,
         ((255-s2)/(255-r2))*(img-r2)+s2]
    ), 0, 255)
    return p.astype(np.uint8)


def logic_operations_xor(img, mask):
    return cv2.bitwise_xor(img, mask)


def logic_operations_or(img, mask):
    return cv2.bitwise_or(img, mask)


def logic_operations_and(img, mask):
    return cv2.bitwise_and(img, mask)


def image_subtraction(img1, img2):
    return cv2.subtract(img1, img2)


def power_law_transformations(img, gamma=1.5, c=255):
    return (c * np.power(img / c, gamma)).astype(np.uint8)


def log_transformations(img):
    c = 255 / np.log(256)
    return (c * np.log(img.astype(np.float32) + 1)).astype(np.uint8)


def image_negative(img):
    return np.max(img) - img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image1")
    parser.add_argument("image2")
    main(parser.parse_args())