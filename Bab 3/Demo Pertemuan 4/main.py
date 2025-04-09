import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import cv2
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path1", help="image1 from path that will be processed")
    parser.add_argument("image_path2", help="image2 from path that will be processed")
    args = parser.parse_args()

    img1 = cv2.imread(args.image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.image_path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        sys.exit("Error loading image files")
    
    results = {
        "Original": img1,
        "Negative": image_negative(img1),
        "Log Transform": log_transformations(img1),
        "Power-Law": power_law_transformations(img1),
        "Contrast Stretch": contrast_stretching(img1),
        "Bit-Plane": bit_plane_slicing(img1)[7],
        "Subtraction": image_subtraction(img1, img2),
        "AND Operation": logic_operations_and(img1, img2),
        "OR Operation": logic_operations_or(img1, img2),
        "XOR Operation": logic_operations_xor(img1, img2),
    }

    row = 3
    col = math.ceil(len(results) / row)

    fig, axes = plt.subplots(row, col, figsize=(15, 5))
    axes = axes.flatten()

    for i, (title, image) in enumerate(results.items()):
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def bit_plane_slicing(img):
    height, width = img.shape
    bit_planes = []
    
    for i in range(8):
        bit_plane = np.zeros((height, width), dtype=np.uint8)
        for row in range(height):
            for col in range(width):
                if (img[row][col] >> i) & 1:
                    bit_plane[row, col] = 255
        bit_planes.append(bit_plane)

    return bit_planes


def contrast_stretching(img):
    min_out = 0
    max_out = 255
    min_in = np.min(img)
    max_in = np.max(img)
    
    if min_in == max_in:
        stretched_img = img.copy()
    else:
        stretched_img = (img - min_in) * (max_out - min_out) / (max_in - min_in) + min_out
        stretched_img = np.clip(stretched_img, min_out, max_out).astype(np.uint8)
    return stretched_img


def logic_operations_xor(img1, img2):
    return cv2.bitwise_xor(img1, img2)


def logic_operations_or(img1, img2):
    return cv2.bitwise_or(img1, img2)


def logic_operations_and(img1, img2):
    return cv2.bitwise_and(img1, img2)


def image_subtraction(img1, img2):
    return cv2.subtract(img1, img2)


def power_law_transformations(img):
    gamma = 2.0
    c = 1.0
    img_normalized = img / 255.0
    power_law_img_normalized = c * (img_normalized ** gamma)
    return np.clip(power_law_img_normalized * 255, 0, 255).astype(np.uint8)


def log_transformations(img):
    c = 1.0
    img_normalized = img / 255.0
    log_img_normalized = c * np.log(1 + img_normalized)
    return np.clip(log_img_normalized * 255, 0, 255).astype(np.uint8)


def image_negative(img):
    return cv2.minMaxLoc(img)[1] - img


if __name__ == "__main__":
    main()