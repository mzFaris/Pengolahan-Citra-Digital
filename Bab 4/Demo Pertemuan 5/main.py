import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import cv2
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="image from path that will be processed")
    args = parser.parse_args()

    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit("File not found!")

    equalized_img = cv2.equalizeHist(img)

    kernel_size = (5, 5)
    blurred_img = cv2.blur(img, kernel_size)

    scale = 0.5
    laplacian_kernel = np.array([[0, -1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=np.float32)
    
    laplacian = cv2.filter2D(img, cv2.CV_32F, laplacian_kernel)
    laplacian = np.clip(img - scale * laplacian, 0, 255).astype(np.uint8)

    result = {
        "Original Image": img,
        "Histogram Equalization": equalized_img,
        "Spatial Smoothing (Averaging Filter)": blurred_img,
        "Spatial Sharpening (Laplacian FIlter)": laplacian
    }

    row = 2
    col = math.ceil(len(result) / row)

    fig, axes = plt.subplots(row, col, figsize=(15, 5))
    axes = axes.flatten()

    for i, (title, image) in enumerate(result.items()):
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()