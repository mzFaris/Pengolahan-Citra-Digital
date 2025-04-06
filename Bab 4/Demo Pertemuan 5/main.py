import cv2
import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    args = parse_arguments()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit("Image not found")
    
    histogram_equalization_result = cv2.equalizeHist(img)
        


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to the input image")
    return parser.parse_args()



if __name__ == "__main__":
    main()