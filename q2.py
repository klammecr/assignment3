# Third Party
import argparse
import cv2
import numpy as np

# In House



    
    

def main(img1_file, img2_file, intr_file, corr_file):
    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--image1")
    parser.add_argument("-i2", "--image2")
    parser.add_argument("-k", "--intr")
    parser.add_argument("-c", "--corr")
    args = parser.parse_args()

    main(args.image1, args.image2, args.intr, args.corr)