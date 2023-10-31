# Third Party
import argparse
import cv2
import numpy as np

# In House


def main(img1_file, img2_file, intr_file, corr_file):
    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--image_one", default="data/q1a/teddy/image_1.jpg")
    parser.add_argument("-i2", "--image_two", default="data/q1a/teddy/image_2.jpg")
    parser.add_argument("-k", "--intrinsics", default="data/q1a/teddy/intrinsic_matrices_teddy.npz")
    parser.add_argument("-c", "--corr", default="data/q1a/teddy/teddy_corresp_raw.npz")
    parser.add_argument("-o", "--out_dir", default="output/q1a_teddy")
    args = parser.parse_args()

    main(args.image_one, args.image_two, args.intrinsics, args.corr, args.out_dir)