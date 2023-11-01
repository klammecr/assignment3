# Third Party
import argparse
import cv2
import numpy as np

# In House
from q1a import calc_F_eight, draw_epipolar_lines
from q2 import run_ransac

def find_correspondences(kp1, kp2, desc1, desc2):
    matches = {}
    for idx, d in enumerate(desc1):
        dist = np.sum((d - desc2)**2, 1)
        closest_pt = np.argmin(dist)
        if kp2[closest_pt] not in matches.values():
            matches[kp1[idx]] = kp2[closest_pt]

    pts1 = []
    pts2 = []
    for k,v in matches.items():
        pts1.append(k.pt)
        pts2.append(v.pt)

    return np.array(pts1), np.array(pts2)


def main(img1, img2, out_path):
    img_name = img1.split("/")[-1][:-5]

    # Read images
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    # To grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute SIFT features
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1_gray, None)
    kp2, desc2 = sift.detectAndCompute(img2_gray, None)

    # Find correspondences
    pts1, pts2 = find_correspondences(kp1, kp2, desc1, desc2)

    # Run RANSAC with eight point
    F, inlier_pct = run_ransac(pts1, pts2, F_fn=calc_F_eight)

    # Draw epipolar lines
    draw_epipolar_lines(img1, img2, F, pts1, pts2, f"{out_path}/{img_name}.jpg", show_lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--image1", default="data/USC1.jpg")
    parser.add_argument("-i2", "--image2", default="data/USC2.jpg")
    parser.add_argument("-o", "--out_path", default="output/q5")
    args = parser.parse_args()

    main(args.image1, args.image2, args.out_path)