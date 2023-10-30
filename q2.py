# Third Party
import argparse
import cv2
import numpy as np

# In House

def compute_T_norm(pts):
    x0, y0 = np.mean(pts, axis = 0)
    d_avg = np.mean(((pts[:, 0] - x0)**2 + (pts[:, 1] - y0)**2)**0.5)
    s = (2**0.5) * d_avg
    T = np.array([
        [s, 0, -s*x0],
        [0, s, -s*y0],
        [0, 0, 1]
    ])
    return T

def normalize_pts(pts):
    """
    Normalize the points via a similarity transform.
    This needs to be done for each pointset.
    The idea is to be 0 center and unit variance.
    """
    T = compute_T_norm(pts)
    pts_T = T @ np.hstack((pts.T, np.ones((pts.shape[0], 1))))
    return pts[:2]

def calc_F_seven(pts1, pts2):
    # Step 1: Normalize points
    pts1_norm = normalize_pts(pts1)
    pts2_norm = normalize_pts(pts2)

    # Step 2: Solve via SVD
    

    # The right and left null space are F1 and F2
    # Then, find the roots using a solver to get 1 or 3 solutions
    # np.roots(...)

    # Step 4: Unnormalize fundamental matrix
    np.linalg.inv(T) @ F
    
    

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