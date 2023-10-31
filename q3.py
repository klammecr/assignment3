# Third Party
import argparse
import cv2
import numpy as np

# In House

def cross_product_mat(vec):
    """
    Create a skew symmetric cross product matrix.

    Args:
        line_coeff (np.ndarray): [3,1] coefficients of line
    """
    return np.array([
    [0, -vec[2], vec[1]],
    [vec[2], 0, -vec[0]],
    [-vec[1], vec[0], 0]
    ])

def triangulate(P1, P2, pts1, pts2):
    # We need to find the nullspace of P1 to find C1
    # AKA: solve P1C1 = 0 to find the camera center
    U, S, Vt = np.linalg.svd(P1)
    C = Vt[-1]

    # This is the projeciton of the other camera center in the second image
    e_prime = P2 @ C
    e_prime /= e_prime[-1]

    # Find the fundamental matrix
    # F = [e']_cross P2 P1^+
    F = cross_product_mat(e_prime) @ P2 @ np.linalg.pinv(P1)



def main(img1, img2, P1, P2, pts1, pts2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    P1 = np.load(P1)
    P2 = np.load(P2)
    pts1 = np.load(pts1)
    pts2 = np.load(pts2)

    # Triangle from camera matrices and points
    triangulate(P1, P2, pts1, pts2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--image1", "data/q3/img1.jpg")
    parser.add_argument("-i2", "--image2", "data/q3/img2.jpg")
    parser.add_argument("-P1", "--P1", "data/q3/P1.npy")
    parser.add_argument("-P2", "--P2", "data/q3/P2.npy")
    parser.add_argument("-p1", "--pts1", "data/q3/pts1.npy")
    parser.add_argument("-p2", "--pts2", "data/q3/pts2.npy")

    args = parser.parse_args()
    main(args.image1, args.image2, args.P1, args.P2, args.pts1, args.pts2)