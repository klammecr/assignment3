# Third Party
import argparse
import cv2
import numpy as np
import os
from copy import deepcopy

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
    pts_T = T @ np.hstack((pts, np.ones((pts.shape[0], 1)))).T
    return pts_T[:2].T, T

def calc_F_eight(pts1, pts2):
    """
    Calculate fundamental matrix from correspondences.
    8 Point Algo

    Args:
        pts1 (np.ndarray): [N,2] array of points of image 1
        pts2 (np.ndarray): [N,2] array of points of image 2

    Returns:
        np.ndarray: Fundamental Matrix F [3,3]
    """
    A = np.zeros((pts1.shape[0], 9))
    i = 0
    pts1_norm, T1 = normalize_pts(pts1)
    pts2_norm, T2 = normalize_pts(pts2)
    for p_prime, p in zip(pts2_norm, pts1_norm):
        A[i, 0] = p_prime[0] * p[0]
        A[i, 1] = p_prime[0] * p[1]
        A[i, 2] = p_prime[0]
        A[i, 3] = p_prime[1] * p[0]
        A[i, 4] = p_prime[1] * p[1]
        A[i, 5] = p_prime[1]
        A[i, 6] = p[0]
        A[i, 7] = p[1]
        A[i, 8] = 1
        i += 1
    
    # Solve Ax = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    
    F = Vt[-1].reshape(3,3)

    # Decompose F and ensure it is rank 2
    U, S, Vt = np.linalg.svd(F)
    S_new = np.diag([S[0], S[1], 0])

    # Reform F
    F = U @ S_new @ Vt
    
    # Bring solution to pixel space
    F_final = T2.T @ F @ T1
    F_final /= F_final[-1, -1]

    # DEBUG: Should be close to 0
    # for p_prime, p in zip(pts2, pts1):
    #     res = np.array([p_prime[0], p_prime[1], 1]) @ F_final @ np.array([p[0], p[1], 1])
    #     print(res)

    return F_final

def compute_E_from_F(F, K1, K2):
    """
    Compute essential matrix from F

    Args:
        F (_type_): _description_
    """
    E = K2.T @ F @ K1
    E /= E[-1, -1]
    return E

def draw_epipolar_lines(img1, img2, F, pts1, pts2, out_fp, show_lines=False):
    out_im1 = deepcopy(img1)
    out_im2 = deepcopy(img2)
    colors  = [list(np.random.choice(range(256), size=3).astype("int")) for i in range(pts1.shape[0])]
    i = 0

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    for p, p_prime in zip(pts1, pts2):
        # Get color
        color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))

        # Find epipolar lines for each 
        l_prime = F @ np.array([p[0], p[1], 1])
        l_prime/= l_prime[-1]
        l       = F.T @ np.array([p_prime[0], p_prime[1], 1])
        l      /= l[-1]

        # Put point p_i and corresponding epipolar line l_i
        cv2.circle(out_im1, center=tuple(p.astype("int")), radius=5, thickness=-1, color = color)
        cv2.circle(out_im2, center=tuple(p_prime.astype("int")), radius=5, thickness=-1, color=color)

        # Coeff calculation for line
        if show_lines:
            lower_l_s  = int(-p_prime[0]/l_prime[0])
            higher_l_s = int((w2-1-p_prime[0]) /l_prime[0])
            lower_y_val  = int(p_prime[1] + lower_l_s*l_prime[1])
            higher_y_val = int(p_prime[1] + higher_l_s*l_prime[1])
            cv2.line(out_im2, (0, lower_y_val), (w1-1, higher_y_val), color = color, thickness=2)

        # Increment counter
        i+=1

    # Display result
    cv2.putText(out_im1, "View 1 with Point Annotations", (0, 20), cv2.FONT_HERSHEY_COMPLEX,1, (255, 255, 255))
    # cv2.imshow("f", out_im1)
    # cv2.waitKey()
    cv2.putText(out_im2, "View 2 with Epipolar Lines", (0, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
    # cv2.imshow("example", out_im1)
    # cv2.waitKey()
    cv2.imwrite(out_fp, np.hstack((out_im1, out_im2)))

def main(img1_file, img2_file, intr, corr, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Read files of interest
    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)
    intr = np.load(intr)
    K1 = intr["K1"]
    K2 = intr["K2"]
    corr = np.load(corr)
    pts1 = corr["pts1"]
    pts2 = corr["pts2"]

    # Fundamental matrix from correspondences (8 Point)
    F_eight = calc_F_eight(pts1, pts2)
    np.savetxt(f"{out_dir}/F.txt", F_eight)

    # Calculate the essential matrix from F
    E = compute_E_from_F(F_eight, K1, K2)
    np.savetxt(f"{out_dir}/E.txt", E)

    # Draw the epipoilar lines
    draw_epipolar_lines(img1, img2, F_eight, pts1[:10], pts2[:10], f"{out_dir}/F_epipolar_eight_pt.png", show_lines=True)

# Entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--image_one", default="data/q1a/teddy/image_1.jpg")
    parser.add_argument("-i2", "--image_two", default="data/q1a/teddy/image_2.jpg")
    parser.add_argument("-k", "--intrinsics", default="data/q1a/teddy/intrinsic_matrices_teddy.npz")
    parser.add_argument("-c", "--corr", default="data/q1a/teddy/teddy_corresp_raw.npz")
    parser.add_argument("-o", "--out_dir", default="output/q1a_teddy")
    args = parser.parse_args()

    main(args.image_one, args.image_two, args.intrinsics, args.corr, args.out_dir)

