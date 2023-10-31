# Third Party
import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# In House
from q1a import calc_F_eight, draw_epipolar_lines
from q1b import calc_F_seven

def run_ransac(pts1, pts2, ransac_iter=250, tol=5e-3, F_fn = calc_F_eight):
    # First select number of points based on the algo:
    if F_fn == calc_F_eight:
        num_pts = 8
    elif F_fn == calc_F_seven:
        num_pts = 7
    else:
        raise ValueError("Unknown func")
    
    # RANSAC loop
    best_F = np.eye(3)
    best_num_inliers = -1
    for i in range(ransac_iter):
        # Randomly select points
        chosen_idxs = np.random.choice(list(range(pts1.shape[0])), num_pts)
        pts1_rand   = pts1[chosen_idxs, :]
        pts2_rand   = pts2[chosen_idxs, :]

        # Fit model based on points
        calc_F = F_fn(pts1_rand, pts2_rand)

        if type(calc_F) == np.ndarray:
            F_list = [calc_F]
        else:
            F_list = calc_F


        pts1_homog = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        pts2_homog = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
        for F in F_list:
            # Calculate error for selected points
            errors = []
            for i in range(pts1.shape[0]):
                errors.append(pts2_homog[i] @ F @ pts1_homog[i])

            # See how many inliers bsed on the error
            num_inliers = np.sum(np.abs(np.array(errors)) <= tol)

            # Set the best yet
            if num_inliers > best_num_inliers:
                best_F = F
                best_num_inliers = num_inliers
    
    return best_F, best_num_inliers/pts1.shape[0]
    

def main(img1_file, img2_file, intr_file, corr_file, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Read files of interest
    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)
    intr = np.load(intr_file)
    K1 = intr["K1"]
    K2 = intr["K2"]
    corr = np.load(corr_file)
    pts1 = corr["pts1"]
    pts2 = corr["pts2"]

    # Create plot over different number of iterations
    iteration_list = [1, 5, 10, 25, 100, 150, 250, 500, 1000]
    inlier_pct_eight = []
    inlier_pct_seven = []
    # for num_iter in iteration_list:
    #     _, eight_inlier_pct = run_ransac(pts1, pts2, ransac_iter=num_iter, F_fn=calc_F_eight)
    #     inlier_pct_eight.append(eight_inlier_pct)
    #     _, seven_inlier_pct = run_ransac(pts1, pts2, ransac_iter=num_iter, F_fn=calc_F_seven)
    #     inlier_pct_seven.append(seven_inlier_pct)

    # # Plot results for eight point
    # plt.figure(1)
    # plt.plot(iteration_list, inlier_pct_eight, "go--", label="Eight Point Algo")
    # plt.plot(iteration_list, inlier_pct_seven, "ro-", label="Seven Point Algo")
    # plt.title("Inliner % vs. Number of Iterations for Eight and Seven Point Algo")
    # plt.xlabel("Number of Iterations")
    # plt.ylabel("Ratio of Inliers")
    # plt.legend()
    # plt.show()

    # Run RANSAC on eight-point and seven-point
    F_eight, eight_percent_inliner = run_ransac(pts1, pts2, F_fn=calc_F_eight)
    draw_epipolar_lines(img1, img2, F_eight, pts1, pts2, f"{out_dir}/eight_point_epi.png", show_lines=True)
    F_seven, seven_percent_inlier  = run_ransac(pts1, pts2, F_fn=calc_F_seven)
    draw_epipolar_lines(img1, img2, F_seven, pts1, pts2, f"{out_dir}/seven_point_epi.png", show_lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--image_one", default="data/q1a/teddy/image_1.jpg")
    parser.add_argument("-i2", "--image_two", default="data/q1a/teddy/image_2.jpg")
    parser.add_argument("-k", "--intrinsics", default="data/q1a/teddy/intrinsic_matrices_teddy.npz")
    parser.add_argument("-c", "--corr", default="data/q1a/teddy/teddy_corresp_raw.npz")
    parser.add_argument("-o", "--out_dir", default="output/q2_ransac")
    args = parser.parse_args()

    main(args.image_one, args.image_two, args.intrinsics, args.corr, args.out_dir)