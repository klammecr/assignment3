# Third Party
import numpy as np
import argparse
import cv2
import sympy

# In House
from q1a import normalize_pts, draw_epipolar_lines

def calc_F_seven(pts1, pts2):
    # Step 1: Normalize Points
    pts1_norm, T1 = normalize_pts(pts1)
    pts2_norm, T2 = normalize_pts(pts2)

    # Step 2: Solve for F via SVD
    A = np.zeros((pts1.shape[0], 9))
    i = 0
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
    U, S, Vt = np.linalg.svd(A)

    # Step 3: Get f1 and f2 as the last two singular vectors
    F1 = Vt[-1].reshape(3,3)
    F1 /= F1[-1, -1]
    F2 = Vt[-2].reshape(3,3)
    F2 /= F2[-1, -1]

    # Use sympy to solve
    # lambda_ = sympy.symbols('lambda')
    # F1_ = sympy.Matrix(F1)
    # F2_ = sympy.Matrix(F2)
    # expr = lambda_ * F1_ + (1 - lambda_) * F2_
    # equation = sympy.Eq(sympy.det(expr), 0)
    # soln = sympy.solve(equation, lambda_)
    # accept_solns = [s.evalf() for s in soln]
    # real_solns = [s.as_real_imag()[0] for s in accept_solns if abs(s.as_real_imag()[1]) < 1e-9]
    
    # # Find the real roots
    # F_list = []
    # for s in real_solns:
    #     F = s* F1 + (1-s)*F2
    #     F_final = T2.T @ F @ T1
    #     F_final /= F_final[-1, -1]
    #     F_list.append(F_final)

    # Source:
    # https://ela.kpi.ua/bitstream/123456789/50759/1/%2892-93%29_Monastyrskyi.pdf
    constant = np.linalg.det(F1)
    linear = np.linalg.det(F1)*np.trace(F2*np.linalg.inv(F1))
    quadratic = np.linalg.det(F2)*np.trace(F1*np.linalg.inv(F2))
    cubic = np.linalg.det(F2)

    soln = np.roots(np.array([cubic, quadratic, linear, constant]))
    real_solns = soln[~np.iscomplex(soln)]
    F_list = []
    for real_soln in real_solns:
        F = F1 + real_soln.real*F2
        F_final = T2.T @ F @ T1
        F_final /= F_final[-1, -1]
        F_list.append(F_final)

    # Only return the real solutions
    # If we have RANSAC, we can find the best solution
    # real_solns = soln[~np.iscomplex(soln)]
    # F_list = []
    # for real_soln in real_solns:
    #     F = real_soln.real * F1 + (1-real_soln.real)*F2
    #     F_final = T2.T @ F @ T1
    #     F_final /= F_final[-1, -1]
    #     F_list.append(F_final)

    return F_list


def main(img1_file, img2_file, intr_file, corr_file, precise_corr, out_dir):
    # Read the files
    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)
    intr = np.load(intr_file)
    K1 = intr["K1"]
    K2 = intr["K2"]
    precise_corr = np.load(precise_corr)
    pts1_precise = precise_corr["pts1"]
    pts2_precise = precise_corr["pts2"]
    corr = np.load(corr_file)
    pts1 = corr["pts1"]
    pts2 = corr["pts2"]

    # Compute seven point algorithm
    F_list = calc_F_seven(pts1_precise, pts2_precise)

    draw_epipolar_lines(img1, img2, F_list[0], pts1_precise, pts2_precise, f"{out_dir}/F_epipolar_seven_point.png", show_lines=True)

    # DEBUG: Check F
    for i in range(pts1.shape[0]):
        print(np.array([pts2[i,0], pts2[i, 1], 1]) @ F_list[0] @ np.array([pts1[i,0], pts1[i,1], 1]))

# Entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--image_one", default="data/q1b/toybus/image_1.jpg")
    parser.add_argument("-i2", "--image_two", default="data/q1b/toybus/image_2.jpg")
    parser.add_argument("-k", "--intrinsics", default="data/q1b/toybus/intrinsic_matrices_toybus.npz")
    parser.add_argument("-c", "--corr", default="data/q1b/toybus/toybus_corresp_raw.npz")
    parser.add_argument("-pc", "--precise_corr", default="data/q1b/toybus/toybus_7_point_corresp.npz")
    parser.add_argument("-o", "--out_dir", default="output/q1b_toybus")
    args = parser.parse_args()

    main(args.image_one, args.image_two, args.intrinsics, args.corr, args.precise_corr, args.out_dir)