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

def calc_geometric_error(pt1, pt2, F):
    # Homog points
    pt1_homog = np.array([pt1[0], pt1[1], 1])
    pt2_homog = np.array([pt2[0], pt2[1], 1])

    # Calculate epipolar lines and normalize
    l_prime = F @ pt1_homog
    l_prime /= l_prime[-1]
    l = F.T @ pt2_homog
    l /= l[-1]

    # Calculate sampson distance
    l_prime_sq = l_prime**2
    l_sq = l**2
    epsilon_sq = (pt1_homog @ F @ pt2_homog)**2
    JJt = l_prime_sq[0] + l_prime_sq[1] + l_sq[0] + l_sq[1]
    return epsilon_sq/JJt

def solve_DLT_triangulation(pts1, pts2, P1, P2):
    Q = P1
    R = P2

    # Grab point locations
    pts3d = []
    for p1, p2 in zip(pts1, pts2):
        A = np.zeros((6, 4))
        A[0] = p1[1]*Q[2] - Q[1]
        A[1] = Q[0] - p1[0]*Q[2]
        A[2] = p1[0]*Q[1]-p1[1]*Q[0]
        A[3] = p2[1]*R[2]-R[1]
        A[4] = R[0] - p2[0]*R[2]
        A[5] = p2[0]*R[1]-p2[1]*R[0]

        # Solve via SVD
        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[-1]
        pts3d.append(X)

    return pts3d
        

def triangulate(P1, P2, pts1, pts2):
    # Get the 3D points
    pts3d = solve_DLT_triangulation(pts1, pts2, P1, P2)
    
    # # We need to find the nullspace of P1 to find C1
    # # AKA: solve P1C1 = 0 to find the camera center
    # U, S, Vt = np.linalg.svd(P1)
    # C = Vt[-1]

    # # This is the projeciton of the other camera center in the second image
    # e_prime = P2 @ C
    # e_prime /= e_prime[-1]

    # # Find the fundamental matrix
    # # F = [e']_cross P2 P1^+
    # F = cross_product_mat(e_prime) @ P2 @ np.linalg.pinv(P1)
    return pts3d

def main(img1, img2, P1, P2, pts1, pts2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    P1 = np.load(P1)
    P2 = np.load(P2)
    pts1 = np.load(pts1)
    pts2 = np.load(pts2)

    # Triangle from camera matrices and points
    pts3d = triangulate(P1, P2, pts1, pts2)

    # Get the colors
    colors = []
    for pt in pts1:
        colors.append(img1[pt[1], pt[0]])

    # Display the colorized point cloud
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    x_tildes = np.vstack(pts3d)[:, :3]
    colors = np.vstack(colors)/255.
    colors_rgb = np.zeros_like(colors)
    colors_rgb[:, 0] = colors[:, 2]
    colors_rgb[:, 1] = colors[:, 1]
    colors_rgb[:, 2] = colors[:, 0]

    # Create a visualization window and add the point cloud
    pcd.points = o3d.utility.Vector3dVector(x_tildes)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--image1", default="data/q3/img1.jpg")
    parser.add_argument("-i2", "--image2", default="data/q3/img2.jpg")
    parser.add_argument("-P1", "--P1", default="data/q3/P1.npy")
    parser.add_argument("-P2", "--P2", default="data/q3/P2.npy")
    parser.add_argument("-p1", "--pts1", default="data/q3/pts1.npy")
    parser.add_argument("-p2", "--pts2", default="data/q3/pts2.npy")

    args = parser.parse_args()
    main(args.image1, args.image2, args.P1, args.P2, args.pts1, args.pts2)