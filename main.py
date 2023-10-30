# In House
from q1a import main as q1a_main
from q1b import main as q1b_main

# Question 1a
q1a_main("data/q1a/teddy/image_1.jpg", "data/q1a/teddy/image_2.jpg", "data/q1a/teddy/intrinsic_matrices_teddy.npz", "data/q1a/teddy/teddy_corresp_raw.npz", "output/q1a_teddy")
q1a_main("data/q1a/chair/image_1.jpg", "data/q1a/chair/image_1.jpg", "data/q1a/chair/intrinsic_matrices_chair.npz", "data/q1a/chair/chair_corresp_raw.npz", "output/q1a_chair")

# Question 1b
q1b_main("data/q1b/toybus/image_1.jpg", "data/q1b/toybus/image_2.jpg", "data/q1b/toybus/intrinsic_matrices_toybus.npz", "data/q1b/toybus/toybus_corresp_raw.npz", "data/q1b/toybus/toybus_7_point_corresp.npz", "output/q1b_toybus")
q1b_main("data/q1b/toytrain/image_1.jpg", "data/q1b/toytrain/image_2.jpg", "data/q1b/toytrain/intrinsic_matrices_toytrain.npz", "data/q1b/toytrain/toytrain_corresp_raw.npz", "data/q1b/toytrain/toytrain_7_point_corresp.npz", "output/q1b_toytrain")