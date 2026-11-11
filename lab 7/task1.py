import matplotlib
matplotlib.use("Agg")
import os
import time
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)
sample = cv2.imread("images/sample.jpg")


def show(ax, im, title=""):
    ax.imshow(im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cmap="gray")
    ax.set_title(title)
    ax.axis("off")


def poly_area(pts):
    pts = np.asarray(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def order_points(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def find_quad(image):
    g = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) != 4:
        approx = cv2.boxPoints(cv2.minAreaRect(c))
    return order_points(approx)


def quad_size(q):
    tl, tr, br, bl = q
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    return w, h


sh, sw = sample.shape[:2]
CW, CH = 900, 700
ox, oy = (CW - sw) // 2, (CH - sh) // 2
base = np.full((CH, CW, 3), 240, np.uint8)
base[oy:oy + sh, ox:ox + sw] = sample
cv2.rectangle(base, (ox, oy), (ox + sw - 1, oy + sh - 1), (0, 0, 255), 3)
box = np.float32([[ox, oy], [ox + sw, oy], [ox + sw, oy + sh], [ox, oy + sh]])
cx, cy = CW / 2, CH / 2


def about_centre(M):
    C = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], float)
    Ci = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    return C @ M @ Ci


def rot2(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


transforms = {
    "Translation (+140, +90)": np.array([[1, 0, 140], [0, 1, 90], [0, 0, 1]], float),
    "Rotation 25 deg": about_centre(rot2(25)),
    "Scaling (1.4, 0.7)": about_centre(np.diag([1.4, 0.7, 1.0])),
    "Shear x 0.35": about_centre(np.array([[1, 0.35, 0], [0, 1, 0], [0, 0, 1]], float)),
    "Shear y 0.30": about_centre(np.array([[1, 0, 0], [0.30, 1, 0], [0, 0, 1]], float)),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
show(axes[0, 0], base, "Original")
a0 = poly_area(box)
print(f"{'Transform':26s}{'area ratio':>12s}{'det(A)':>10s}{'corner angle':>14s}")
for ax, (name, M) in zip(axes.ravel()[1:], transforms.items()):
    moved_img = cv2.warpAffine(base, M[:2], (CW, CH), borderValue=(240, 240, 240))
    moved = cv2.perspectiveTransform(box.reshape(-1, 1, 2), M).reshape(-1, 2)
    v1, v2 = moved[1] - moved[0], moved[3] - moved[0]
    ang = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    print(f"{name:26s}{poly_area(moved) / a0:12.3f}{np.linalg.det(M[:2, :2]):10.3f}{ang:14.1f}")
    show(ax, moved_img, name)
plt.tight_layout()
plt.savefig("output/q1a_2d_transforms.png", dpi=120)
plt.close()


L = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], float)
height = 1.2
n = len(L)
V = np.vstack([np.c_[L, np.zeros(n)], np.c_[L, np.full(n, height)]])
E = [(i, (i + 1) % n) for i in range(n)]
E += [(n + i, n + (i + 1) % n) for i in range(n)]
E += [(i, n + i) for i in range(n)]
Vh = np.c_[V, np.ones(len(V))]
centroid = V.mean(axis=0)


def mat4(R3=None, t=(0, 0, 0)):
    M = np.eye(4)
    if R3 is not None:
        M[:3, :3] = R3
    M[:3, 3] = t
    return M


def rot3(axis, deg):
    v = np.zeros(3)
    v["xyz".index(axis)] = np.deg2rad(deg)
    return cv2.Rodrigues(v)[0]


def about_c(M):
    return mat4(t=centroid) @ M @ mat4(t=-centroid)


combined_R = rot3("z", 40) @ rot3("y", 40) @ rot3("x", 40)
variants = [
    ("Original", np.eye(4)),
    ("Rotate 40 deg about x", about_c(mat4(rot3("x", 40)))),
    ("Rotate 40 deg about y", about_c(mat4(rot3("y", 40)))),
    ("Rotate 40 deg about z", about_c(mat4(rot3("z", 40)))),
    ("Translate (2, -1, 3)", mat4(t=(2, -1, 3))),
    ("Rotate xyz + translate", mat4(t=(2, -1, 3)) @ about_c(mat4(combined_R))),
]

moved_sets = [(name, (M @ Vh.T).T[:, :3]) for name, M in variants]
allp = np.vstack([p for _, p in moved_sets])
lo, hi = allp.min(axis=0) - 0.5, allp.max(axis=0) + 0.5
edge_len = lambda P: np.array([np.linalg.norm(P[i] - P[j]) for i, j in E])

fig = plt.figure(figsize=(16, 9))
for k, (name, P) in enumerate(moved_sets):
    ax = fig.add_subplot(2, 3, k + 1, projection="3d")
    for i, j in E:
        ax.plot3D(*zip(P[i], P[j]), color="tab:blue" if k else "black")
    ax.scatter(*P.T, s=12, color="tab:red")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_title(name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    print(f"{name:26s} centroid {np.round(P.mean(axis=0), 3)}  max edge-length change {np.abs(edge_len(P) - edge_len(V)).max():.2e}")
plt.tight_layout()
plt.savefig("output/q1b_3d_transforms.png", dpi=120)
plt.close()
print("Combined 4x4 homogeneous matrix:\n", np.round(variants[-1][1], 4))


p = np.array([4.0, 3.0])
ph = np.append(p, 1.0)
print("Cartesian point:", p, "| homogeneous:", ph)
print("Same point written with w = 3:", 3 * ph, "-> divide by w:", (3 * ph)[:2] / (3 * ph)[2])

t = np.array([6.0, -2.0])
Th = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
print("Translation as 3x3 product:", (Th @ ph)[:2], "| expected", p + t)

Rm = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
Sm = np.diag([2.0, 2.0, 1.0])
M = Th @ Sm @ Rm
stepwise = Th @ (Sm @ (Rm @ ph))
print("Rotate 90, scale 2, translate in one matrix:\n", M)
print("One matrix result:", (M @ ph)[:2], "| step by step:", stepwise[:2], "| equal:", np.allclose(M @ ph, stepwise))

P = np.array([[1, 0, 0], [0, 1, 0], [0.002, 0.001, 1.0]])
q = P @ ph
print("Projective matrix output:", np.round(q, 4), "-> after dividing by w:", np.round(q[:2] / q[2], 4))

d = np.array([1.0, 2.0, 0.0])
print("Direction with w = 0 after translation:", Th @ d, "(unchanged, translation does not move directions)")
print("The extra coordinate turns translation into a matrix product, lets every transform combine into one matrix, and gives perspective its division by w.")


sh, sw = sample.shape[:2]
src3 = np.float32([[0, 0], [sw - 1, 0], [0, sh - 1]])
dst3 = np.float32([[50, 40], [sw - 90, 10], [90, sh - 30]])
A = cv2.getAffineTransform(src3, dst3)
affine_img = cv2.warpAffine(sample, A, (sw + 100, sh + 60), borderValue=(255, 255, 255))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
show(axes[0], sample, "Original")
show(axes[1], affine_img, "After affine transformation")
plt.tight_layout()
plt.savefig("output/q1d_affine.png", dpi=120)
plt.close()

A3 = np.vstack([A, [0, 0, 1]])


def apply_pts(pts):
    return (A3 @ np.c_[pts, np.ones(len(pts))].T).T[:, :2]


xs = np.linspace(40, 560, 60)
line1 = np.c_[xs, np.full_like(xs, 270.0)]
line2 = np.c_[xs, np.full_like(xs, 330.0)]
t1, t2 = apply_pts(line1), apply_pts(line2)


def straight_residual(P):
    c = np.polyfit(P[:, 0], P[:, 1], 1)
    return np.abs(np.polyval(c, P[:, 0]) - P[:, 1]).max()


d1 = (t1[-1] - t1[0]) / np.linalg.norm(t1[-1] - t1[0])
d2 = (t2[-1] - t2[0]) / np.linalg.norm(t2[-1] - t2[0])
mid_before = line1[len(xs) // 2 - 1:len(xs) // 2 + 1].mean(axis=0)
mid_after = apply_pts(mid_before[None])[0]
ratio_after = np.linalg.norm(mid_after - t1[0]) / np.linalg.norm(t1[-1] - t1[0])
ratio_before = np.linalg.norm(mid_before - line1[0]) / np.linalg.norm(line1[-1] - line1[0])
corner = np.float32([[0, 0], [sw, 0], [0, sh]])
cm = apply_pts(corner)
va, vb = cm[1] - cm[0], cm[2] - cm[0]
angle_after = np.degrees(np.arccos(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))))

print("Affine matrix A:\n", np.round(A, 4))
print("Straightness residual of the two lines after transform:", np.round([straight_residual(t1), straight_residual(t2)], 10))
print("Direction of the two parallel lines after transform:", np.round(d1, 4), np.round(d2, 4), "| cross product:", round(float(d1[0] * d2[1] - d1[1] * d2[0]), 10))
print("Length ratio of a midpoint before / after:", round(float(ratio_before), 4), round(float(ratio_after), 4))
print("Right angle at the image corner becomes:", round(float(angle_after), 2), "degrees")
print("Area scale factor det(A):", round(float(np.linalg.det(A[:, :2])), 4))
print("Straight lines remain straight:", bool(max(straight_residual(t1), straight_residual(t2)) < 1e-6))
print("Parallel lines remain parallel:", bool(abs(d1[0] * d2[1] - d1[1] * d2[0]) < 1e-9))


book = cv2.imread("images/book_tilted.jpg")
flat_reference = cv2.imread("images/book_flat_reference.jpg")
true_quad = np.float32([[180, 90], [760, 150], [700, 620], [120, 540]])

corners = find_quad(book)
wd, hd = quad_size(corners)
target = np.float32([[0, 0], [wd - 1, 0], [wd - 1, hd - 1], [0, hd - 1]])
Hm = cv2.getPerspectiveTransform(corners, target)
flat = cv2.warpPerspective(book, Hm, (wd, hd), flags=cv2.INTER_CUBIC)

marked = book.copy()
cv2.polylines(marked, [np.int32(corners)], True, (0, 255, 0), 4)
for pt in corners:
    cv2.circle(marked, tuple(int(v) for v in pt), 9, (0, 0, 255), -1)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
show(axes[0], book, "Tilted photo")
show(axes[1], marked, "Detected page corners")
show(axes[2], flat, "Corrected with homography")
plt.tight_layout()
plt.savefig("output/q1e_homography.png", dpi=120)
plt.close()
cv2.imwrite("output/q1e_flat_book.jpg", flat)

top = corners[1] - corners[0]
bottom = corners[2] - corners[3]
angle_between = np.degrees(np.arccos(np.dot(top, bottom) / (np.linalg.norm(top) * np.linalg.norm(bottom))))
corner_err = np.linalg.norm(corners - true_quad, axis=1)
mae = np.mean(cv2.absdiff(cv2.resize(flat, (700, 500)), flat_reference))
print("Detected corners (TL, TR, BR, BL):\n", np.round(corners, 1))
print("Homography matrix:\n", np.round(Hm, 5))
print("Output size:", (wd, hd))
print("Top and bottom page edges differ by", round(float(angle_between), 2), "degrees in the photo (parallel on the real page)")
print("Corner error against the known page position (px):", np.round(corner_err, 2))
print("Mean absolute difference from the true flat page:", round(float(mae), 2), "/ 255")
