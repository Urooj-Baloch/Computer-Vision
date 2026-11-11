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


def apply_perspective_transform(image, src_points, dst_points, output_size):
    M = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
    return cv2.warpPerspective(image, M, output_size, flags=cv2.INTER_CUBIC), M


def unwarp_document(image):
    tilted_quad = find_quad(image)
    width, height = quad_size(tilted_quad)
    rectangle = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    flat_image, M = apply_perspective_transform(image, tilted_quad, rectangle, (width, height))
    return flat_image, M, tilted_quad, rectangle


tilted = cv2.imread("images/doc_tilted.jpg")
flat_reference = cv2.imread("images/doc_flat_reference.jpg")
corrected, M, tilted_quad, rectangle = unwarp_document(tilted)

restored = cv2.warpPerspective(corrected, np.linalg.inv(M), (tilted.shape[1], tilted.shape[0]), flags=cv2.INTER_CUBIC)
inside = np.zeros(tilted.shape[:2], np.uint8)
cv2.fillPoly(inside, [np.int32(tilted_quad)], 255)
inside = cv2.erode(inside, np.ones((7, 7), np.uint8)) > 0

marked = tilted.copy()
cv2.polylines(marked, [np.int32(tilted_quad)], True, (0, 255, 0), 4)
for pt in tilted_quad:
    cv2.circle(marked, tuple(int(v) for v in pt), 9, (0, 0, 255), -1)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
show(axes[0], marked, "Tilted view with detected corners")
show(axes[1], corrected, "Unwarped (head-on)")
show(axes[2], restored, "Inverse mapping back to tilted view")
plt.tight_layout()
plt.savefig("output/q2_unwarp.png", dpi=120)
plt.close()
cv2.imwrite("output/q2_unwarped_document.jpg", corrected)
cv2.imwrite("output/q2_tilted_document.jpg", tilted)

mae_flat = np.mean(cv2.absdiff(cv2.resize(corrected, (700, 500)), flat_reference))
mae_back = np.mean(cv2.absdiff(restored, tilted)[inside])
print("Source corners (tilted view, TL TR BR BL):\n", np.round(tilted_quad, 1))
print("Destination corners (rectangle):\n", rectangle)
print("Perspective matrix M:\n", np.round(M, 5))
print("Output size:", quad_size(tilted_quad))
print("Mean absolute difference from the true flat page:", round(float(mae_flat), 2), "/ 255")
print("Round trip error inside the page after the inverse mapping:", round(float(mae_back), 2), "/ 255")
