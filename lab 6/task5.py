import cv2
import numpy as np
import glob

paths = sorted(glob.glob("images/pano_*.jpg"))
images = [cv2.imread(p) for p in paths]

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()


def stitch_pair(left, right):
    g1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    matches = bf.knnMatch(d2, d1, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    all_pts = np.concatenate([np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2), warped_corners])
    xmin, ymin = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    canvas = cv2.warpPerspective(right, T @ H, (xmax - xmin, ymax - ymin))
    canvas_left = np.zeros_like(canvas)
    canvas_left[-ymin:h1 - ymin, -xmin:w1 - xmin] = left
    mask_left = canvas_left.sum(axis=2) > 0
    mask_right = canvas.sum(axis=2) > 0
    overlap = mask_left & mask_right
    result = canvas.copy()
    result[mask_left] = canvas_left[mask_left]
    if overlap.any():
        cols = np.where(overlap.any(axis=0))[0]
        x0, x1 = cols.min(), cols.max()
        alpha = np.linspace(1, 0, x1 - x0 + 1)[None, :, None]
        blend = canvas_left[:, x0:x1 + 1].astype(np.float32) * alpha + canvas[:, x0:x1 + 1].astype(np.float32) * (1 - alpha)
        region_mask = overlap[:, x0:x1 + 1][..., None]
        result[:, x0:x1 + 1] = np.where(region_mask, blend.astype(np.uint8), result[:, x0:x1 + 1])
    return result, len(good), int(mask.sum())


def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


pano = images[0]
for i in range(1, len(images)):
    pano, g, inl = stitch_pair(pano, images[i])
    pano = crop_black(pano)
    print(f"Stitched image {i + 1}: good matches {g}, inliers {inl}")

cv2.imwrite("output/task5_panorama.jpg", pano)
print("Panorama size:", pano.shape[1], "x", pano.shape[0])
