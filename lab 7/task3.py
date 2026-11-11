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


COLS, ROWS = 9, 6
objp = np.zeros((COLS * ROWS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

obj_pts, img_pts, kept = [], [], []
for path in sorted(glob.glob("calib/*.jpg")):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    found, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        obj_pts.append(objp)
        img_pts.append(corners)
        kept.append(path)
img_size = (gray.shape[1], gray.shape[0])
print(f"Chessboard found in {len(kept)}/{len(glob.glob('calib/*.jpg'))} images")

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)
np.savez("output/camera_calibration.npz", camera_matrix=K, dist_coeffs=dist, rms=rms)

per_image = []
for o, ip, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
    proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
    per_image.append(np.sqrt(np.mean(np.sum((proj - ip) ** 2, axis=2))))

K_gt = np.array([[700.0, 0, 480.0], [0, 700.0, 360.0], [0, 0, 1]])
D_gt = np.array([-0.28, 0.12, 0.0, 0.0, -0.03])
print("RMS reprojection error (px):", round(float(rms), 4))
print("Estimated camera matrix:\n", np.round(K, 3))
print("Known camera matrix:\n", K_gt)
print("Estimated distortion [k1 k2 p1 p2 k3]:", np.round(dist.ravel(), 4))
print("Known distortion      [k1 k2 p1 p2 k3]:", D_gt)

first = cv2.imread(kept[0])
drawn = first.copy()
cv2.drawChessboardCorners(drawn, (COLS, ROWS), img_pts[0], True)
newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, img_size, 0, img_size)
first_fixed = cv2.undistort(first, K, dist, None, newK)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
show(axes[0, 0], drawn, "Detected chessboard corners")
axes[0, 1].bar(range(1, len(per_image) + 1), per_image, color="tab:blue")
axes[0, 1].axhline(rms, color="red", linestyle="--", label=f"RMS {rms:.3f}px")
axes[0, 1].set_title("Reprojection error per image")
axes[0, 1].set_xlabel("image")
axes[0, 1].set_ylabel("pixels")
axes[0, 1].legend()
show(axes[1, 0], first, "Calibration image (distorted)")
show(axes[1, 1], first_fixed, "After cv2.undistort")
plt.tight_layout()
plt.savefig("output/q3_calibration.png", dpi=120)
plt.close()


def undistort_stream(source=0, show_window=False, max_frames=None, save_to=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (fw, fh), 0, (fw, fh))
    writer = None
    if save_to:
        writer = cv2.VideoWriter(save_to, cv2.VideoWriter_fourcc(*"mp4v"), 20, (fw * 2, fh))
    samples = []
    count = 0
    start = time.time()
    while max_frames is None or count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fixed = cv2.undistort(frame, K, dist, None, new_K)
        pair = np.hstack([frame, fixed])
        cv2.putText(pair, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(pair, "Undistorted", (fw + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 160, 0), 2)
        if writer is not None:
            writer.write(pair)
        if count % 30 == 0:
            samples.append(pair.copy())
        if show_window:
            cv2.imshow("Original | Undistorted (press q to quit)", pair)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        count += 1
    cap.release()
    if writer is not None:
        writer.release()
    return count, count / max(time.time() - start, 1e-6), samples


probe = cv2.VideoCapture(0)
has_camera = probe.isOpened()
probe.release()
source = 0 if has_camera else "videos/distorted_feed.mp4"
print("Source:", "webcam" if has_camera else source)

frames_done, fps, samples = undistort_stream(source, show_window=False, max_frames=300, save_to="output/undistorted_feed.mp4")
if has_camera:
    cv2.destroyAllWindows()
print(f"Processed {frames_done} frames at {fps:.0f} FPS")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
show(axes[0], samples[0], "First frame")
show(axes[1], samples[-1], "Later frame")
plt.tight_layout()
plt.savefig("output/q3_live_undistort.png", dpi=120)
plt.close()


cap = cv2.VideoCapture("videos/distorted_feed.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ok, raw = cap.read()
cap.release()
ideal = cv2.imread("images/feed_ideal_frame_30.jpg")
fixed = cv2.undistort(raw, K, dist, None, K_gt)
region = (slice(100, 620), slice(120, 840))
err_raw = np.mean(cv2.absdiff(raw[region], ideal[region]))
err_fixed = np.mean(cv2.absdiff(fixed[region], ideal[region]))
print(f"Mean absolute difference from the distortion-free frame: raw {err_raw:.2f}, undistorted {err_fixed:.2f}")
print(f"Error reduced by {100 * (1 - err_fixed / err_raw):.1f}%")
