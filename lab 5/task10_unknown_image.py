"""Task 10 - Three segmentation approaches on one challenging image (smarties.png): find all candies."""
from common import *

img = load("smarties.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def n_objects(m, min_area=150):
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return int(np.sum(st[1:, cv2.CC_STAT_AREA] >= min_area))


# ---- Method 1: Otsu thresholding on grayscale -------------------------------------------------
def m1(blur=5, offset=0):
    g = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.threshold(g, t + offset, 255, cv2.THRESH_BINARY_INV)[1], t

# ---- Method 2: colour (HSV) thresholding: candies = pixels with enough saturation -------------
def m2(s_min=60, v_min=30):
    m = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([179, 255, 255]))
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k3), cv2.MORPH_CLOSE, k3)

# ---- Method 3: K-Means on colour; candy = every cluster except the brightest (white background) --
def m3(K=4, seed=0):
    cv2.setRNGSeed(seed)
    data = np.float32(img.reshape(-1, 3))
    _, lab, cen = cv2.kmeans(data, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3, cv2.KMEANS_PP_CENTERS)
    bg = int(np.argmax(cen.sum(axis=1)))
    m = np.uint8((lab.flatten() != bg).reshape(gray.shape)) * 255
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k3), cv2.MORPH_CLOSE, k3)


def n_holes(m, min_area=3):
    """Interior holes inside objects (e.g. specular highlights that fell below the threshold)."""
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    return sum(1 for c, hh in zip(cnts, hier[0]) if hh[3] != -1 and cv2.contourArea(c) >= min_area)


def n_specks(m, max_area=60):
    """Tiny stray fragments outside the main candies (shadow / halo leakage)."""
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return int(np.sum((st[1:, cv2.CC_STAT_AREA] > 0) & (st[1:, cv2.CC_STAT_AREA] < max_area)))


M1, T_otsu = m1(); M2 = m2(); M3 = m3()
iou = lambda a, b: float(np.sum((a > 0) & (b > 0)) / max(1, np.sum((a > 0) | (b > 0))))

lines = ["TASK 10 - three methods on smarties.png (14 candies are visible, 2 of them cut by the border)",
         f"Method 1 Otsu(gray, blur 5): T={T_otsu:.0f}, objects={n_objects(M1)}, mask area={np.mean(M1 > 0) * 100:.1f}%",
         f"Method 2 HSV (S>=60, V>=30, open+close 5x5): objects={n_objects(M2)}, mask area={np.mean(M2 > 0) * 100:.1f}%",
         f"Method 3 K-Means (K=4, background = brightest cluster): objects={n_objects(M3)}, mask area={np.mean(M3 > 0) * 100:.1f}%",
         "Interior holes (>=3 px) / stray specks (<60 px): Otsu %d / %d, HSV %d / %d, K-Means %d / %d" % (n_holes(M1), n_specks(M1), n_holes(M2), n_specks(M2), n_holes(M3), n_specks(M3)),
         "Agreement (IoU): M1-M2 %.2f, M1-M3 %.2f, M2-M3 %.2f" % (iou(M1, M2), iou(M1, M3), iou(M2, M3)),
         "", "Parameter sensitivity: IoU of the changed mask with the default mask of the same method (1.00 = no effect)"]
sens = {}
for off in (-40, -20, 20, 40):
    sens[f"Otsu threshold {off:+d}"] = iou(m1(5, off)[0], M1)
for b in (1, 11):
    sens[f"Otsu blur {b}"] = iou(m1(b)[0], M1)
for s in (30, 100, 150):
    sens[f"HSV S_min={s}"] = iou(m2(s_min=s), M2)
for v in (10, 80, 120):
    sens[f"HSV V_min={v}"] = iou(m2(v_min=v), M2)
for K in (2, 3, 6):
    sens[f"K-Means K={K}"] = iou(m3(K), M3)
for sd in (1, 2):
    sens[f"K-Means seed={sd}"] = iou(m3(4, sd), M3)
lines += [f"  {k:22s}: IoU = {v:.2f}" for k, v in sens.items()]
log("task10", lines)

for name, m in (("otsu", M1), ("hsv", M2), ("kmeans", M3)):
    save(f"task10_{name}_mask.png", m)
panel([img, M1, M2, M3], ["Original", "Method 1: Otsu (gray)", "Method 2: HSV colour threshold", "Method 3: K-Means (K=4)"],
      "task10_final_comparison.png", cols=4, cell=(4.2, 3.7))
