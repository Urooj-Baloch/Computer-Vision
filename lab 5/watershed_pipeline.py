"""Marker-based watershed pipeline (OpenCV watershed tutorial stages), shared by Task 7 and Task 8."""
from common import *


def run_watershed(img, dist_frac=0.5):
    """Return every intermediate stage plus counts. dist_frac = distance-transform threshold as a fraction of its maximum."""
    # 1. preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # 2. thresholding (Otsu, inverted because coins are darker than the paper)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 3. noise removal: morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # 4. sure background: dilate the objects, what is left outside is certainly background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # 5. distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # 6. sure foreground: keep only pixels far from the border (centres of coins)
    _, sure_fg = cv2.threshold(dist, dist_frac * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    # 7. unknown region = sure background - sure foreground
    unknown = cv2.subtract(sure_bg, sure_fg)
    # 8. marker labelling (background must be 1, unknown 0, objects 2..n)
    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    # 9. watershed transformation (-1 marks the boundaries)
    ws = cv2.watershed(img.copy(), markers.copy())
    # 10. boundary visualisation
    overlay = img.copy()
    overlay[ws == -1] = (0, 0, 255)
    regions = [l for l in np.unique(ws) if l > 1]
    return dict(gray=gray, thresh=thresh, opening=opening, sure_bg=sure_bg, dist=dist, sure_fg=sure_fg, unknown=unknown,
                markers=markers, ws=ws, overlay=overlay, n_fg_markers=n_markers - 1, n_regions=len(regions), regions=regions)


def dist_vis(dist):
    return cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def numbered_overlay(r, img):
    """Overlay with region ids written at region centroids (to count coins by eye)."""
    out = r["overlay"].copy()
    for l in r["regions"]:
        ys, xs = np.nonzero(r["ws"] == l)
        cv2.putText(out, str(l - 1), (int(xs.mean()) - 6, int(ys.mean()) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    return out
