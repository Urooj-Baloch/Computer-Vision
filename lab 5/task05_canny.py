"""Task 5 - Boundary detection with Canny (two-threshold hysteresis)."""
from common import *

img = load("smarties.png")
gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 1.4)

PAIRS = [(20, 60), (60, 150), (150, 300)]       # (low, high): permissive, medium, strict
edges = [cv2.Canny(gray, lo, hi) for lo, hi in PAIRS]

# gradient magnitude as Canny sees it (L1 norm of Sobel 3x3) -> used to label strong / weak edge pixels
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
mag = np.abs(gx) + np.abs(gy)

def comp_count(e, min_area=0):
    n, _, st, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
    return int(np.sum(st[1:, cv2.CC_STAT_AREA] >= min_area))

lines = ["TASK 5 - Canny results (L1 gradient magnitude, aperture 3, GaussianBlur 5x5 sigma 1.4)"]
for (lo, hi), e in zip(PAIRS, edges):
    strong = int(np.sum((e > 0) & (mag >= hi))); weak = int(np.sum((e > 0) & (mag < hi)))
    lines.append(f"  low={lo:3d} high={hi:3d}: edge px={int(np.sum(e > 0)):5d}  strong px={strong:5d}  weak-but-kept px={weak:5d}  "
                 f"components={comp_count(e)}  tiny comps(<10px)={comp_count(e) - comp_count(e, 10)}")
# edges present in the permissive result but absent in the strict one = "missing" edges for the strict setting
missing = (edges[0] > 0) & (edges[2] == 0)
lines.append(f"  edge px present with ({PAIRS[0]}) but lost with ({PAIRS[2]}): {int(missing.sum())}")
log("task05", lines)

# colour-coded view of the permissive result: strong (white) vs weak (cyan); tiny isolated pieces (red) = likely unwanted noise
lo, hi = PAIRS[0]
vis = np.zeros((*gray.shape, 3), np.uint8)
e0 = edges[0] > 0
vis[e0 & (mag >= hi)] = (255, 255, 255)
vis[e0 & (mag < hi)] = (255, 255, 0)           # BGR cyan
n, lab, st, _ = cv2.connectedComponentsWithStats(edges[0], connectivity=8)
for i in range(1, n):
    if st[i, cv2.CC_STAT_AREA] < 10:
        vis[lab == i] = (0, 0, 255)

footer = ("Raising the HIGH threshold makes it harder to start an edge (only very strong gradients become seeds), so faint edges and noise disappear first and finally real boundaries break up. "
          "Lowering the LOW threshold lets edges continue along weak gradients (hysteresis), giving longer connected contours but also more texture/shadow edges. "
          "Strong edges = above high; weak edges = between low and high and kept only if connected to a strong one; missing = real boundary lost because high is too big; unwanted = isolated noise/shadow edges when low/high are too small.")
panel([img] + edges + [vis], ["Original"] + [f"Edge result {i+1}: low={lo}, high={hi}" for i, (lo, hi) in enumerate(PAIRS)] + ["Result 1 coded: white=strong, cyan=weak, red=tiny noise"],
      "task05_canny_comparison.png", cols=5, cell=(3.4, 3.0), footer=footer)

# extra experiment: fixed low threshold, increasing high threshold
fixed = [(50, h) for h in (100, 200, 300)]
panel([img] + [cv2.Canny(gray, lo, hi) for lo, hi in fixed], ["Original"] + [f"low={lo}, high={hi}" for lo, hi in fixed],
      "task05_fixed_low_varying_high.png", cols=4, cell=(3.6, 3.2), suptitle="Low fixed at 50, high increased")
for i, e in enumerate(edges, 1):
    save(f"task05_edges_{i}.png", e)
