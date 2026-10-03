from common import *

gray = load("sudoku.png", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
BLOCKS = [5, 25, 101]     
CS = [2, 10, 25]          
METHODS = {"Mean": cv2.ADAPTIVE_THRESH_MEAN_C, "Gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}

def noise_blobs(mask, max_area=6):
    """Count tiny isolated black specks (area <= max_area px) -> proxy for 'black noise'."""
    ink = (mask == 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    return int(np.sum(stats[1:, cv2.CC_STAT_AREA] <= max_area))

results, lines = {}, ["TASK 2 - parameter grid (ink% = share of black pixels, specks = tiny isolated black blobs)",
                      f"{'method':9s} {'block':>5s} {'C':>3s} {'ink%':>6s} {'specks':>7s}"]
for mname, m in METHODS.items():
    for b in BLOCKS:
        for c in CS:
            mask = cv2.adaptiveThreshold(blur, 255, m, cv2.THRESH_BINARY, b, c)
            results[(mname, b, c)] = mask
            lines.append(f"{mname:9s} {b:5d} {c:3d} {np.mean(mask == 0) * 100:6.1f} {noise_blobs(mask):7d}")
log("task02", lines)

for mname in METHODS:
    imgs = [gray] + [results[(mname, b, c)] for b in BLOCKS for c in CS]
    titles = ["Original"] + [f"{mname} block={b} C={c}" for b in BLOCKS for c in CS]
    panel(imgs, titles, f"task02_{mname.lower()}_grid.png", cols=5, cell=(3.4, 3.6),
          suptitle=f"Task 2 - {mname}-based adaptive thresholding, blockSize x C")


imgs = [gray] + [results[(m, b, 10)] for m in METHODS for b in BLOCKS]
titles = ["Original"] + [f"{m} block={b}, C=10" for m in METHODS for b in BLOCKS]
panel(imgs, titles, "task02_summary_six_results.png", cols=4, cell=(4, 4.2), suptitle="Task 2 - original + six adaptive results")
