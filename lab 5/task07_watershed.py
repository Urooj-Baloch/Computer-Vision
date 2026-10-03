"""Task 7 - Separating touching coins with marker-based watershed."""
from watershed_pipeline import *

img = load("water_coins.jpg")
DIST_FRAC = 0.5          # sure-foreground threshold = 0.5 * max(distance transform)
r = run_watershed(img, DIST_FRAC)

# number of connected white regions in the plain threshold (what a naive count would use)
n_naive = cv2.connectedComponents(r["opening"])[0] - 1
log("task07", ["TASK 7 - marker-based watershed",
               f"Pre-processing: Gaussian 5x5; Otsu inverse threshold; opening 3x3 x2; sure-bg dilate 3x; distance transform L2/5; sure-fg = {DIST_FRAC} * max",
               f"Connected white regions in thresholded mask (naive count): {n_naive}",
               f"Foreground markers found: {r['n_fg_markers']}",
               f"Regions after watershed (separated coins): {r['n_regions']}"])

panel([img, r["thresh"], r["sure_bg"], dist_vis(r["dist"]), r["sure_fg"], r["unknown"], r["overlay"]],
      ["Original", "Threshold", "Sure Background", "Distance Transform", "Sure Foreground", "Unknown Region", "Final Watershed (red boundaries)"],
      "task07_watershed_stages.png", cols=4, cell=(3.2, 3.9))
save("task07_final_watershed.png", r["overlay"])
save("task07_final_numbered.png", numbered_overlay(r, img))
