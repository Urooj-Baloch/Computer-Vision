"""Task 1 - Automated inspection of a document under uneven lighting (global vs adaptive threshold)."""
from common import *

gray = load("sudoku.png", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(gray, (3, 3), 0)      # light denoising before thresholding

# three global thresholds (THRESH_BINARY: ink -> 0/black, paper -> 255/white)
T_VALUES = [70, 110, 150]
global_masks = [cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)[1] for t in T_VALUES]

# adaptive threshold: Gaussian-weighted 25x25 neighbourhood, constant C = 10
BLOCK, C = 25, 10
adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BLOCK, C)

# quantify the effect of uneven lighting: ink fraction in left / right thirds of the page
h, w = gray.shape
def ink_fraction(mask, x0, x1):
    return float(np.mean(mask[:, x0:x1] == 0)) * 100
def mean_bright(x0, x1):
    return float(np.mean(gray[:, x0:x1]))
thirds = [(0, w // 3), (w // 3, 2 * w // 3), (2 * w // 3, w)]

lines = ["TASK 1 - parameters and measurements",
         f"Global thresholds tested: {T_VALUES} (on Gaussian-blurred 3x3 grayscale)",
         f"Adaptive: ADAPTIVE_THRESH_GAUSSIAN_C, blockSize={BLOCK}, C={C}",
         "Mean paper brightness left/middle/right third: " + ", ".join(f"{mean_bright(*t):.0f}" for t in thirds),
         "Ink (black) % in left/middle/right third:"]
for name, m in [(f"Global T={t}", m) for t, m in zip(T_VALUES, global_masks)] + [("Adaptive", adaptive)]:
    lines.append(f"  {name:14s}: " + ", ".join(f"{ink_fraction(m, *t):5.1f}" for t in thirds))
log("task01", lines)

footer = ("Why does a single threshold struggle when illumination changes across the image?\n"
          "Because one fixed value compares every pixel with the same intensity level, but the paper itself is darker on one side than on the other. "
          "A low T keeps the bright side clean but erases text/grid in the shadow; a high T keeps the text in the shadow but turns the shadowed paper into black blobs.\n"
          "Adaptive thresholding computes T from each pixel's local neighbourhood, so it follows the lighting.")
panel([gray] + global_masks + [adaptive],
      ["Original (gray)"] + [f"Global T{i+1} = {t}" for i, t in enumerate(T_VALUES)] + [f"Adaptive (Gaussian, {BLOCK}, C={C})"],
      "task01_comparison.png", cols=5, cell=(3.6, 3.8), footer=footer)
save("task01_final_adaptive_mask.png", adaptive)
