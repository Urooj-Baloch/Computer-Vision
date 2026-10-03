
from common import *
from collections import deque

gray = load("brain.png", cv2.IMREAD_GRAYSCALE)
smooth = cv2.GaussianBlur(gray, (3, 3), 0)         


def region_grow(im, seed, thr):
    """8-connected region growing. A neighbour joins if |I(neighbour) - I_seed| <= thr,
    where I_seed is the mean of the 3x3 neighbourhood around the seed."""
    h, w = im.shape
    x0, y0 = seed
    ref = float(im[max(0, y0 - 1):y0 + 2, max(0, x0 - 1):x0 + 2].mean())
    mask = np.zeros((h, w), np.uint8)
    mask[y0, x0] = 255
    q = deque([(x0, y0)])
    while q:
        x, y = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] == 0 and abs(float(im[ny, nx]) - ref) <= thr:
                    mask[ny, nx] = 255
                    q.append((nx, ny))
    return mask, ref


SEEDS = {"A (100,75) bright tissue": (100, 75), "B (118,45) grey region": (118, 45)}
THRS = [10, 25, 45]

lines = ["TASK 6 - region growing (8-connectivity, reference = mean of 3x3 around seed, image smoothed 3x3)"]
imgs, titles = [], []
for sname, seed in SEEDS.items():
    marked = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(marked, seed, 3, (0, 0, 255), -1)
    imgs.append(marked); titles.append(f"Seed {sname}")
    for t in THRS:
        m, ref = region_grow(smooth, seed, t)
        imgs.append(m); titles.append(f"seed {seed}, T={t}: {int(np.sum(m > 0))} px")
        save(f"task06_mask_seed{seed[0]}_{seed[1]}_T{t}.png", m)
        lines.append(f"  seed={seed} ref_intensity={ref:5.1f} T={t:2d} -> region size {int(np.sum(m > 0)):5d} px ({np.mean(m > 0) * 100:4.1f}% of image)")
log("task06", lines)

footer = ("Why can changing the seed point significantly change the final segmented region?\n"
          "Region growing only adds pixels that are connected to the seed AND close to the seed's intensity. A different seed has a different reference intensity "
          "and sits in a different connected area, so it accepts a different set of pixels - the result depends on where you start, unlike a global threshold.\n"
          "A larger intensity threshold lets the region leak into neighbouring tissue; a small one leaves holes/under-segments.")
panel(imgs, titles, "task06_region_growing.png", cols=4, cell=(3.4, 3.4), footer=footer)
