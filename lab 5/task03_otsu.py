from common import *

img = load("water_coins.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def otsu(g):
    t, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   
    return t, m

def hist_image(g, thr, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.hist(g.ravel(), bins=256, range=(0, 256), color="gray")
    ax.axvline(thr, color="r", label=f"Otsu T = {thr:.0f}")
    ax.set_title(title, fontsize=9); ax.legend(); ax.set_xlabel("intensity"); ax.set_ylabel("pixels")
    fig.tight_layout(); fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
variants = [
    ("Plain grayscale", gray),
    ("Gaussian blur 5x5", cv2.GaussianBlur(gray, (5, 5), 0)),
    ("Darker (beta = -50)", cv2.convertScaleAbs(gray, alpha=1.0, beta=-50)),
    ("Low contrast (alpha = 0.6)", cv2.convertScaleAbs(gray, alpha=0.6, beta=0)),
    ("CLAHE", clahe.apply(gray)),
]
lines = ["TASK 3 - Otsu automatically selected thresholds"]
rows = []
for name, g in variants:
    t, m = otsu(g)
    lines.append(f"  {name:22s}: T = {t:6.1f}   foreground = {np.mean(m == 255) * 100:5.1f}%")
    rows.append((name, g, t, m))
log("task03", lines)
name, g, t, m = rows[0]
footer = ("Why is Otsu useful when the programmer does not know the correct threshold beforehand?\n"
          "Otsu searches all 256 levels and picks the one that best separates the histogram into two classes (it minimises within-class variance), "
          "so the threshold adapts to every image's brightness/contrast instead of being hand-tuned.")
panel([img, hist_image(g, t, "Histogram of grayscale image"), m], ["Original", "Histogram", f"Otsu binary mask (T={t:.0f})"],
      "task03_original_hist_otsu.png", cols=3, footer=footer)
save("task03_otsu_mask.png", m)
imgs, titles = [], []
for name, g, t, m in rows:
    imgs += [hist_image(g, t, name), m]
    titles += [f"{name}: histogram", f"mask, T={t:.0f}"]
panel(imgs, titles, "task03_threshold_comparison.png", cols=4, cell=(3.4, 3.4), suptitle="Task 3 - Otsu on differently preprocessed images")
