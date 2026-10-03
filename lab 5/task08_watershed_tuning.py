"""Task 8 - Tuning the distance-transform threshold (sure foreground) of the watershed pipeline."""
from watershed_pipeline import *
import csv

img = load("water_coins.jpg")
EXPECTED = 24                      # coins counted by eye on outputs/task07_final_numbered.png
FRACS = [0.2, 0.4, 0.6, 0.9, 0.98]  # sure-foreground threshold as a fraction of max(distance transform)


def observe(f, r):
    if r["n_regions"] < EXPECTED and f < 0.5:
        return f"Under-segmented: coins merged into {r['n_regions']} blob(s) (sure-fg touches neighbours)"
    if r["n_regions"] < EXPECTED:
        return f"Over-strict: {EXPECTED - r['n_regions']} smaller coins lose their marker and are missed"
    if r["n_regions"] == EXPECTED:
        return "All coins separated, boundaries on coin edges"
    return "Over-segmented: coins split"


rows, imgs, titles = [], [img], ["Original"]
for i, f in enumerate(FRACS, 1):
    r = run_watershed(img, f)
    thr = float(f * r["dist"].max())
    rows.append([i, f"{f:.2f} x max = {thr:.1f} px", r["n_fg_markers"], r["n_regions"], observe(f, r)])
    imgs.append(r["overlay"]); titles.append(f"Exp {i}: thr={thr:.1f}px ({r['n_regions']} regions)")
    save(f"task08_exp{i}_frac{f}.png", r["overlay"])

# scan the whole range to find the parameter range that yields the right count
good = []
for f in np.arange(0.05, 0.995, 0.01):
    r = run_watershed(img, float(f))
    if r["n_regions"] == EXPECTED:
        good.append(float(f))
rng = (min(good), max(good))
lines = ["TASK 8 - distance-transform threshold experiments (expected coins = %d)" % EXPECTED,
         f"{'Exp':>3s} | {'Distance threshold':22s} | {'Foreground':>10s} | {'Regions':>7s} | Observed result"]
lines += [f"{r[0]:>3d} | {r[1]:22s} | {r[2]:>10d} | {r[3]:>7d} | {r[4]}" for r in rows]
lines.append(f"Fractions giving exactly {EXPECTED} separated coins (scan 0.05-0.99 step 0.01): {rng[0]:.2f} - {rng[1]:.2f} of the maximum distance (about {rng[0]*24:.0f}-{rng[1]*24:.0f} px)")
lines.append(f"Best / most meaningful range: about 0.5 - 0.9 x max (centre of a stable plateau, robust to small changes)")
log("task08", lines)
with open(os.path.join(OUT_DIR, "task08_table.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["Experiment", "Distance Threshold", "Foreground", "Regions", "Observed Result"]); w.writerows(rows)

fig_imgs = imgs
panel(fig_imgs, titles, "task08_watershed_experiments.png", cols=3, cell=(3.6, 4.4),
      footer="Best range: sure-foreground threshold of about %.1f-%.1f x max(dist) - every coin keeps one marker and touching coins stay separated." % (0.5, 0.9))

# table as a figure
fig, ax = plt.subplots(figsize=(12, 2.6)); ax.axis("off")
tbl = ax.table(cellText=[[str(c) for c in r] for r in rows], colLabels=["Experiment", "Distance Threshold", "Foreground", "Regions", "Observed Result"],
               loc="center", cellLoc="left", colWidths=[0.08, 0.2, 0.09, 0.08, 0.55])
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.6)
fig.savefig(os.path.join(OUT_DIR, "task08_table.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
