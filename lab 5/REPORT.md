# Lab 05 – Image Segmentation (Tasks 1–10)

All code uses **Python, OpenCV and NumPy only** (Matplotlib is used just to lay out the comparison figures). No prebuilt
segmentation library or deep-learning model is used. Every figure/mask is saved in `outputs/`, and the exact parameters
and measurements of each task are in `outputs/taskNN_log.txt`.

## Images used

| Task | Image | Source |
|---|---|---|
| 1, 2 | `sudoku.png` | OpenCV `samples/data` |
| 3, 7, 8 | `water_coins.jpg` | OpenCV watershed tutorial image |
| 4, 5, 10 | `smarties.png` | OpenCV `samples/data` |
| 6 | `brain.png` | Brain MRI slice taken from the `MR_small.dcm` test file shipped with `pydicom` (64×64, up-scaled ×2). The article figure from the manual was not reachable, so this is a substitute. |
| 9 | `messi5.jpg` | OpenCV `samples/data` (substitute for `dog.jpeg`: the Packt link returned 404) |

`prepare_images.py` re-creates the `images/` folder.

---

## Task 1 – Uneven lighting (global vs adaptive)  
Figure: `outputs/task01_comparison.png` (Original → Global T1/T2/T3 → Adaptive)

**Parameters.** Grayscale + 3×3 Gaussian blur. Global thresholds **70, 110, 150**. Adaptive: `ADAPTIVE_THRESH_GAUSSIAN_C`, blockSize **25**, C **10**.
Measured paper brightness: 81 (left) / 99 (middle) / 125 (right third), so the lighting really does change across the page.

| Ink (black) % | left | middle | right |
|---|---|---|---|
| Global 70 | 36.0 | 13.1 | 5.7 |
| Global 110 | 88.1 | 61.6 | 24.0 |
| Global 150 | 98.9 | 100.0 | 92.5 |
| **Adaptive** | 11.7 | 15.3 | 16.8 |

**Where global thresholding fails.** The dark/shadowed left and lower-left part of the page (and the dark top-left corner): T=70 already turns the lower-left paper
into a black blob; T=110 and T=150 blacken most of the page. At T=70 the bright right side is fine but is the best a single value can do, so **T=70 is the best global result**.
**Best overall: adaptive Gaussian, block 25, C 10** – ink share is almost constant across the page (11.7–16.8 %), digits and grid lines are intact on both sides.

**Why does a single threshold struggle when illumination changes across the image?** One fixed value compares every pixel against the same intensity level, but the paper itself is
darker on one side than the other. A threshold low enough for the shadowed side is not reached by the text there, and one high enough to keep that text also swallows the shadowed paper. Adaptive thresholding computes the
threshold from each pixel's local neighbourhood, so it follows the lighting.

## Task 2 – Adaptive strategy: blockSize, C, Mean vs Gaussian  
Figures: `task02_summary_six_results.png` (original + 6), `task02_mean_grid.png`, `task02_gaussian_grid.png` (full 3×3 grids = 18 results each method)
Tested blockSize **5, 25, 101** × C **2, 10, 25** × {Mean, Gaussian}. Speck = tiny isolated black blob (≤ 6 px), a proxy for black noise (`outputs/task02_log.txt`).

1. **Neighbourhood too small (5):** the window is about the size of a stroke, so only stroke *edges* survive – digits become hollow outlines/broken, and flat paper texture turns into noise (Mean 5/C2: 1997 specks; Gaussian 5/C10: 491 specks).
2. **Neighbourhood too large (101):** it behaves more like a global threshold – large dark regions (the "SUDOKU" label, picture at the top, page edge) fill in as solid black blobs and the thick strokes get heavier.
3. **C increased:** the threshold moves further below the local average, so fewer pixels count as ink: noise disappears (good) but thin strokes vanish (bad). Gaussian 5/C25 leaves 0.0 % ink; Mean 25: ink 26.3 % (C2) → 17.7 % (C10) → 9.5 % (C25, thin lines broken).
4. **Cleanest segmentation:** blockSize **25 with C = 10** (about 40 specks, ink 14–18 %, characters complete). Small C (2) is noisy, large C (25) breaks digits.
5. **Mean vs Gaussian:** very close on this image. Mean 25/C10 has slightly fewer specks (32 vs 40) but draws strokes a bit thicker (17.7 % vs 14.6 % ink); Gaussian weights nearby pixels more, giving crisper, thinner strokes. I choose **Gaussian 25/C10** as slightly better, but the difference is small.

## Task 3 – Otsu thresholding  
Figures: `task03_original_hist_otsu.png` (Original → Histogram → Otsu mask), `task03_threshold_comparison.png`

The histogram of `water_coins.jpg` is clearly bimodal (coins ≈ 90, paper ≈ 240). Otsu was called with `THRESH_BINARY_INV + THRESH_OTSU`; the threshold is *not* chosen by hand.

| Preprocessing | Otsu T |
|---|---|
| Plain grayscale | **162** |
| Gaussian blur 5×5 | 162 |
| Darker (β = −50) | 112 |
| Low contrast (α = 0.6) | 97 |
| CLAHE | 158 |

When the brightness/contrast changes, Otsu moves its threshold with the histogram (162 → 112 → 97) and the resulting coin mask stays practically the same (foreground 55.9 % in all three). A fixed value such as 162 would have failed on the darkened/low-contrast versions.

**Why is Otsu useful when the programmer does not know the threshold beforehand?** It tests all 256 levels and selects the one that best separates the histogram into two classes (minimum within-class variance), so the threshold adapts automatically to every image's brightness and contrast, without manual tuning per image.

## Task 4 – Colour sorting in HSV  
Figure: `task04_hsv_comparison.png`, masks `task04_final_mask.png`, `task04_final_extracted.png`. Target colour: **green** smarties.

| Mask | lower (H,S,V) | upper (H,S,V) | result |
|---|---|---|---|
| Too restrictive | (58, 230, 230) | (64, 255, 255) | 61 px, 0 objects |
| **Final** | (40, 80, 60) | (85, 255, 255) | 5385 px, 3 objects (2 whole candies + 1 cut by the border) |

Final mask: opening then closing with a 5×5 ellipse (removes specks, fills the small specular-highlight holes). Green hue is ≈ 60 in OpenCV scale; S ≥ 80 rejects the white background (S ≈ 0); V ≥ 60 rejects shadows/dark pixels.

**Why the restrictive mask fails.** Its hue, saturation and value windows are so narrow that they accept only pixels whose colour equals the candy's base colour almost exactly. On a curved glossy candy, shading lowers V and S and the highlight raises V while lowering S, so most of the candy falls outside the window – only a few fragments are found. The wider range captures the whole object while still excluding the white background.

## Task 5 – Canny edges  
Figures: `task05_canny_comparison.png` (Original → Edge 1/2/3 + colour-coded view), `task05_fixed_low_varying_high.png`  
Image `smarties.png`, 5×5 Gaussian blur (σ 1.4) before Canny. Pairs (low, high): **(20, 60)**, **(60, 150)**, **(150, 300)**; extra experiment: low = 50 fixed, high = 100/200/300.

| (low, high) | edge px | strong px | weak-but-kept px | components | tiny comps (<10 px) |
|---|---|---|---|---|---|
| (20, 60) | 3535 | 3403 | 132 | 64 | 17 |
| (60, 150) | 3047 | 2857 | 190 | 32 | 1 |
| (150, 300) | 2615 | 1953 | 662 | 20 | 0 |

* **Strong edges:** gradient ≥ high – the candy outlines (white in the colour-coded panel).
* **Weak edges:** between low and high; kept only because they connect to a strong edge (hysteresis) – short arcs on candy highlights/shading (cyan).
* **Missing edges:** with (150, 300), 920 edge pixels that exist in Result 1 are gone – parts of lower-contrast candy outlines break up, so the boundaries are incomplete.
* **Unwanted edges:** with the permissive pair, 17 tiny isolated fragments (red) plus small loops around the specular highlights inside the candies and short shadow arcs beside them.

**How the thresholds change the result.** Raising *high* with *low* fixed removes the weakest seeds first (noise, then highlights, then genuine faint edges), so the result gets cleaner and then starts to lose real boundary. Lowering *low* lets edges continue along weak gradients – longer connected contours, but more texture and shadow edges. A balanced pair such as (60, 150) gave closed outlines with almost no noise (1 tiny component).

## Task 6 – Region growing  
Figure: `task06_region_growing.png`; masks `task06_mask_seed*_T*.png`  
Own implementation: 8-connected BFS from the seed; a neighbour joins if |I − I_seed| ≤ T (I_seed = mean of the 3×3 window around the seed; image smoothed with 3×3 Gaussian). Thresholds **10, 25, 45**, seeds **A (100,75)** bright tissue and **B (118,45)** grey region.

| Seed | T = 10 | T = 25 | T = 45 |
|---|---|---|---|
| A (ref ≈ 152) | 350 px | 727 px | 2360 px |
| B (ref ≈ 127) | 570 px | 1531 px | 2639 px |

Small T under-segments (region is fragmented), T = 25 gives a compact region for seed A, and T = 45 leaks into neighbouring tissue.
*Note:* the MRI is a small, low-resolution slice (see above), so the regions are not anatomical structures – the experiment shows the method's behaviour.

**Why can changing the seed point significantly change the final region?** Region growing only adds pixels that are connected to the seed *and* close to the seed's intensity. A different seed has a different reference intensity and lies in a different connected area, so a different set of pixels is accepted (here seed A vs B give very different shapes and sizes). Unlike a global threshold, the result depends on where you start.

## Task 7 – Marker-based watershed  
Figure: `task07_watershed_stages.png` (Original → Threshold → Sure Background → Distance Transform → Sure Foreground → Unknown Region → Final Watershed), `task07_final_numbered.png`  
Pipeline: Gaussian 5×5 → Otsu (inverse) → opening 3×3 ×2 → sure background (dilate ×3) → distance transform (L2, mask 5) → sure foreground = 0.5 × max → unknown = sure bg − sure fg → connected-component markers (+1, unknown = 0) → `cv2.watershed` → boundaries in red.

The thresholded mask is **1** connected white region (all coins touching), so counting regions directly gives 1. The watershed produces **24 regions**, and on visual inspection of the numbered output each of the 24 coins has exactly one region and the boundaries follow the coin edges.

## Task 8 – Tuning the distance-transform threshold  
Figures: `task08_watershed_experiments.png`, `task08_table.png`, `outputs/task08_table.csv`

| Experiment | Distance threshold | Foreground | Regions | Observed result |
|---|---|---|---|---|
| 1 | 0.20 × max = 4.8 px | 1 | 1 | Everything merged into one blob |
| 2 | 0.40 × max = 9.6 px | 7 | 7 | Clusters of coins merged (under-segmented) |
| 3 | 0.60 × max = 14.4 px | 24 | 24 | All coins separated |
| 4 | 0.90 × max = 21.6 px | 24 | 24 | All coins separated |
| 5 | 0.98 × max = 23.5 px | 20 | 20 | 4 smaller coins lose their marker and are missed |

Expected count: 24 (counted by eye). Scanning 0.05–0.99 in steps of 0.01, exactly 24 coins are obtained for **0.45–0.97 × max (≈ 11–23 px)**; the most meaningful and robust range is about **0.5–0.9 × max**, in the middle of that plateau.
Observed trade-off: too low → sure foreground of neighbouring coins touches → merged objects; too high → small coins shrink below the threshold and vanish. On this image I did not see coins being *split* in two at high thresholds – they were lost instead.

## Task 9 – K-Means  
Figure: `task09_kmeans_comparison.png` (Original → K=2 → K=4 → K=6). Procedure: reshape pixels to N×3, convert to `float32`, `cv2.kmeans` (20 iterations / ε 0.5, 5 attempts, k-means++ start, fixed RNG seed), reconstruct using the cluster centres.

| K | clusters | unique colours (original 59 147) | PSNR | major regions represented |
|---|---|---|---|---|
| 2 | 2 | 2 | 17.5 dB | dark (crowd, shadows, hair, shirt) vs olive-grey (pitch + bright parts); player almost lost |
| 4 | 4 | 4 | 20.9 dB | dark crowd (36 %), purple-grey mid-tones: jersey/skin/crowd (31 %), **green pitch (20 %)**, light grey stands/boots/ball (12 %) |
| 6 | 6 | 6 | 22.7 dB | adds a separate **red** cluster (4 %) that shows the red stripes of the jersey, and splits dark/mid-grey crowd levels |

Higher K keeps more of the original colours (visually closer, higher PSNR) and shows smaller objects such as the jersey stripes, but simplifies less; K = 2 is the strongest simplification and loses the player.

## Task 10 – Segmentation engineer: three approaches on `smarties.png`  
Goal: segment all candies (14 are visible, 2 cut by the border). Figure: `task10_final_comparison.png` (Original | Method 1 | Method 2 | Method 3)

| | Method 1 – Otsu (gray, blur 5) | Method 2 – HSV colour threshold (S ≥ 60, V ≥ 30, open+close) | Method 3 – K-Means (K = 4; background = brightest cluster) |
|---|---|---|---|
| Assumption | Histogram is bimodal: dark objects on a brighter background | Objects are more saturated than the near-white background | Pixels form a few distinct colour groups; background is the lightest group |
| Identified successfully | All candies (T = 177), 13 objects – separated several touching candies | All candies as solid blobs, no holes (1 hole) | All candies, comparable to HSV (2 holes) |
| Incorrectly segmented | 14 interior holes where glossy highlights are brighter than T; a stray speck; shadow halo next to some candies | The touching bottom-left cluster merges into one blob (11 objects) with a ragged edge | Same merge as HSV; small notches on the dark candy |
| Parameter with greatest effect | Threshold value: −40 → IoU 0.73 with the default mask | S/V lower bounds: V_min 80–120 or S_min 150 → IoU 0.91 | K: K = 6 → IoU 0.96 (seed has no effect, IoU 1.00) |

Agreement between methods is high (IoU 0.94–0.96), so the background/candy split is easy for this image.

**Conclusion.** The **HSV colour threshold** produced the most meaningful regions: each candy is a complete, hole-free blob (1 interior hole vs 14 for Otsu), shadows are ignored, and the colour channel could further be used to sort candies by colour. This works because the image has a white, desaturated background and strongly saturated objects – a property of colour, not of grey-level intensity (glossy highlights break the grey-level assumption of Otsu). Otsu gave the object count closest to the 14 candies (13 vs 11) because it split some touching candies, but its masks are full of highlight holes. None of the three methods reliably separates touching candies; for that a watershed on the distance transform (Tasks 7–8) would be the next step.
