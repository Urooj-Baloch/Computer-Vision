"""Task 9 - K-Means colour segmentation (K = 2, 4, 6)."""
from common import *

img = load("messi5.jpg")                     # stand-in for dog.jpeg (the Packt link was unreachable)
h, w, _ = img.shape
data = np.float32(img.reshape(-1, 3))        # one 3-D (B,G,R) feature vector per pixel, float32 as cv2.kmeans requires
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)

results, lines = {}, ["TASK 9 - K-Means (criteria: 20 iterations or eps 0.5, attempts=5, KMEANS_PP_CENTERS)"]
for K in (2, 4, 6):
    cv2.setRNGSeed(0)
    compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    seg = centers[labels.flatten()].reshape(img.shape)
    results[K] = seg
    save(f"task09_kmeans_K{K}.png", seg)
    mse = float(np.mean((img.astype(np.float32) - seg.astype(np.float32)) ** 2))
    psnr = 10 * np.log10(255 ** 2 / mse)
    n_orig = len(np.unique(img.reshape(-1, 3), axis=0)); n_seg = len(np.unique(seg.reshape(-1, 3), axis=0))
    share = np.bincount(labels.flatten(), minlength=K) / labels.size * 100
    lines.append(f"K={K}: clusters={K}, unique colours {n_orig} -> {n_seg} ({100 * (1 - n_seg / n_orig):.2f}% fewer), PSNR vs original = {psnr:.1f} dB")
    for k in np.argsort(-share):
        b, g, r = centers[k]
        lines.append(f"      cluster {k}: RGB=({r:3d},{g:3d},{b:3d})  covers {share[k]:5.1f}% of pixels")
log("task09", lines)

panel([img, results[2], results[4], results[6]], ["Original", "K = 2", "K = 4", "K = 6"], "task09_kmeans_comparison.png", cols=4, cell=(4.6, 3.2))
