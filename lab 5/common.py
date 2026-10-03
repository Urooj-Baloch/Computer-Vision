
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "images")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load(name, flag=cv2.IMREAD_COLOR):
    im = cv2.imread(os.path.join(IMG_DIR, name), flag)
    if im is None:
        raise FileNotFoundError(name)
    return im


def to_rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im


def panel(images, titles, out_name, cols=None, cell=(4, 4), footer=None, suptitle=None):
    """Show several images in a grid and save the figure to outputs/<out_name>."""
    n = len(images)
    cols = cols or n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cell[0] * cols, cell[1] * rows + (0.9 if footer else 0.2)))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(to_rgb(im), cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(t, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    if footer:
        fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.08 if footer else 0, 1, 0.95 if suptitle else 0.97), h_pad=2.5)
    fig.subplots_adjust(hspace=0.3)
    path = os.path.join(OUT_DIR, out_name)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def save(name, im):
    cv2.imwrite(os.path.join(OUT_DIR, name), im)


def log(task, lines):
    """Write the recorded parameters / measurements of a task to outputs/taskNN_log.txt and echo them."""
    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, f"{task}_log.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
