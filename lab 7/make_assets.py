import cv2
import numpy as np
import os

for d in ["images", "images/colmap_input", "calib", "videos", "output"]:
    os.makedirs(d, exist_ok=True)


def textured(h, w, seed=0):
    r = np.random.default_rng(seed)
    base = r.integers(0, 255, (h, w, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2.2)
    base = cv2.normalize(base, None, 30, 235, cv2.NORM_MINMAX)
    for _ in range(max(10, (h * w) // 6000)):
        c = tuple(int(x) for x in r.integers(0, 255, 3))
        p1 = (int(r.integers(0, w)), int(r.integers(0, h)))
        p2 = (int(r.integers(0, w)), int(r.integers(0, h)))
        k = r.integers(0, 3)
        if k == 0:
            cv2.rectangle(base, p1, p2, c, int(r.integers(1, 4)))
        elif k == 1:
            cv2.circle(base, p1, int(r.integers(4, 30)), c, int(r.integers(1, 4)))
        else:
            cv2.line(base, p1, p2, c, int(r.integers(1, 4)))
    return base


sample = np.full((400, 600, 3), 245, np.uint8)
for x in range(0, 600, 50):
    cv2.line(sample, (x, 0), (x, 399), (210, 210, 210), 1)
for y in range(0, 400, 50):
    cv2.line(sample, (0, y), (599, y), (210, 210, 210), 1)
cv2.rectangle(sample, (60, 60), (260, 200), (200, 80, 40), 4)
cv2.circle(sample, (430, 140), 70, (40, 120, 220), -1)
cv2.line(sample, (40, 330), (560, 330), (30, 30, 30), 5)
cv2.line(sample, (40, 270), (560, 270), (30, 30, 30), 5)
cv2.putText(sample, "FAST NUCES", (150, 380), cv2.FONT_HERSHEY_DUPLEX, 1.6, (20, 20, 160), 3)
cv2.imwrite("images/sample.jpg", sample)


def make_page(w, h, title, seed):
    r = np.random.default_rng(seed)
    page = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(page, (0, 0), (w - 1, h - 1), (190, 190, 190), 8)
    cv2.putText(page, title, (40, 80), cv2.FONT_HERSHEY_DUPLEX, 1.6, (20, 20, 20), 3)
    cv2.line(page, (40, 100), (w - 40, 100), (30, 30, 30), 3)
    words = ["vision", "camera", "homography", "affine", "matrix", "plane", "corner", "pixel", "scene", "image"]
    for i in range(9):
        y = 150 + i * 36
        text = " ".join(r.choice(words, 5))
        cv2.putText(page, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 1)
    block = textured(170, 210, seed + 5)
    page[190:360, w - 270:w - 60] = block
    cv2.rectangle(page, (w - 270, 190), (w - 60, 360), (0, 0, 0), 3)
    cv2.circle(page, (120, h - 90), 40, (40, 40, 200), -1)
    cv2.rectangle(page, (200, h - 130), (330, h - 50), (200, 100, 30), -1)
    return page


def place(page, quad, bg_seed, size=(900, 700)):
    bg = textured(size[1], size[0], bg_seed) // 5 + 10
    bg = bg.astype(np.uint8)
    h, w = page.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, np.float32(quad))
    warped = cv2.warpPerspective(page, M, size)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, size)
    out = bg.copy()
    out[mask > 0] = warped[mask > 0]
    return out


page_book = make_page(700, 500, "Computer Vision", 1)
cv2.imwrite("images/book_flat_reference.jpg", page_book)
book = place(page_book, [(180, 90), (760, 150), (700, 620), (120, 540)], 2)
cv2.imwrite("images/book_tilted.jpg", book)
page_doc = make_page(700, 500, "Lab Report", 3)
cv2.imwrite("images/doc_flat_reference.jpg", page_doc)
doc = place(page_doc, [(250, 60), (780, 110), (840, 600), (110, 640)], 4)
cv2.imwrite("images/doc_tilted.jpg", doc)

W, H = 960, 720
K = np.array([[700.0, 0, 480.0], [0, 700.0, 360.0], [0, 0, 1]])
D = np.array([-0.28, 0.12, 0.0, 0.0, -0.03])
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
pts = np.stack([gx, gy], axis=-1).reshape(-1, 1, 2)
und = cv2.undistortPoints(pts, K, D, P=K).reshape(H, W, 2)
mapx = np.ascontiguousarray(und[..., 0]).astype(np.float32)
mapy = np.ascontiguousarray(und[..., 1]).astype(np.float32)


def distort(clean):
    return cv2.remap(clean, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))


board = np.full((440, 560, 3), 255, np.uint8)
for i in range(2, 9):
    for j in range(2, 12):
        if (i + j) % 2 == 0:
            board[i * 40:(i + 1) * 40, j * 40:(j + 1) * 40] = 0
S = np.array([[1 / 40.0, 0, -7.0], [0, 1 / 40.0, -5.5], [0, 0, 1]])
rng = np.random.default_rng(5)
for n in range(20):
    rvec = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)])
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(15, 22)])
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
    clean = cv2.warpPerspective(board, Hm, (W, H), borderValue=(128, 128, 128))
    cv2.imwrite(f"calib/calib_{n + 1:02d}.jpg", distort(clean))


def clean_scene(i):
    img = np.full((H, W, 3), 235, np.uint8)
    for x in range(0, W, 60):
        cv2.line(img, (x, 0), (x, H - 1), (60, 60, 60), 2)
    for y in range(0, H, 60):
        cv2.line(img, (0, y), (W - 1, y), (60, 60, 60), 2)
    cv2.rectangle(img, (200, 150), (500, 380), (200, 90, 30), 4)
    cv2.rectangle(img, (560, 300), (860, 600), (30, 120, 200), 4)
    cx = int(100 + (i / 90.0) * 760)
    cy = int(360 + 150 * np.sin(i / 10.0))
    cv2.circle(img, (cx, cy), 45, (30, 30, 220), -1)
    cv2.putText(img, "LIVE FEED", (330, 90), cv2.FONT_HERSHEY_DUPLEX, 1.8, (20, 20, 20), 3)
    return img


cv2.imwrite("images/feed_ideal_frame_30.jpg", clean_scene(30))
vw = cv2.VideoWriter("videos/distorted_feed.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
for i in range(90):
    vw.write(distort(clean_scene(i)))
vw.release()

ground_tex = textured(800, 800, 100)
box_defs = [((-0.8, 0.8), (-0.6, 0.6), (0.0, 1.4)), ((0.9, 1.5), (-1.1, -0.5), (0.0, 0.8))]
faces = []
for bi, (xr, yr, zr) in enumerate(box_defs):
    x0, x1 = xr
    y0, y1 = yr
    z0, z1 = zr
    quads = [
        ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1)),
        ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0)),
        ([(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)], (0, 1, 0)),
        ([(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)], (-1, 0, 0)),
        ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0)),
    ]
    for fi, (q, nrm) in enumerate(quads):
        faces.append((np.array(q, np.float64), np.array(nrm, np.float64), textured(256, 256, 200 + bi * 10 + fi)))

tiles = []
for gxi in range(8):
    for gyi in range(8):
        q = np.array([[gxi - 4, gyi - 4, 0], [gxi - 3, gyi - 4, 0], [gxi - 3, gyi - 3, 0], [gxi - 4, gyi - 3, 0]], np.float64)
        tex = ground_tex[gyi * 100:(gyi + 1) * 100, gxi * 100:(gxi + 1) * 100]
        tiles.append((q, tex))

K2 = np.array([[800.0, 0, 480.0], [0, 800.0, 360.0], [0, 0, 1]])


def camera_pose(C, target):
    f = (target - C) / np.linalg.norm(target - C)
    r = np.cross(f, np.array([0, 0, 1.0]))
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    t = -R @ C
    return R, t


def project(P, R, t):
    Pc = (R @ P.T).T + t
    z = Pc[:, 2]
    uv = (K2 @ Pc.T).T
    return uv[:, :2] / uv[:, 2:3], z


def draw_quad(canvas, tex, uv):
    h, w = tex.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(src, np.float32(uv))
    warped = cv2.warpPerspective(tex, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
    canvas[mask > 0] = warped[mask > 0]


n_views = 24
target = np.array([0.0, 0.0, 0.6])
for i in range(n_views):
    theta = 2 * np.pi * i / n_views
    Rad = 5.0 + 0.4 * np.sin(3 * theta)
    C = np.array([Rad * np.cos(theta), Rad * np.sin(theta), 2.0 + 1.3 * (i % 2)])
    R, t = camera_pose(C, target)
    canvas = np.full((H, W, 3), (200, 180, 160), np.uint8)
    for q, tex in tiles:
        uv, z = project(q, R, t)
        if z.min() < 0.5:
            continue
        draw_quad(canvas, tex, uv)
    order = sorted(range(len(faces)), key=lambda k: -np.linalg.norm(faces[k][0].mean(axis=0) - C))
    for k in order:
        q, nrm, tex = faces[k]
        if np.dot(nrm, C - q.mean(axis=0)) <= 0:
            continue
        uv, z = project(q, R, t)
        if z.min() < 0.3:
            continue
        draw_quad(canvas, tex, uv)
    cv2.imwrite(f"images/colmap_input/view_{i + 1:02d}.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

print("assets ready")
