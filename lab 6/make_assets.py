import cv2
import numpy as np
import os

rng = np.random.default_rng(7)
os.makedirs("images", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("output", exist_ok=True)


def textured(h, w, seed=0):
    r = np.random.default_rng(seed)
    base = r.integers(0, 255, (h, w, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    base = cv2.normalize(base, None, 40, 230, cv2.NORM_MINMAX)
    for _ in range(60):
        c = tuple(int(x) for x in r.integers(0, 255, 3))
        p1 = (int(r.integers(0, w)), int(r.integers(0, h)))
        p2 = (int(r.integers(0, w)), int(r.integers(0, h)))
        kind = r.integers(0, 3)
        if kind == 0:
            cv2.rectangle(base, p1, p2, c, int(r.integers(1, 4)))
        elif kind == 1:
            cv2.circle(base, p1, int(r.integers(5, 40)), c, int(r.integers(1, 4)))
        else:
            cv2.line(base, p1, p2, c, int(r.integers(1, 4)))
    return base


def draw_monitor(img, x, y, w, h, on=True):
    cv2.rectangle(img, (x - 8, y - 8), (x + w + 8, y + h + 8), (25, 25, 25), -1)
    if on:
        screen = np.zeros((h, w, 3), np.uint8)
        screen[:] = (200, 140, 60)
        cv2.rectangle(screen, (10, 10), (w // 2, h // 3), (255, 255, 255), -1)
        cv2.circle(screen, (w - 40, h - 40), 20, (60, 200, 255), -1)
        img[y:y + h, x:x + w] = screen
    else:
        img[y:y + h, x:x + w] = (12, 12, 12)
    cv2.rectangle(img, (x + w // 2 - 12, y + h + 8), (x + w // 2 + 12, y + h + 30), (60, 60, 60), -1)
    cv2.rectangle(img, (x + w // 2 - 45, y + h + 30), (x + w // 2 + 45, y + h + 38), (60, 60, 60), -1)


def draw_keyboard(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cols = 12
    rows = 4
    kw = (w - 10) // cols
    kh = (h - 10) // rows
    for r in range(rows):
        for c in range(cols):
            cv2.rectangle(img, (x + 5 + c * kw + 1, y + 5 + r * kh + 1),
                          (x + 5 + (c + 1) * kw - 2, y + 5 + (r + 1) * kh - 2), (200, 200, 200), -1)


lab = np.full((700, 1100, 3), (170, 185, 195), np.uint8)
cv2.rectangle(lab, (0, 560), (1100, 700), (110, 120, 130), -1)
states = [[True, True, False, True], [True, False, True, True]]
rows_y = [80, 360]
for ri, ry in enumerate(rows_y):
    cv2.rectangle(lab, (30, ry + 190), (1070, ry + 215), (90, 70, 50), -1)
    for ci in range(4):
        x = 70 + ci * 255
        draw_monitor(lab, x, ry, 180, 120, states[ri][ci])
        draw_keyboard(lab, x + 15, ry + 150, 150, 35)
cv2.imwrite("images/computer_lab.jpg", lab)

mon_ref = np.full((220, 260, 3), (170, 185, 195), np.uint8)
draw_monitor(mon_ref, 40, 30, 180, 120, True)
mon_ref = textured(220, 260, 3) // 3 + mon_ref // 2 * 1
mon_ref = mon_ref.astype(np.uint8)
cv2.imwrite("images/ref_monitor.jpg", mon_ref)

kb_ref = textured(80, 240, 5) // 3
kb_ref = kb_ref.astype(np.uint8)
draw_keyboard(kb_ref, 10, 10, 220, 60)
cv2.imwrite("images/ref_keyboard.jpg", kb_ref)

lab2 = textured(700, 1100, 11) // 2 + 60
lab2 = lab2.astype(np.uint8)
mon_patch = cv2.resize(mon_ref, (300, 254))
kb_patch = cv2.resize(kb_ref, (330, 110))
lab2[60:314, 80:380] = mon_patch
rot = cv2.getRotationMatrix2D((165, 55), 12, 0.8)
kb_rot = cv2.warpAffine(kb_patch, rot, (330, 110), borderMode=cv2.BORDER_REPLICATE)
lab2[420:530, 120:450] = kb_rot
mon2 = cv2.resize(mon_ref, (200, 169))
lab2[200:369, 700:900] = mon2
cv2.imwrite("images/lab_inventory.jpg", lab2)

ref_obj = textured(260, 200, 21)
cv2.putText(ref_obj, "LAB", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
cv2.rectangle(ref_obj, (0, 0), (199, 259), (0, 0, 0), 4)
cv2.imwrite("images/ref_object.jpg", ref_obj)

bg = textured(480, 720, 33) // 2 + 70
bg = bg.astype(np.uint8)
fps = 20
vw = cv2.VideoWriter("videos/object_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (720, 480))
for i in range(100):
    frame = bg.copy()
    scale = 0.7 + 0.4 * np.sin(i / 100 * np.pi)
    angle = i * 1.5
    cx = 150 + i * 5
    cy = 240 + 80 * np.sin(i / 15)
    M = cv2.getRotationMatrix2D((100, 130), angle, scale)
    M[0, 2] += cx - 100
    M[1, 2] += cy - 130
    warped = cv2.warpAffine(ref_obj, M, (720, 480), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpAffine(np.full(ref_obj.shape[:2], 255, np.uint8), M, (720, 480))
    frame[mask > 0] = warped[mask > 0]
    if 40 < i < 55:
        cv2.rectangle(frame, (cx - 40, int(cy) - 90), (cx + 100, int(cy) + 10), (60, 60, 60), -1)
    vw.write(frame)
vw.release()

world = textured(500, 2600, 44)
cv2.GaussianBlur(world, (0, 0), 0.6, world)
step = 450
for i in range(5):
    x0 = i * step
    crop = world[:, x0:x0 + 800].copy()
    dy = int(6 * np.sin(i))
    M = np.float32([[1, 0, 0], [0, 1, dy]])
    crop = cv2.warpAffine(crop, M, (800, 500), borderMode=cv2.BORDER_REPLICATE)
    crop = cv2.convertScaleAbs(crop, alpha=1.0 + 0.03 * (i - 2), beta=0)
    cv2.imwrite(f"images/pano_{i + 1}.jpg", crop)


def road_image(seed, shift):
    img = np.zeros((540, 960, 3), np.uint8)
    img[:] = (150, 170, 190)
    img[:230] = (235, 205, 160)
    cv2.rectangle(img, (0, 230), (960, 540), (70, 70, 70), -1)
    r = np.random.default_rng(seed)
    noise = r.integers(0, 25, (540, 960, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    vp = (480 + shift, 230)
    left_b = (60 + shift // 2, 540)
    right_b = (900 + shift // 2, 540)
    cv2.line(img, vp, left_b, (255, 255, 255), 10)
    cv2.line(img, vp, right_b, (255, 255, 255), 10)
    for t in np.linspace(0.05, 0.95, 9):
        a = (int(vp[0] + (480 + shift // 2 - vp[0]) * t), int(vp[1] + (540 - vp[1]) * t))
        t2 = min(t + 0.06, 1.0)
        b = (int(vp[0] + (480 + shift // 2 - vp[0]) * t2), int(vp[1] + (540 - vp[1]) * t2))
        cv2.line(img, a, b, (0, 220, 255), 5)
    cv2.rectangle(img, (0, 0), (960, 230), (235, 205, 160), -1)
    cv2.circle(img, (800, 80), 40, (80, 220, 255), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


for i, s in enumerate([0, -60, 70, 30]):
    cv2.imwrite(f"images/road_{i + 1}.jpg", road_image(i, s))


def coin_image(seed, n, size=(600, 900)):
    r = np.random.default_rng(seed)
    img = np.full((size[0], size[1], 3), (210, 215, 220), np.uint8)
    noise = r.integers(0, 15, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    placed = []
    tries = 0
    while len(placed) < n and tries < 2000:
        tries += 1
        rad = int(r.choice([34, 42, 52, 60]))
        x = int(r.integers(rad + 10, size[1] - rad - 10))
        y = int(r.integers(rad + 10, size[0] - rad - 10))
        if all((x - px) ** 2 + (y - py) ** 2 > (rad + pr + 8) ** 2 for px, py, pr in placed):
            placed.append((x, y, rad))
            col = [(60, 140, 200), (170, 175, 180), (40, 110, 170)][int(r.integers(0, 3))]
            cv2.circle(img, (x, y), rad, col, -1)
            cv2.circle(img, (x, y), rad, (30, 60, 90), 3)
            cv2.circle(img, (x, y), int(rad * 0.7), tuple(int(c * 0.85) for c in col), 2)
            cv2.putText(img, "5", (x - 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 60, 90), 2)
    return img


cv2.imwrite("images/coins_1.jpg", coin_image(1, 9))
cv2.imwrite("images/coins_2.jpg", coin_image(2, 14))
cv2.imwrite("images/coins_3.jpg", coin_image(3, 6))

sw = cv2.VideoWriter("videos/security_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480))
room = np.full((480, 640, 3), (200, 205, 210), np.uint8)
cv2.rectangle(room, (0, 380), (640, 480), (120, 125, 130), -1)
for i in range(120):
    frame = room.copy()
    if i > 30:
        x = int(60 + (i - 30) * 5.5)
        cv2.rectangle(frame, (x, 300), (x + 70, 370), (30, 30, 200), -1)
        cv2.circle(frame, (x + 35, 335), 12, (255, 255, 255), -1)
    sw.write(frame)
sw.release()

print("assets ready")
