import cv2
import numpy as np

scene = cv2.imread("images/lab_inventory.jpg")
refs = {
    "Monitor": cv2.imread("images/ref_monitor.jpg"),
    "Keyboard": cv2.imread("images/ref_keyboard.jpg"),
}

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
kp_s, des_s = sift.detectAndCompute(scene_gray, None)

output = scene.copy()
inventory = {}
colors = {"Monitor": (0, 255, 0), "Keyboard": (255, 0, 0)}
ratio = {"Monitor": 0.7, "Keyboard": 0.9}
min_inl = {"Monitor": 10, "Keyboard": 25}

for name, ref in refs.items():
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    kp_r, des_r = sift.detectAndCompute(ref_gray, None)
    found = 0
    mask_scene = np.ones(len(kp_s), dtype=bool)
    for _ in range(4):
        idx = np.where(mask_scene)[0]
        if len(idx) < 10:
            break
        sub_kp = [kp_s[i] for i in idx]
        sub_des = des_s[idx]
        matches = bf.knnMatch(des_r, sub_des, k=2)
        good = []
        for m, n in matches:
            if m.distance < ratio[name] * n.distance:
                good.append(m)
        if len(good) < 8:
            break
        src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([sub_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or inliers.sum() < min_inl[name]:
            break
        h, w = ref_gray.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        box = cv2.perspectiveTransform(pts, H)
        cv2.polylines(output, [np.int32(box)], True, colors[name], 3)
        cv2.putText(output, f"{name} #{found + 1}", tuple(np.int32(box[0][0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[name], 2)
        found += 1
        inl = inliers.ravel() == 1
        for m, keep in zip(good, inl):
            if keep:
                mask_scene[idx[m.trainIdx]] = False
        poly_mask = np.zeros(scene_gray.shape, np.uint8)
        cv2.fillPoly(poly_mask, [np.int32(box)], 255)
        for i in idx:
            x, y = kp_s[i].pt
            if poly_mask[int(min(max(y, 0), poly_mask.shape[0] - 1)), int(min(max(x, 0), poly_mask.shape[1] - 1))]:
                mask_scene[i] = False
    inventory[name] = found

print("Inventory:")
for k, v in inventory.items():
    print(f"{k}: {v}")

cv2.imwrite("output/task2_inventory.jpg", output)
