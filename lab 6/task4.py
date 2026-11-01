import cv2
import numpy as np

ref = cv2.imread("images/ref_object.jpg")
ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp_r, des_r = sift.detectAndCompute(ref_gray, None)
bf = cv2.BFMatcher()

cap = cv2.VideoCapture("videos/object_video.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output/task4_recognition.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

rh, rw = ref_gray.shape
corners = np.float32([[0, 0], [0, rh - 1], [rw - 1, rh - 1], [rw - 1, 0]]).reshape(-1, 1, 2)

frame_no = 0
detected = 0
saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_f, des_f = sift.detectAndCompute(gray, None)
    found = False
    if des_f is not None and len(kp_f) >= 2:
        matches = bf.knnMatch(des_r, des_f, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.7 * n.distance]
        if len(good) >= 10:
            src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is not None and mask.sum() >= 8:
                box = cv2.perspectiveTransform(corners, H)
                x, y, bw, bh = cv2.boundingRect(np.int32(box))
                cv2.polylines(frame, [np.int32(box)], True, (0, 255, 0), 3)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                cv2.putText(frame, "Object", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                found = True
                detected += 1
    if not found:
        cv2.putText(frame, "Not found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if frame_no in (1, 25, 50, 75, 100):
        cv2.imwrite(f"output/task4_frame_{frame_no}.jpg", frame)
        saved += 1
    out.write(frame)

cap.release()
out.release()
print("Frames:", frame_no)
print("Detected in:", detected)
