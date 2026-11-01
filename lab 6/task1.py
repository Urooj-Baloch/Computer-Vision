import cv2
import numpy as np

img = cv2.imread("images/computer_lab.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=10)
line_img = img.copy()
horizontals = []
verticals = []
if lines is not None:
    for l in lines:
        x1, y1, x2, y2 = np.asarray(l).ravel()[:4]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if abs(y2 - y1) < 5:
            horizontals.append((x1, y1, x2, y2))
        elif abs(x2 - x1) < 5:
            verticals.append((x1, y1, x2, y2))

dark = cv2.inRange(gray, 0, 40)
dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result = img.copy()
filtered = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if 150 < w < 260 and 100 < h < 180 and 1.1 < w / h < 1.9:
        filtered.append((x + 8, y + 8, w - 16, h - 16))
filtered.sort(key=lambda s: (round(s[1] / 100), s[0]))

on_count = 0
off_count = 0
for (x, y, w, h) in filtered:
    roi = gray[y + 5:y + h - 5, x + 5:x + w - 5]
    mean = roi.mean() if roi.size else 0
    status = "ON" if mean > 60 else "OFF"
    if status == "ON":
        on_count += 1
        color = (0, 255, 0)
    else:
        off_count += 1
        color = (0, 0, 255)
    cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)
    cv2.putText(result, status, (x + 5, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

expected_per_row = 4
rows = {}
for (x, y, w, h) in filtered:
    key = round(y / 100)
    rows.setdefault(key, []).append(x)

missing = 0
for key, xs in rows.items():
    if len(xs) < expected_per_row:
        missing += expected_per_row - len(xs)

print("Screens detected:", len(filtered))
print("ON:", on_count, "OFF:", off_count)
print("Missing screens:", missing)

cv2.imwrite("output/task1_edges.jpg", edges)
cv2.imwrite("output/task1_hough_lines.jpg", line_img)
cv2.imwrite("output/task1_screens.jpg", result)
