import cv2
import numpy as np
import glob


def region_of_interest(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    polygon = np.array([[(0, h), (w, h), (w // 2 + 150, int(h * 0.45)), (w // 2 - 150, int(h * 0.45))]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)


def make_line(y1, y2, params):
    slope, intercept = params
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, y1, x2, y2)


for path in sorted(glob.glob("images/road_*.jpg")):
    name = path.split("/")[-1].split(".")[0]
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 160)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=100)

    left, right = [], []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = np.asarray(l).ravel()[:4]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.4:
                continue
            intercept = y1 - slope * x1
            if slope < 0:
                left.append((slope, intercept))
            else:
                right.append((slope, intercept))

    result = img.copy()
    overlay = np.zeros_like(img)
    h = img.shape[0]
    y_top = int(h * 0.5)
    pts = []
    if left:
        L = make_line(h, y_top, np.mean(left, axis=0))
        cv2.line(result, (L[0], L[1]), (L[2], L[3]), (0, 0, 255), 8)
        pts += [(L[0], L[1]), (L[2], L[3])]
    if right:
        R = make_line(h, y_top, np.mean(right, axis=0))
        cv2.line(result, (R[0], R[1]), (R[2], R[3]), (0, 0, 255), 8)
        pts += [(R[2], R[3]), (R[0], R[1])]
    if len(pts) == 4:
        cv2.fillPoly(overlay, [np.array(pts, np.int32)], (0, 255, 0))
        result = cv2.addWeighted(result, 1, overlay, 0.3, 0)

    cv2.imwrite(f"output/task6_{name}_edges.jpg", roi)
    cv2.imwrite(f"output/task6_{name}_lanes.jpg", result)
    print(name, "left lines:", len(left), "right lines:", len(right))
