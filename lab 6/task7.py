import cv2
import numpy as np
import glob

for path in sorted(glob.glob("images/coins_*.jpg")):
    name = path.split("/")[-1].split(".")[0]
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
                               param1=100, param2=40, minRadius=25, maxRadius=70)
    result = img.copy()
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0]:
            cv2.circle(result, (x, y), r, (0, 255, 0), 3)
            cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
            count += 1
    cv2.putText(result, f"Coins: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv2.imwrite(f"output/task7_{name}.jpg", result)
    print(name, "coins counted:", count)
