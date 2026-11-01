import cv2
import numpy as np

cap = cv2.VideoCapture("videos/security_video.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output/task8_security.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

zone = (380, 250, 600, 400)
bg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=40, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

frame_no = 0
alarm_frames = 0
first_alarm = None
saved = False

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1
    mask = bg.apply(frame)
    if frame_no <= 10:
        mask[:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    intruder = False
    for c in contours:
        if cv2.contourArea(c) < 800:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        ix1 = max(x, zone[0])
        iy1 = max(y, zone[1])
        ix2 = min(x + bw, zone[2])
        iy2 = min(y + bh, zone[3])
        if ix1 < ix2 and iy1 < iy2:
            intruder = True

    zone_color = (0, 0, 255) if intruder else (255, 255, 0)
    cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), zone_color, 3)
    cv2.putText(frame, "SECURITY ZONE", (zone[0], zone[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)

    if intruder:
        alarm_frames += 1
        if first_alarm is None:
            first_alarm = frame_no
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 255), -1)
        cv2.putText(frame, "ALARM: UNAUTHORIZED OBJECT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        if alarm_frames == 15 and not saved:
            cv2.imwrite("output/task8_alarm_frame.jpg", frame)
            saved = True

    if frame_no == 20:
        cv2.imwrite("output/task8_clear_frame.jpg", frame)

    out.write(frame)

cap.release()
out.release()
print("Frames:", frame_no)
print("First alarm frame:", first_alarm)
print("Alarm frames:", alarm_frames)
