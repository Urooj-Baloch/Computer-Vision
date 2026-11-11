# Task 3: Real-Time Echocardiogram Video Analysis

Notebook: `realtime_echo.ipynb`

Per-frame pipeline: gray, histogram equalization, COLORMAP_JET, running gray-world color balance, logarithmic transform, power-law transform (gamma 1.5). A live `cv2.imshow` window is used when a display exists. Original and enhanced frames are written side by side.

Input: `data/echo_sample.mp4` (synthetic; replace with a real echo clip of the same name). Outputs in `output/`: `echo_side_by_side.mp4`, `frame_000/030/060_side_by_side.png`, `side_by_side_screenshots.png`, `pipeline_stages.png`.
