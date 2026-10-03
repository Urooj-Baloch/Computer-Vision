# Lab 05 – Image Segmentation (OpenCV + NumPy)

Ten segmentation tasks: global / adaptive / Otsu thresholding, HSV colour masks, Canny edges, region growing,
marker-based watershed (and its tuning), K-Means, and a three-method comparison.

```
pip install -r requirements.txt
python prepare_images.py      # optional, images/ is already included
python run_all.py             # runs task01 ... task10, results in outputs/
```

* `taskNN_*.py` – one script per task (shared helpers in `common.py`, `watershed_pipeline.py`)
* `outputs/` – comparison figures, saved masks, `taskNN_log.txt` with the parameters used
* `REPORT.md` – parameters, justifications and answers to each task's analysis questions
