# Lab 08: Object Detection Using R-CNN

Dataset (not BCCD): Raccoon (datitran/raccoon_dataset, 200 images) and Kangaroo (experiencor/kangaroo, 164 images), combined into a two-class detection set of 364 images, split 291 train / 73 test.

Pipeline in `task1.py`: selective search proposals (OpenCV) -> warp each region to 64x64 -> CNN fine-tuned on region labels (IoU >= 0.5 positive, < 0.3 background) -> CNN features -> one linear SVM per class -> bounding-box regression -> per-class NMS.

The CNN is trained from scratch because pretrained weights could not be downloaded.

Run from this folder:

```
pip install -r requirements.txt
python task1.py
```

Results on the 73 test images (AP at IoU 0.5):

| Class | Without box regression | With box regression |
|---|---|---|
| raccoon | 0.643 | 0.700 |
| kangaroo | 0.346 | 0.362 |
| mAP | 0.495 | 0.531 |

Average inference: 338 proposals and 0.67 s per image on 2 CPU cores, of which 0.32 s is selective search.

Outputs are in `output/`: `fig1` to `fig6`, `metrics.json`, `training_log.txt`, `predictions/` (every test image with detections), and the trained `region_cnn.keras`.
