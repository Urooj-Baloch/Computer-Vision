"""Re-create the images/ folder (needs internet for the OpenCV samples; pydicom ships the MRI test file)."""
import os, urllib.request
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); IMG = os.path.join(HERE, "images"); os.makedirs(IMG, exist_ok=True)
RAW = "https://raw.githubusercontent.com/opencv/opencv/4.x/"
FILES = {"sudoku.png": RAW + "samples/data/sudoku.png",
         "smarties.png": RAW + "samples/data/smarties.png",
         "messi5.jpg": RAW + "samples/data/messi5.jpg",
         "water_coins.jpg": RAW + "doc/py_tutorials/py_imgproc/py_watershed/images/water_coins.jpg"}
for name, url in FILES.items():
    urllib.request.urlretrieve(url, os.path.join(IMG, name))

# brain MRI slice: MR_small.dcm from the pydicom test data, scaled to 8 bit and enlarged x2
import pydicom
from pydicom.data import get_testdata_file
a = pydicom.dcmread(get_testdata_file("MR_small.dcm")).pixel_array.astype(np.float32)
a = (255 * (a - a.min()) / (a.max() - a.min())).astype(np.uint8)
cv2.imwrite(os.path.join(IMG, "brain.png"), cv2.resize(a, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
print("images ready")
