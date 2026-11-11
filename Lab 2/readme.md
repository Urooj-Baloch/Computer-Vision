# Medical Imaging Portfolio

Urooj Baloch, FAST NUCES Karachi. Computer Vision Lab 02: point operations and intensity transforms on medical images.

## Structure

```
Urooj_Baloch_Medical_Imaging_Portfolio/
    README.md
    requirements.txt
    Task_1_Chest_XRay/
        README.md
        xray_enhancement.ipynb
        data/
        output/
    Task_2_Cardiac_Fusion/
        README.md
        modal_fusion.ipynb
        data/
        output/
    Task_3_Echo_Analysis/
        README.md
        realtime_echo.ipynb
        data/
        output/
```

## Run

```
pip install -r requirements.txt
cd Task_1_Chest_XRay
jupyter notebook xray_enhancement.ipynb
```

Repeat from the other task folders. All paths are relative, so run each notebook from inside its own task folder.

## Data

- Task 1 uses two public chest X-rays from the ieee8023 covid-chestxray-dataset (CC BY-NC-SA).
- Task 2 (`ct_slice.png`, `mri_slice.png`) and Task 3 (`echo_sample.mp4`) use synthetic stand-in data. To use real Kaggle data, replace those files with your own using the same file names.
