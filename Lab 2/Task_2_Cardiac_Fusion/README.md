# Task 2: Multi-Modal Cardiac Image Fusion

Notebook: `modal_fusion.ipynb`

Operations: equalization, BONE and JET color mapping with a 50/50 overlay, weighted fusion (alpha 0.7, beta 0.3, gamma 1.3) with a weight sweep, normalized log plus power-law refinement, and a CT | MRI | fused comparison with validation metrics.

Inputs: `data/ct_slice.png`, `data/mri_slice.png` (synthetic registered pair; replace with real slices of the same names). Results are written to `output/`.
