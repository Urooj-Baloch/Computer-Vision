"""Run all ten tasks in order; results go to outputs/."""
import runpy
for name in ["task01_global_vs_adaptive", "task02_adaptive_params", "task03_otsu", "task04_hsv_color", "task05_canny",
             "task06_region_growing", "task07_watershed", "task08_watershed_tuning", "task09_kmeans", "task10_unknown_image"]:
    print(f"\n===== {name} =====")
    runpy.run_path(f"{name}.py", run_name="__main__")
