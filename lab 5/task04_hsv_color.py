from common import *

img = load("smarties.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)       
lower_a, upper_a = np.array([58, 230, 230]), np.array([64, 255, 255])
lower_b, upper_b = np.array([40, 80, 60]), np.array([85, 255, 255])

mask_a = cv2.inRange(hsv, lower_a, upper_a)
mask_b = cv2.inRange(hsv, lower_b, upper_b)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_b_clean = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel)      
mask_b_clean = cv2.morphologyEx(mask_b_clean, cv2.MORPH_CLOSE, kernel) 

ext_a = cv2.bitwise_and(img, img, mask=mask_a)
ext_b = cv2.bitwise_and(img, img, mask=mask_b_clean)

def n_obj(m, min_area=100):
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return int(np.sum(st[1:, cv2.CC_STAT_AREA] >= min_area)), int(np.sum(m > 0))

na, pa = n_obj(mask_a); nb, pb = n_obj(mask_b_clean)
log("task04", ["TASK 4 - HSV parameters",
               f"Restrictive mask: lower={lower_a.tolist()} upper={upper_a.tolist()} -> {pa} px, {na} objects (>=100 px)",
               f"Final mask      : lower={lower_b.tolist()} upper={upper_b.tolist()} -> {pb} px, {nb} objects (>=100 px)",
               "Justification: green candy hue is about 60 in OpenCV scale; the white background has S close to 0 so S>=80 removes it;",
               "V>=60 drops dark pixels (the brown candy / shadows)."])

footer = ("Why does the restrictive mask fail? Its hue, saturation and value windows are so narrow that they only accept the few pixels whose colour exactly equals the candy's "
          "base colour. Shading on the curved surface lowers V and S, and the specular highlight raises V / lowers S, so most of the candy falls outside the range and "
          "only small fragments are detected. The wider range keeps the whole object but still rejects the white background (S near 0).")
panel([img, mask_a, ext_a, mask_b_clean, ext_b],
      ["Original", "Mask 1: too restrictive", "Extracted (restrictive)", "Mask 2: final", "Extracted (final)"],
      "task04_hsv_comparison.png", cols=5, cell=(3.4, 3.2), footer=footer)
save("task04_final_mask.png", mask_b_clean)
save("task04_final_extracted.png", ext_b)
