import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from skimage import io, measure
from skimage.color import label2rgb
from skimage.measure import regionprops
from scipy import ndimage as nd
import pandas as pd

# --- Load image safely ---
img_path = Path(__file__).with_name("plates.png")
if not img_path.exists():
    raise FileNotFoundError(
        f"Could not find {img_path.name} next to this script. Expected at: {img_path}"
    )

# skimage loads RGB by default
img = io.imread(str(img_path))

# --- 1) Show original ---
plt.figure(figsize=(6, 6))
plt.title("Original")
plt.imshow(img)
plt.axis("off")

# --- 2) Segment white plates in HSV ---
# Ensure uint8 for OpenCV
if img.dtype != np.uint8:
    img_u8 = img.astype(np.uint8)
else:
    img_u8 = img

hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
# White mask (tune if needed)
mask = cv2.inRange(hsv, (0, 0, 180), (180, 70, 255))

plt.figure(figsize=(6, 6))
plt.title("Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

# --- 3) Clean mask (closing) ---
# ndimage expects boolean; convert once and back if needed
mask_bool = mask > 0
closed_mask = nd.binary_closing(mask_bool, structure=np.ones((7, 7), dtype=bool))

plt.figure(figsize=(6, 6))
plt.title("Closed mask")
plt.imshow(closed_mask, cmap="gray")
plt.axis("off")

# --- 4) Label connected components ---
label_image = measure.label(closed_mask)

plt.figure(figsize=(6, 6))
plt.title("Labels")
plt.imshow(label_image)
plt.axis("off")

# --- 5) Overlay labels on image + draw label text ---
image_label_overlay = label2rgb(label_image, image=img, bg_label=0)

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_title("Detected plates (labeled)")
ax.imshow(image_label_overlay)

for r in regionprops(label_image):
    # Skip tiny regions (noise). Adjust threshold as needed.
    if r.area < 500:
        continue

    ax.text(
        r.centroid[1],
        r.centroid[0],
        f"{r.label}",
        color="red",
        fontsize=12,
        ha="center",
        va="center",
    )

ax.axis("off")

# --- 6) Table of region properties ---
props = measure.regionprops_table(
    label_image,
    intensity_image=img,
    properties=[
        "label",
        "area",
        "equivalent_diameter",
        "mean_intensity",
        "solidity",
        "centroid",
    ],
)

df = pd.DataFrame(props)

# Filter same noise threshold used above (keep consistent)
df = df[df["area"] >= 500].reset_index(drop=True)

count = len(df)
print("Total plates:", count)
print(df.head())

# --- Show all figures when running as a script ---
plt.show()