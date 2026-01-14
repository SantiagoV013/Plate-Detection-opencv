# Plate Detection and Segmentation with OpenCV

This repository demonstrates a **classical computer vision pipeline** for detecting, segmenting, and measuring white plates of different sizes from a top-down image using **OpenCV** and **scikit-image**.

The project focuses on **image segmentation, connected-component labeling, visualization, and region-based measurements**, without using machine learning.

---

## Features

- RGB → HSV color space conversion
- White-object segmentation using HSV thresholding
- Morphological cleanup (binary closing)
- Connected-component labeling
- Overlay visualization with object IDs
- Quantitative region analysis:
  - Area (pixel count)
  - Equivalent diameter
  - Mean intensity
  - Solidity
  - Centroid
- Tabular output using Pandas

---

## How to Run

### 1. Clone the repository

```
git clone <REPOSITORY_URL>
cd plate-detection-opencv
```

### 2. Create a virtual environment

Mac: 
```
python3 -m venv venv
source venv/bin/activate
```

Windows:
```
venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. File detection

```
cd Actividad1
```

### 5. Verify input image

```
seg_platos.py
plates.png
```

### 6. Run the script

```
python seg_platos.py
```



