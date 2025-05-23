# CV-Model-Comparison

This project compares the performance of two computer vision models for pedestrian detection on the Penn Fudan dataset:

- **Histogram of Oriented Gradients (HoG) Model**
- **Faster R-CNN Model**

## Dataset

Both models use the [Penn Fudan Pedestrian Dataset](https://www.cis.upenn.edu/~jshi/ped_html/) located at `../lab9 data/PennFudanPed`.

## HoG Model

The HoG model uses OpenCV's HOG descriptor and a pre-trained SVM for pedestrian detection.

**File:** [`HoG Model.py`](HoG%20Model.py)

**Steps:**
1. Loads an image from the dataset.
2. Converts the image to grayscale.
3. Extracts and displays the HOG feature map.
4. Uses OpenCV's default people detector to detect pedestrians.
5. Draws bounding boxes around detected people and displays the result.

## Faster R-CNN Model

The Faster R-CNN model uses PyTorch's torchvision implementation with a pre-trained ResNet-50 backbone.

**File:** [`Faster R-CNN Model.py`](Faster%20R-CNN%20Model.py)

**Steps:**
1. Loads an image from the dataset.
2. Converts the image to RGB and normalizes it.
3. Passes the image through the Faster R-CNN model.
4. Draws bounding boxes for detected objects with confidence above a threshold.
5. Displays the detection result.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- scikit-image
- torchvision
- torch

Install dependencies with:

```sh
pip install opencv-python scikit-image torch torchvision
```
