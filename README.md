# CV-Model-Comparison

A simple Python project to compare two popular computer vision models for pedestrian detection: Histogram of Oriented Gradients (HoG) and Faster R-CNN.

## Features

- Loads a sample image from the Penn Fudan Pedestrian dataset
- Detects pedestrians using:
  - HoG + SVM (OpenCV)
  - Faster R-CNN (PyTorch/Torchvision)
- Visualises detection results with bounding boxes

## Requirements

- Python 3.x
- OpenCV
- scikit-image
- PyTorch
- Torchvision

## Usage

1. **Install dependencies:**

    ```sh
    pip install opencv-python scikit-image torch torchvision
    ```

2. **Download the Penn Fudan Pedestrian dataset** and place it in `../lab9 data/PennFudanPed` relative to the scripts.

3. **Run the HoG model:**

    ```sh
    python "HoG Model.py"
    ```

4. **Run the Faster R-CNN model:**

    ```sh
    python "Faster R-CNN Model.py"
    ```

Each script will display the original image and the detection results.

## Notes

- Ensure the dataset path is correct in both scripts.
- The HoG model uses greyscale images and classical feature-based detection.
- The Faster R-CNN model uses a deep learning approach and may require a GPU for faster inference.
- Results may vary depending on the image and environment.
