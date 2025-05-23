# CV-Model-Comparison

A simple Python project to compare two popular computer vision models for pedestrian detection: Histogram of Oriented Gradients (HoG) and Faster R-CNN.

## Features

- Loads all images from the Penn Fudan Pedestrian dataset
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

2. **Download the Penn Fudan Pedestrian dataset** and place it in a folder, e.g. `c:/path/to/PennFudanPed/PNGImages`.

3. **Update the `images_folder` variable** at the top of both `HoG Model.py` and `Faster R-CNN Model.py` to the path where your dataset images are stored:

    ```python
    images_folder = 'c:/path/to/PennFudanPed/PNGImages'  # Update this path as needed
    ```

4. **Run the HoG model:**

    ```sh
    python "HoG Model.py"
    ```

5. **Run the Faster R-CNN model:**

    ```sh
    python "Faster R-CNN Model.py"
    ```

Each script will process and display the detection results for every image in the dataset folder.

## Notes

- Ensure the dataset path is correct in both scripts.
- The HoG model uses greyscale images and classical feature-based detection.
- The Faster R-CNN model uses a deep learning approach and may require a GPU for faster inference.
- Results may vary depending on the image and environment.
