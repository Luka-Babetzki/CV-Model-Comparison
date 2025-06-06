# ****** HoG Model ******

import cv2
from skimage import feature, exposure
import os

# Path to the Penn Fudan dataset images folder
images_folder = 'c:/path/to/PennFudanPed/PNGImages'  # Update this path as needed

# Get a list of all PNG images in the folder
image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]

for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)

    if not os.path.isfile(image_path):
        print(f"Error: Image file not found at {image_path}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        continue

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Input Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ****** HOG map of input image ******
    # Extract HOG features from the grayscale image
    hog_features, hog_image = feature.hog(
        grayimg, orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), visualize=True
    )

    hog_image_rescaled = exposure.rescale_intensity(
        hog_image, in_range=(0, 10)
    )

    cv2.imshow("HOG", hog_image_rescaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ****** Person detection ******
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog.detectMultiScale(
        grayimg,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.25,
        useMeanshiftGrouping=False
    )
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Person Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
