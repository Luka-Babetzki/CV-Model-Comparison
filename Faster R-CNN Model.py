# ****** Faster R-CNN Model ******

import cv2
from skimage import feature, exposure
import os
import torchvision

# Model creating
print("Creating model")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Path to the Penn Fudan dataset images folder
images_folder = 'c:/path/to/PennFudanPed/PNGImages'  # Update this path as needed

# Get a list of all PNG images in the folder
image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]

for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    print(f"Processing {image_path}")

    if not os.path.isfile(image_path):
        print(f"Error: Image file not found at {image_path}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform the image to a tensor
    img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
    input = [img_tensor]

    # Forward pass through the model
    out = model(input)
    boxes = out[0]['boxes']
    scores = out[0]['scores']

    # Draw bounding boxes on the image
    threshold = 0.5
    for idx in range(boxes.shape[0]):
        if scores[idx] >= threshold:
            x1, y1, x2, y2 = boxes[idx]
            cv2.rectangle(img, (int(x1), int(y1)),
                          (int(x2), int(y2)), (255, 0, 0), 2)

    # Resize the image for display
    img_disp = cv2.resize(img, (800, 600))

    # Display the result
    cv2.imshow('Detection_result', img_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()