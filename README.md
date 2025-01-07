# CV-Model-Comparison
## Goal
Compare the performance of Histogram of Oriented Gradients (HoG) model &amp; the Faster R-CNN model when trained on the Penn Fudan dataset

## Overview
### HoG
The Histogram of Oriented Gradients (HoG) algorithm, introduced by Navneet Dalal and Bill Triggs in 2005, is a prominent feature descriptor in computer vision. It is widely employed for object detection tasks by capturing local object appearance and shape through the distribution of intensity gradients or edge directions in an image. The process involves dividing the image into regions, calculating gradient magnitude and orientation, and constructing histograms within each region. This creates a feature vector representing the image's local structure, making HoG effective for tasks like pedestrian and face detection.

### Faster R-CNN
Faster R-CNN, proposed by Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun in 2015, is an advanced object detection algorithm. It combines deep learning with region proposal networks to enhance speed and accuracy. Unlike its predecessor, R-CNN, Faster R-CNN integrates region proposal generation and object detection into a unified network. A key innovation is the Region Proposal Network (RPN), a neural module efficiently proposing candidate object regions. This eliminates the need for external proposal methods, significantly speeding up detection. Faster R-CNN uses a CNN backbone for feature extraction, which is utilised for both region proposal and subsequent object classification. This integration enhances accuracy and computational efficiency, marking a significant advancement in computer vision's object detection domain.

## How to use
