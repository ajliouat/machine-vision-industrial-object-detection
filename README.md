# Industrial Object Detection with Faster R-CNN

This repository contains a highly advanced implementation of an object detection system specifically designed for detecting objects in industrial images. The code utilizes state-of-the-art techniques and deep learning architectures, such as the Faster R-CNN model with a ResNet-50 backbone, to achieve high accuracy and performance.

## Features

- Utilizes the powerful Faster R-CNN architecture with a ResNet-50 backbone for accurate object detection.
- Leverages pre-trained weights from the torchvision library for improved performance.
- Implements advanced training and evaluation loops with support for variable-length targets and batching.
- Calculates precision, recall, and F1-score metrics at different IoU thresholds for comprehensive evaluation.
- Provides a `detect_objects` function to detect objects in new images, returning class labels, confidence scores, and bounding boxes.
- Utilizes the COCO dataset, a widely-used public dataset for object detection, to train and evaluate the model.

## Requirements

- Python 3.6+
- PyTorch 1.7+
- Torchvision 0.8+
- Pillow
- NumPy
- OpenCV (cv2)

## Dataset

This code uses the COCO dataset for training and evaluation. You need to download the COCO dataset and provide the paths to the training and validation directories (`root` parameter) and the corresponding annotation files (`annFile` parameter) in the `datasets.CocoDetection` function.

Please make sure to replace `'path/to/coco/train'`, `'path/to/coco/annotations/instances_train2017.json'`, `'path/to/coco/val'`, and `'path/to/coco/annotations/instances_val2017.json'` with the actual paths to the COCO dataset on your system.

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/ajliouat/machine-vision-industrial-object-detection.git
   ```

2. Install the required dependencies:
   ```
   pip install torch torchvision pillow numpy opencv-python
   ```

3. Download the COCO dataset and update the dataset paths in the code.

4. Run the code:
   ```
   python object_detection.py
   ```

5. The code will train the object detection model on the COCO dataset for the specified number of epochs. It will display the training loss for each epoch.

6. After training, the code will evaluate the model on the validation set and calculate precision, recall, and F1-score metrics at different IoU thresholds.

7. To perform object detection on a new image, provide the path to the image in the `new_image_path` variable and run the code. It will display the predicted objects along with their class labels, confidence scores, and bounding boxes.

## Model Architecture

The object detection model is based on the Faster R-CNN architecture with a ResNet-50 backbone. Faster R-CNN is a state-of-the-art object detection model that consists of two main components: a Region Proposal Network (RPN) and a Fast R-CNN detector. The RPN generates object proposals, and the Fast R-CNN detector refines the proposals and produces the final object detections.

The ResNet-50 backbone is used as the feature extractor in the Faster R-CNN model. It provides a deep and powerful representation of the input images, enabling accurate object detection.

## Evaluation Metrics

The code evaluates the object detection model using precision, recall, and F1-score metrics at different IoU thresholds. These metrics provide a comprehensive assessment of the model's performance:

- Precision: The proportion of true positive detections among all positive detections.
- Recall: The proportion of true positive detections among all actual objects.
- F1-score: The harmonic mean of precision and recall, providing a balanced measure of the model's accuracy.

The code calculates these metrics at IoU thresholds of 0.5 and 0.75, allowing for a granular evaluation of the model's performance.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more information.

## Contact

For questions or feedback, please contact [a.jliouat@yahoo.fr](mailto:a.jliouat@yahoo.fr).
