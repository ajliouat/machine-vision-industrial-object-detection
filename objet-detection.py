import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import cv2
import os

def collate_fn(batch):
    return tuple(zip(*batch))

def train(model, dataloader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

def evaluate(model, dataloader, device):
    model.eval()
    cpu_device = torch.device("cpu")
    iou_thresholds = [0.5, 0.75]
    stats = {iou: {'tp': 0, 'fp': 0, 'fn': 0} for iou in iou_thresholds}

    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for i in range(len(images)):
                boxes_true = targets[i]['boxes'].to(cpu_device).numpy().astype(np.int32)
                boxes_pred = outputs[i]['boxes'].to(cpu_device).numpy().astype(np.int32)

                for iou_threshold in iou_thresholds:
                    tp, fp, fn = calculate_metrics(boxes_true, boxes_pred, iou_threshold)
                    stats[iou_threshold]['tp'] += tp
                    stats[iou_threshold]['fp'] += fp
                    stats[iou_threshold]['fn'] += fn

    for iou_threshold in iou_thresholds:
        precision = stats[iou_threshold]['tp'] / (stats[iou_threshold]['tp'] + stats[iou_threshold]['fp'] + 1e-6)
        recall = stats[iou_threshold]['tp'] / (stats[iou_threshold]['tp'] + stats[iou_threshold]['fn'] + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        print(f"IoU Threshold: {iou_threshold:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")

def calculate_metrics(boxes_true, boxes_pred, iou_threshold):
    tp = 0
    fp = 0
    fn = 0

    used_pred_boxes = set()
    for true_box in boxes_true:
        best_iou = 0
        best_pred_box = None
        for pred_box in boxes_pred:
            iou = calculate_iou(true_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_box = tuple(pred_box)
        
        if best_iou >= iou_threshold:
            if best_pred_box not in used_pred_boxes:
                tp += 1
                used_pred_boxes.add(best_pred_box)
            else:
                fp += 1
        else:
            fn += 1

    fp += len(boxes_pred) - len(used_pred_boxes)
    return tp, fp, fn

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou

def detect_objects(model, image_path, device, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs[0]

# Set up the dataset and data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.CocoDetection(root='path/to/coco/train', annFile='path/to/coco/annotations/instances_train2017.json', transform=transform)
val_dataset = datasets.CocoDetection(root='path/to/coco/val', annFile='path/to/coco/annotations/instances_val2017.json', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)

# Set up the model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = len(train_dataset.coco.getCatIds()) + 1  # +1 for background class
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Train and evaluate the model
epochs = 10
train(model, train_dataloader, optimizer, device, epochs)
evaluate(model, val_dataloader, device)

# Perform object detection on a new image
new_image_path = 'new_industrial_image.jpg'
predicted_objects = detect_objects(model, new_image_path, device, transform)
print("Predicted objects:")
for obj in predicted_objects:
    print(f"Class: {obj['label']}, Confidence: {obj['score']:.4f}, Bounding Box: {obj['boxes']}")
