import sys
import os
import cv2
import numpy as np
import torch
import torchvision
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hw2 Object Detection")
        self.setGeometry(100, 100, 400, 500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Load Image Button
        self.btn_load_image = QPushButton("Load image")
        self.btn_load_image.clicked.connect(self.load_image)
        self.layout.addWidget(self.btn_load_image)

        # Image Label
        self.lbl_image = QLabel("Loaded image:")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.lbl_image)
        
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumHeight(200)
        self.layout.addWidget(self.image_display)

        # 1.1 Show Architecture
        self.btn_show_architecture = QPushButton("1.1 Show architecture")
        self.btn_show_architecture.clicked.connect(self.show_architecture)
        self.layout.addWidget(self.btn_show_architecture)

        # 1.2 Show Training Loss
        self.btn_show_training_loss = QPushButton("1.2 Show training loss")
        self.btn_show_training_loss.clicked.connect(self.show_training_loss)
        self.layout.addWidget(self.btn_show_training_loss)

        # 1.3 Inference
        self.btn_inference = QPushButton("1.3 Inference")
        self.btn_inference.clicked.connect(self.inference)
        self.layout.addWidget(self.btn_inference)

        self.loaded_image_path = None
        
        self.CLASSES = [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
            "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
            "horse", "motorbike", "person", "pottedplant", "sheep", 
            "sofa", "train", "tvmonitor"
        ]

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_path:
            self.loaded_image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_display.setPixmap(pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            print(f"Loaded image: {file_path}")

    def show_architecture(self):
        # Use model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=21)
        # Note: num_classes=21 includes 20 object classes + 1 background
        # Dataset: Pascal VOC 2007 (data/VOCtrainval_06-Nov-2007)
        try:
            model = fasterrcnn_resnet50_fpn(num_classes=21)
            print(model)
        except Exception as e:
            print(f"Error loading model: {e}")

    def show_training_loss(self):
        try:
            if os.path.exists("training_loss.png"):
                self.loss_window = QWidget()
                self.loss_window.setWindowTitle("Training Loss")
                layout = QVBoxLayout()
                label = QLabel()
                pixmap = QPixmap("training_loss.png")
                label.setPixmap(pixmap)
                layout.addWidget(label)
                self.loss_window.setLayout(layout)
                self.loss_window.show()
            else:
                print("Training loss figure not found. Please train the model first.")
        except Exception as e:
            print(f"Error showing training loss: {e}")

    def inference(self):
        if not self.loaded_image_path:
            print("Please load an image first.")
            return

        # 1. Load Model
        try:
            # Reconstruct the model structure
            weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            model = fasterrcnn_resnet50_fpn(weights=weights)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21) # 21 classes
            
            # Load weights
            model_path = "best_model.pth"
            if not os.path.exists(model_path):
                 print(f"Model not found at {model_path}. Please train the model first.")
                 return

            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"Model loaded from {model_path}")

        except Exception as e:
            print(f"Error loading model: {e}")
            return

        # 2. Prepare Image
        try:
            image = cv2.imread(self.loaded_image_path)
            if image is None:
                print("Failed to read image.")
                return
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = transforms.functional.to_tensor(image_rgb).to(device)
            
            # 3. Inference
            with torch.no_grad():
                prediction = model([image_tensor])[0]

            # 4. Draw Results
            threshold = 0.5
            
            boxes = prediction['boxes'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            
            for box, label, score in zip(boxes, labels, scores):
                if score > threshold:
                    xmin, ymin, xmax, ymax = box.astype(int)
                    class_name = self.CLASSES[label]
                    text = f"{class_name}: {score:.2f}"
                    
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 5. Show in GUI
            image_rgb_out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb_out.shape
            bytes_per_line = ch * w
            q_image = QImage(image_rgb_out.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            self.image_display.setPixmap(pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            print("Inference done.")
            
        except Exception as e:
            print(f"Error during inference: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
