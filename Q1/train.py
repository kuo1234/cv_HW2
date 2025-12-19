import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from dataset import VOCDatasetWrapper, collate_fn

def get_model(num_classes):
    # Load pre-trained model
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    # Replace the classifier with a new one, that has num_classes which is user-defined
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Data path
    # Assuming data is in ../Hw2/data relative to this script
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Hw2/data/VOCtrainval_06-Nov-2007")
    
    if not os.path.exists(data_path):
        print(f"Data path not found: {data_path}")
        # Fallback to absolute path if needed or check current dir
        data_path = "c:/Users/p76141495/Downloads/Hw2/Hw2/data/VOCtrainval_06-Nov-2007"
    
    print(f"Loading data from: {data_path}")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Dataset
    try:
        dataset = VOCDatasetWrapper(root=data_path, year='2007', image_set='trainval', download=False, transforms=transform)
        print(f"Dataset loaded. Size: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # Model
    num_classes = 21 # 20 classes + background
    model = get_model(num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 20
    loss_history = []
    best_loss = float('inf')

    print("Start training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(dataloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Iter {i}, Loss: {losses.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        lr_scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model.")

    # Plot loss
    plt.figure()
    plt.plot(range(1, num_epochs+1), loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('training_loss.png')
    print("Saved training loss plot to training_loss.png")

if __name__ == "__main__":
    train()
