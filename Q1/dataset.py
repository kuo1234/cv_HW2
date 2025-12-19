import torch
from torchvision.datasets import VOCDetection
import torchvision.transforms as transforms
import os

class VOCDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, root, year, image_set, download, transforms=None):
        self.dataset = VOCDetection(root, year=year, image_set=image_set, download=download)
        self.transforms = transforms
        
        # Define class names for VOC
        self.classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
            'sofa', 'train', 'tvmonitor'
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        
        # Convert target to Faster R-CNN format
        boxes = []
        labels = []
        
        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]
            
        for obj in objects:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            
            label_name = obj['name']
            labels.append(self.class_to_idx[label_name])
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)
        
        target_formatted = {}
        target_formatted['boxes'] = boxes
        target_formatted['labels'] = labels
        target_formatted['image_id'] = image_id
        target_formatted['area'] = area
        target_formatted['iscrowd'] = iscrowd
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, target_formatted

    def __len__(self):
        return len(self.dataset)

def collate_fn(batch):
    return tuple(zip(*batch))
