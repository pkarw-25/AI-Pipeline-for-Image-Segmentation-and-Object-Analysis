import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def load_model():
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def segment_image(image_path, model):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    masks = outputs[0]['masks'].cpu().detach().numpy()
    labels = outputs[0]['labels'].cpu().detach().numpy()
    
    return image, masks, labels

def visualize_segmentation(image, masks, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    for mask in masks:
        mask = mask[0]
        plt.imshow(mask, alpha=0.5)
    
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()
