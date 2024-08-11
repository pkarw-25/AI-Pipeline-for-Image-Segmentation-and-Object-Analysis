import torch
from PIL import Image
from torchvision import transforms, models

def load_identification_model():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def identify_objects(image_path, model):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    labels = outputs[0]['labels'].cpu().detach().numpy()
    scores = outputs[0]['scores'].cpu().detach().numpy()
    
    return labels, scores
