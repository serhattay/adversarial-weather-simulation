from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

model_name = "facebook/detr-resnet-50"
    
def test():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

def get_model(num_classes, model_name):
    
    # Load the standard, fully updated main branch model
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True 
    )
    
    # Replace the classification head for your KITTI dataset
    model.class_labels_classifier = torch.nn.Linear(256, num_classes + 1)
    
    return model


def get_processor(model_name):
    return DetrImageProcessor.from_pretrained(model_name)


processor = get_processor(model_name)
model = get_model(num_classes=8, model_name=model_name)  # KITTI dataset has 8 classes
print(model)