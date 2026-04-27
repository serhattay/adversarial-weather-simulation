import torch
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection

# PATHS
CHECKPOINT = "/projectnb/ec523/students/mz314/checkpoints/checkpoint_best.pth"
IMAGE_PATH = "/projectnb/ec523/projects/proj_adversarial_weather/kitti/training/image_02/0000/000000.png"

# MODEL SETUP
model_name = "facebook/detr-resnet-50"
num_classes = 8

processor = DetrImageProcessor.from_pretrained(
    model_name, size={"width": 640, "height": 384}
)

model = DetrForObjectDetection.from_pretrained(
    model_name, num_labels=num_classes, ignore_mismatched_sizes=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# LOAD CHECKPOINT
checkpoint = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded")

# LOAD IMAGE
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

inputs = processor(images=image_rgb, return_tensors="pt").to(device)

# INFERENCE
with torch.no_grad():
    outputs = model(**inputs)

print("Inference ran successfully")
