import os
import random
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrImageProcessor, DetrForObjectDetection
from dataset import KittiSequenceDataset

# --- CONFIGURATION ---
CHECKPOINT  = "/projectnb/ec523/students/serhat/detr_finetuning/outputs/checkpoint_best.pth"
IMG_DIR     = "/projectnb/ec523/projects/proj_adversarial_weather/kitti/training/image_02"
LBL_DIR     = "/projectnb/ec523/projects/proj_adversarial_weather/kitti/training/label_02"
OUT_DIR     = "/projectnb/ec523/students/serhat/detr_finetuning/outputs/inference"
NUM_IMAGES  = 10
THRESHOLD   = 0.5   # Confidence threshold — standard for DETR

CLASS_NAMES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

# One distinct color per class
CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45"
]

os.makedirs(OUT_DIR, exist_ok=True)

# --- MODEL SETUP ---
model_name = "facebook/detr-resnet-50"
num_classes = 8

processor = DetrImageProcessor.from_pretrained(model_name, size={"width": 640, "height": 384})
model = DetrForObjectDetection.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading checkpoint from {CHECKPOINT}...")
checkpoint = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")

# --- LOAD VALIDATION DATASET ---
# Use val sequences (13-16) — same split as training
val_dataset = KittiSequenceDataset(IMG_DIR, LBL_DIR, processor, sequence_ids=range(13, 17))

# Pick 10 random samples
random.seed(42)
indices = random.sample(range(len(val_dataset)), NUM_IMAGES)

# --- INFERENCE ---
print(f"\nRunning inference on {NUM_IMAGES} images...")

for i, idx in enumerate(indices):
    # Load the raw image (before processor resizing) for drawing
    img_path = val_dataset.samples[idx]["img_path"]
    raw_image = Image.open(img_path).convert("RGB")

    # Process the image for the model
    encoding = processor(images=raw_image, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device)
    pixel_mask   = torch.ones(1, pixel_values.shape[2], pixel_values.shape[3], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    # Convert raw outputs to boxes and scores
    # Target sizes: original image size for correct box scaling
    target_sizes = torch.tensor([raw_image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(
        outputs, threshold=THRESHOLD, target_sizes=target_sizes
    )[0]

    scores = results["scores"].cpu()
    labels = results["labels"].cpu()
    boxes  = results["boxes"].cpu()

    # Draw on the raw image
    draw = ImageDraw.Draw(raw_image)

    for score, label, box in zip(scores, labels, boxes):
        class_idx = label.item()
        # Guard against unexpected label indices
        if class_idx >= len(CLASS_NAMES):
            continue
        color = CLASS_COLORS[class_idx]
        class_name = CLASS_NAMES[class_idx]
        x1, y1, x2, y2 = box.tolist()

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background and text
        text = f"{class_name} {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), text)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
        draw.text((x1, y1), text, fill="white")

    # Save
    seq_name = img_path.split("/")[-2]
    frame_name = img_path.split("/")[-1].replace(".png", "")
    out_path = os.path.join(OUT_DIR, f"{seq_name}_{frame_name}_detections.png")
    raw_image.save(out_path)

    num_detections = len(scores)
    print(f"  [{i+1}/{NUM_IMAGES}] {seq_name}/{frame_name} — {num_detections} detections — saved to {out_path}")

print(f"\nDone. Images saved to {OUT_DIR}")
