import torch
import cv2
import os

# Load MiDaS 
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load transforms
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.dpt_transform

input_dir = "/projectnb/ec523/projects/proj_adversarial_weather/kitti/training/image_02/0000"
output_dir = "/projectnb/ec523/students/mz314/depth/0000"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if not file.endswith(".png"):
        continue

    img_path = os.path.join(input_dir, file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # Normalize
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    out_path = os.path.join(output_dir, file)
    cv2.imwrite(out_path, (depth * 255).astype("uint8"))

print("Depth maps generated.")
