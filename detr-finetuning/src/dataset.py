import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class KittiSequenceDataset(Dataset):
    def __init__(self, image_base_dir, label_base_dir, processor, sequence_ids):
        """
        sequence_ids: A list or range of sequence numbers (e.g., range(0, 13) for 0000-0012)
        """
        self.processor = processor
        self.samples = [] 
        
        class_map = {
            'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
            'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7
        }
        
        # Loop ONLY through the specific sequences assigned to this split
        for seq_id in sequence_ids:
            seq_name = f"{seq_id:04d}" 
            seq_img_dir = os.path.join(image_base_dir, seq_name)
            seq_label_file = os.path.join(label_base_dir, f"{seq_name}.txt")
            
            frame_labels = {} 
            
            # Parse the text file
            if os.path.exists(seq_label_file):
                with open(seq_label_file, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        frame_idx = int(parts[0])
                        class_name = parts[2]     
                        
                        if class_name == 'DontCare' or class_name not in class_map:
                            continue
                            
                        class_id = class_map[class_name]
                        x_min, y_min, x_max, y_max = map(float, parts[6:10])
                        
                        if frame_idx not in frame_labels:
                            frame_labels[frame_idx] = {'boxes': [], 'classes': []}
                            
                        frame_labels[frame_idx]['boxes'].append([x_min, y_min, x_max, y_max])
                        frame_labels[frame_idx]['classes'].append(class_id)
            
            # Match to PNGs
            if os.path.exists(seq_img_dir):
                img_files = sorted([f for f in os.listdir(seq_img_dir) if f.endswith('.png')])
                for img_file in img_files:
                    frame_idx = int(img_file.split('.')[0])
                    img_path = os.path.join(seq_img_dir, img_file)
                    labels_for_frame = frame_labels.get(frame_idx, {'boxes': [], 'classes': []})
                    
                    self.samples.append({
                        'img_path': img_path,
                        'boxes': labels_for_frame['boxes'],
                        'classes': labels_for_frame['classes']
                    })
                    
        print(f"Loaded {len(self.samples)} frames for sequences {list(sequence_ids)}.")

    # __len__ and __getitem__ remain exactly the same as before!
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['img_path']).convert("RGB")
        
        # 1. Build the COCO-formatted annotations list
        annotations = []
        for i in range(len(sample['boxes'])):
            x_min, y_min, x_max, y_max = sample['boxes'][i]
            class_id = sample['classes'][i]
            
            # Convert KITTI's max coords to COCO's width/height
            width = x_max - x_min
            height = y_max - y_min
            
            # The processor specifically looks for 'bbox' and 'category_id'
            annotations.append({
                "id": i,                     # Unique ID for the box
                "category_id": class_id,
                "bbox": [x_min, y_min, width, height],
                "area": width * height,      # DETR uses area to sort bounding boxes
                "iscrowd": 0                 # Required by COCO format (0 = single object)
            })
            
        # 2. Package it into the target dictionary the processor expects
        target = {
            "image_id": idx,
            "annotations": annotations
        }
        
        # 3. The processor handles resizing, padding, AND normalizing the bounding boxes!
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        return encoding["pixel_values"].squeeze(), encoding["labels"][0]