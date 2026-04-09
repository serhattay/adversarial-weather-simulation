from transformers import DetrImageProcessor
from dataset import KittiSequenceDataset

p = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', size={'width': 640, 'height': 384})
ds = KittiSequenceDataset('/projectnb/ec523/projects/proj_adversarial_weather/kitti/training/image_02', '/projectnb/ec523/projects/proj_adversarial_weather/kitti/training/label_02', p,
sequence_ids=range(0, 1))
print(ds[0][0].shape)
