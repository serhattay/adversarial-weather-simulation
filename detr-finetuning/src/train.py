import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import KittiSequenceDataset
from transformers import DetrImageProcessor, DetrForObjectDetection

# --- 1. ARGUMENT PARSING ---
# Paths are passed as arguments so the script works correctly regardless of
# the working directory when the SLURM job runs.
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir",    type=str, required=True, help="Absolute path to image_02/")
parser.add_argument("--lbl_dir",    type=str, required=True, help="Absolute path to label_02/")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
parser.add_argument("--epochs",     type=int, default=10)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers",type=int, default=4)
parser.add_argument("--resume",     action="store_true", help="Resume from checkpoint_latest.pth")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# --- 2. MODEL SETUP ---
model_name = "facebook/detr-resnet-50"
num_classes = 8

# Image size matches the 640x384 resolution specified in the project proposal.
# This is also consistent with the depth maps generated in Phase 2.
processor = DetrImageProcessor.from_pretrained(model_name, size={"width": 640, "height": 384})
# Passing num_labels replaces the classifier head AND updates the loss function's
# internal empty_weight tensor to match our 8 classes + 1 background consistently.
model = DetrForObjectDetection.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Lets cuDNN auto-tune the fastest convolution algorithms for our fixed input size.
torch.backends.cudnn.benchmark = True

def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    # All images are resized to exactly 640x384 so no padding is needed.
    # pixel_mask: 1 = real pixel, 0 = padding. All 1s since there is no padding.
    pixel_mask = torch.ones(pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels
    }

# --- 3. DATA SPLITS ---
print("Initializing Datasets...")
# Python's range(x, y) stops BEFORE y. So range(0, 13) includes 0 through 12.
train_dataset = KittiSequenceDataset(args.img_dir, args.lbl_dir, processor, sequence_ids=range(0, 13))
val_dataset   = KittiSequenceDataset(args.img_dir, args.lbl_dir, processor, sequence_ids=range(13, 17))
# test_dataset is reserved for Phase 2 evaluation — not loaded during training runs.
# test_dataset = KittiSequenceDataset(args.img_dir, args.lbl_dir, processor, sequence_ids=range(17, 21))

# num_workers: parallel CPU workers that prefetch data so the GPU never idles.
# pin_memory: pins tensors to page-locked memory for faster CPU->GPU transfers.
train_dataloader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    persistent_workers=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    persistent_workers=True
)

# --- 4. OPTIMIZER ---
param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 1e-5},
]
optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)

# --- 5. RESUME LOGIC ---
start_epoch = 0
best_val_loss = float("inf")

if args.resume:
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed from epoch {start_epoch} (best val loss so far: {best_val_loss:.4f})")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

# --- 6. TRAINING & VALIDATION LOOP ---
print(f"\nStarting training on {device} for {args.epochs} epochs (from epoch {start_epoch + 1})...")

for epoch in range(start_epoch + 1, args.epochs + 1):
    # -- TRAINING PHASE --
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_dataloader):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask   = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        optimizer.zero_grad(set_to_none=True)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(train_dataloader)} | Loss: {loss.item():.4f}")

    # -- VALIDATION PHASE --
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask   = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            val_loss += outputs.loss.item()

    avg_train_loss = train_loss / len(train_dataloader)
    avg_val_loss   = val_loss   / len(val_dataloader)

    print(f"\n--- Epoch {epoch}/{args.epochs} Summary ---")
    print(f"Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

    # -- CHECKPOINT SAVING --
    # Always overwrite checkpoint_latest.pth so we can resume from here if the job is cut short.
    latest_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss":           avg_train_loss,
        "val_loss":             avg_val_loss,
        "best_val_loss":        best_val_loss,
    }, latest_path)

    # Save a separate copy whenever validation loss improves.
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_path = os.path.join(args.output_dir, "checkpoint_best.pth")
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss":           avg_train_loss,
            "val_loss":             avg_val_loss,
            "best_val_loss":        best_val_loss,
        }, best_path)
        print(f"  --> New best model saved (val_loss={best_val_loss:.4f})")

    print()

print("Training complete.")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Checkpoints saved to: {args.output_dir}")
