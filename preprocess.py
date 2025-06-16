import os
import numpy as np
import torch
import json

from transformers import AutoImageProcessor
from datasets import load_dataset

# === global variables === #
SEED = 42
DATASET_NAME = "KagglingFace/vit-cats-dogs"
MODEL_NAME = "google/vit-base-patch16-224"
OUTPUT_DIR = "./preprocessed_data"
TEST_SIZE = 0.2

torch.manual_seed(SEED)
np.random.seed(SEED)

# === load and split dataset === #
print("Loading dataset...")
ds = load_dataset(DATASET_NAME)
ds = ds['train'].train_test_split(test_size=TEST_SIZE, seed=SEED)
print(f"Train samples: {len(ds['train'])}, Test samples: {len(ds['test'])}")

# === loading image processor === #
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)

# === preprocessing function === #
def preprocess(examples):
    """
    this function preprocess all the 
    images in the dataset for the vit 
    """

    images = [img.convert("RGB") for img in examples["image"]]
    processed = processor(images, return_tensors="pt")
    labels = examples["label"]
    if isinstance(labels[0], str):
        label_map = {"cat": 0, "dog": 1}
        labels = [label_map[label] for label in labels]

    return {
        "pixel_values": processed["pixel_values"],
        "labels": labels
    }

processed_ds = ds.map(
    preprocess,
    batched=True,
    batch_size=32,
    remove_columns=["image"],
    desc="Preprocessing"
)

# === save preprocessed dataset === #
print(f"Saving preprocessed dataset in {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

processed_ds.save_to_disk(OUTPUT_DIR)

processor_path = os.path.join(OUTPUT_DIR, "processor")
processor.save_pretrained(processor_path)

# Save dataset info
dataset_info = {
    "train_samples": len(processed_ds['train']),
    "test_samples": len(processed_ds['test']),
    "num_classes": 2,
    "class_names": ["cat", "dog"],
    "model_name": MODEL_NAME,
    "seed": SEED
}

with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), "w") as f:
    json.dump(dataset_info, f, indent=2)

print("Preprocessing completed successfully!")
print(f"Dataset saved to: {OUTPUT_DIR}")
print(f"Train samples: {dataset_info['train_samples']}")
print(f"Test samples: {dataset_info['test_samples']}")
