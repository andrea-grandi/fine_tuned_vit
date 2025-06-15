import os
import math
import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import tqdm
import evaluate
import wandb

from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoModelForImageClassification, AutoImageProcessor, TrainingArguments, Trainer, DefaultDataCollator
from peft import LoraConfig, TaskType, get_peft_model

# === global variables === #
SEED = torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#TODO


# === download the dataset and split it === #
ds = load_dataset("KagglingFace/vit-cats-dogs") # this dataset has only the train split
ds = ds['train'].train_test_split(test_size=0.2)

# === load the model to finetune === #
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True) # this is (like) the tokenizer
model_id = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

# === load and setup LoRA for peft === #
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"]
)

model = get_peft_model(model_id, peft_config)

# === prepare dataset for training === #
def preprocess(example):
    processed = processor(example['image'], return_tensors="pt")
    example["pixel_values"] = processed["pixel_values"].squeeze()
    return example

ds = ds.map(preprocess, batched=False)

# === training loop === #
ds.set_format(type="torch", columns=["pixel_values", "label"])

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.Tensor(logits).argmax(dim=-1)
    return accuracy.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="./vit-cats-dogs",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    remove_unused_columns=False,  # Important for image data!
    report_to="wandb",  # avoid needing wandb etc.
    run_name="vit-fine-tuning"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    tokenizer=processor,  # optional
    data_collator=DefaultDataCollator(),  # handles padding etc.
    compute_metrics=compute_metrics,
)

# train
trainer.train()

#evaluate
metrics = trainer.evaluate()
print(metrics)
