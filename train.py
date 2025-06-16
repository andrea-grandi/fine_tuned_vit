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

from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoModelForImageClassification, AutoImageProcessor, TrainingArguments, Trainer, DefaultDataCollator
from peft import LoraConfig, TaskType, get_peft_model

# === global variables === #
SEED = 42
EPOCHS = 5
LR = 5e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Using device: {DEVICE}")

load_dotenv()

# === wandb logging === #
wandb.login(key=os.getenv("WANDB_API_KEY"))

# === download the dataset and split it === #
ds = load_dataset("KagglingFace/vit-cats-dogs") # this dataset has only the train split
ds = ds['train'].train_test_split(test_size=0.2)
print(f"Train samples: {len(ds['train'])}, Test samples: {len(ds['test'])}")
#print(ds)

# === load the model to finetune === #
processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224",
    use_fast=True
) # this is (like) the tokenizer
model_id = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)

# === load and setup LoRA for peft === #
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"]
)

model = get_peft_model(model_id, peft_config)
model.print_trainable_parameters()

# === prepare dataset for training === #
def preprocess(examples):
    images = [img.convert("RGB") for img in examples['image']]
    processed = processor(images, return_tensors="pt")
    labels = examples['label']
    if isinstance(labels[0], str):
        label_map = {"cat": 0, "dog": 1}
        labels = [label_map[label] for label in labels]

    return {
        "pixel_values": processed["pixel_values"],
        "labels": labels
    }

ds = ds.map(preprocess, batched=True, batch_size=32, remove_columns=["image"])

# === training loop === #
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="./vit-cats-dogs",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=EPOCHS,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    remove_unused_columns=False,
    report_to="wandb", 
    run_name="vit-fine-tuning",
    seed=SEED,
    data_seed=SEED,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    gradient_checkpointing=True,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    learning_rate=LR,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    tokenizer=processor,  
    data_collator=DefaultDataCollator(), 
    compute_metrics=compute_metrics,
)

# train
print("Start training")
try:
    trainer.train()

    # evaluate
    print("Evaluating model...")
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        print(f" {key}: {value}")
    
    # save the model
    trainer.save_model("./vit-cats-dogs-final")
    processor.save_pretrained("./vit-cats-dogs-final")

    print("Training completed succesfully")

except Exception as e:
    print(f"Training failed with error: {e}")
    raise

finally:
    wandb.finish()
