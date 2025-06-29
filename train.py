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
import json

from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForImageClassification, AutoImageProcessor, TrainingArguments, Trainer, DefaultDataCollator
from peft import LoraConfig, TaskType, get_peft_model

# === global variables === #
SEED = 42
EPOCHS = 5
LR = 5e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
LOGS = "./logs"
PREPROCESSED_DATA_DIR = "./preprocessed_data"
OUTPUT_DIR = "./vit_fine_tuned"
FINAL_MODEL_DIR = "./model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Using device: {DEVICE}")

load_dotenv()

# === wandb logging === #
wandb.login(key=os.getenv("WANDB_API_KEY"))

# === load the dataset === #
if not os.path.exists(PREPROCESSED_DATA_DIR):
    raise FileNotFoundError(f"Preprocessed data directory not found: {PREPROCESSED_DATA_DIR}")

ds = load_from_disk(PREPROCESSED_DATA_DIR)

with open(os.path.join(PREPROCESSED_DATA_DIR, "dataset_info.json"), "r") as f:
    dataset_info = json.load(f)

print(f"Train samples: {dataset_info['train_samples']}")
print(f"Test samples: {dataset_info['test_samples']}")
print(f"Classes: {dataset_info['class_names']}")

# === load the model to finetune === #
processor_path = os.path.join(PREPROCESSED_DATA_DIR, "processor")
processor = AutoImageProcessor.from_pretrained(processor_path)
model_id = AutoModelForImageClassification.from_pretrained(
    dataset_info['model_name']
)

# === load and setup LoRA for peft === #
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
    target_modules=["query", "key", "value"]
)

model = get_peft_model(model_id, peft_config)
model.print_trainable_parameters()

# === training loop === #
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    metric_for_best_model="eval_accuracy",
    num_train_epochs=EPOCHS,
    logging_dir=LOGS,
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    report_to="wandb", 
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    processing_class=processor,  
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
    print(f"Saving final model to {FINAL_MODEL_DIR}...")
    trainer.save_model(FINAL_MODEL_DIR)
    processor.save_pretrained(FINAL_MODEL_DIR)

    # Save training info
    training_info = {
        "final_metrics": metrics,
        "training_args": training_args.to_dict(),
        "model_name": dataset_info['model_name'],
        "dataset_info": dataset_info
    }
    
    with open(os.path.join(FINAL_MODEL_DIR, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2, default=str)

    print("Training completed successfully!")
    print(f"Final model saved to: {FINAL_MODEL_DIR}")

except Exception as e:
    print(f"Training failed with error: {e}")
    raise

finally:
    wandb.finish()
