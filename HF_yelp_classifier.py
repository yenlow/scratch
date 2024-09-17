# Databricks notebook source
pip install -U transformers==4.44 accelerate

# COMMAND ----------

!pip show accelerate

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

#################################################################################
#   This script validates multi-GPU training.
#   Usage: python multigpu_trainer.py
#
#   When this script is running, open another terminal and run `nvidia-smi`
#   If your cuda installation was successful, you should see all GPUs on your
#   instance being utilized. More importantly, this script should complete in ~1 minute
#################################################################################

# Load data
dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Tokenize and split
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))


# Load and train the model
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased",
    num_labels=5,
)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# COMMAND ----------

training_args = TrainingArguments(
    output_dir="test_trainer",
    eval_strategy="epoch",
    do_train=True,
    half_precision_backend="cpu_amp",
    save_strategy='no'
    #gradient_checkpointing=True,
    #gradient_checkpointing_kwargs={"use_reentrant": False},
)

# print('-------TRAINING ARGS--------')
# print(training_args)
# print('------------------------------------------------')
with open("simple_training_args.json", "w") as f:
    f.write(training_args.to_json_string())

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

with torch.cuda.amp.autocast():
    trainer.train()

# COMMAND ----------


