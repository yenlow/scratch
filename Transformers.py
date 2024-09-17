# Databricks notebook source
# https://mlflow.org/docs/latest/python_api/mlflow.transformers.html
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/training.ipynb#scrollTo=luHOKSaduaZF
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, TrainingArguments, Trainer
from datasets import load_dataset
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data and model from HF

# COMMAND ----------

# Get model from HF
architecture = "bert-base-cased"
artifact_path = "model"
task="text-classification"
max_length = 128

model = AutoModelForSequenceClassification.from_pretrained(architecture,  num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(architecture)

# Process tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', 
                     truncation=True, max_length=max_length)

# COMMAND ----------

# Just download the smaller test split to save time
dataset = load_dataset("yelp_review_full")

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# COMMAND ----------

from collections import Counter
dict(Counter(eval_dataset['label']))

# COMMAND ----------

# Method 1: Using pipeline
gen_pipeline = pipeline(
    task=task,
    tokenizer=tokenizer,
    model=model,
)

# Test pipeline
# https://huggingface.co/csarron/mobilebert-uncased-squad-v2
queries = ["Unfortunately, the frustration of being Dr. Goldberg's patient is a repeat of the experience I've had with so many other doctors in NYC", 
           "Dr. Eric Goldberg is a fantastic doctor who has correctly diagnosed every issue that my wife and I have had. Unlike many of my past doctors, Dr. Goldberg is very accessible and we have been able to schedule appointments with him and his staff very quickly. We are happy to have him in the neighborhood and look forward to being his patients for many years to come."]
gen_pipeline(queries)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and save model to DBFS via `mlflow`
# MAGIC Optionally register model to UC

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.transformers.log_model(
        transformers_model=gen_pipeline,
        artifact_path=artifact_path,
        registered_model_name="transformer_model"
    )

# Method 2: Using components dict
# with mlflow.start_run() as run:
#     components = {
#         "model": model,
#         "tokenizer": tokenizer,
#     }
#     mlflow.transformers.log_model(
#         transformers_model=components,
#         artifact_path=artifact_path,
#     )
run.info

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load model via `mlflow`
# MAGIC ### 1. For inferencing

# COMMAND ----------

#run_id = "bb6c42956a7c4a64bdb5fe5d45ec36d0"
run_id = run.info.run_id
model_uri = f"runs:/{run_id}/{artifact_path}"
loaded_model = mlflow.transformers.load_model(model_uri)

# Does inferencing
loaded_model.predict(queries)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. For fine-tuning

# COMMAND ----------

training_output_dir = "transformer_model_ft"
#base_model = model #works
base_model = loaded_model.model #does not work

training_args = TrainingArguments(
  output_dir=training_output_dir,
  eval_strategy="steps",
  eval_steps=5,
  max_steps=10,
  logging_steps=2,
#  fp16=True,
  per_device_train_batch_size=8
)

trainer = Trainer(
  model=base_model,
  args=training_args,
  train_dataset=train_dataset,
  eval_dataset=eval_dataset,
)

# COMMAND ----------

type(model), type(loaded_model.model)

# COMMAND ----------

set(dir(model)) - set(dir(loaded_model.model))

# COMMAND ----------

#mlflow.end_run()
with mlflow.start_run() as run_ft:
  trainer.train()
  ft_pipeline = pipeline(task, model=trainer.model, tokenizer=tokenizer)
  transformers_model=ft_pipeline
  artifact_path=f"{artifact_path}_ft"
  registered_model_name="transformer_model_ft"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Use Finetuned model for inferencing

# COMMAND ----------

ft_pipeline(queries)

# COMMAND ----------


