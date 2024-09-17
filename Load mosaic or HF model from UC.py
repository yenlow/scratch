# Databricks notebook source
import mlflow, os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from mlflow import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register mlflow experimental model

# COMMAND ----------

run_id = "81daa482a59b4aa08c9fcae82f37a1c4"
run_name = "llama2-7b-mct-iMzpi2"
# model_uri = f"runs:/{run_id}/model"
model_uri = "dbfs:/databricks/mlflow-tracking/2107924853762878/81daa482a59b4aa08c9fcae82f37a1c4/artifacts/checkpoints"
mlflow.register_model(model_uri, f"yenlow.mcli.{run_name}")

# COMMAND ----------

#dbfs:/databricks/mlflow-tracking/2107924853762878/81daa482a59b4aa08c9fcae82f37a1c4/artifacts/checkpoints
model_path = "/Volumes/yenlow/mcli/models/llama2-finetune-pxrmd6/huggingface/ba2/"
dbutils.fs.ls(f"dbfs:{model_path}")

# COMMAND ----------

!cat /Volumes/yenlow/mcli/models/llama2-finetune-pxrmd6/huggingface/ba2/config.json

# COMMAND ----------

import torch
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    local_files_only=True,
    use_safetensors=True,
    config=f"{model_path}/config.json",
   torch_dtype=torch.bfloat16,
   device_map="cuda:0",
#   hf_quantizer = None
)

# COMMAND ----------

from safetensors import safe_open
checkpoint_file = f"{model_path}model-00003-of-00003.safetensors"
with safe_open(checkpoint_file, framework="pt") as f:
  metadata = f.metadata()


# COMMAND ----------

from safetensors.torch import load_file as safe_load_file
loaded = safe_load_file(checkpoint_file)

# COMMAND ----------

loaded.keys()

# COMMAND ----------

login("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")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer, model

# COMMAND ----------

type(tokenizer), type(model)

# COMMAND ----------


run_id = "81daa482a59b4aa08c9fcae82f37a1c4"
client = MlflowClient()
local_dest = "models"
local_path = client.download_artifacts(run_id, "checkpoints/huggingface/ba2",local_dest)

# COMMAND ----------

model_loaded = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="")

# COMMAND ----------

artifact_path: model
flavors:
  python_function:
    loader_module: mlflow.transformers
    python_version: 3.11.9
  transformers:
    code: null
    components:
    - tokenizer
    framework: pt
#    instance_type: TextClassificationPipeline
    model_binary: model
    pipeline_model_type: LlamaForCausalLM
    source_model_name: bert-base-cased
    task: text-generation
    tokenizer_type: LlamaTokenizerFast
    torch_dtype: torch.float32
    transformers_version: 4.41.2
mlflow_version: 2.14.3
#model_uuid: 0cf510c002824837a1a4320dc9ed82e6
run_id: 81daa482a59b4aa08c9fcae82f37a1c4

# COMMAND ----------

mlflow.log_text("text string", "MLmodel")

# COMMAND ----------

#run_id = run.info.run_id
run_id = "456c65b8a2ba4978ba595e94b1bd6548"
artifact_path = "checkpoints/huggingface"
model_uri = f"runs:/{run_id}/{artifact_path}"
loaded_model = mlflow.transformers.load_model(model_uri)

# Does inferencing
loaded_model.predict("The answer to life, the universe, and happiness is")

# COMMAND ----------


