# Databricks notebook source
# MAGIC %md
# MAGIC Since C4 (HF [legaacy-dataset/c4](https://huggingface.co/datasets/legacy-datasets/c4)) is already in a spark table, do [this](https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/spark_dataframe_to_mds.html).<br>
# MAGIC Else, if source are partitioned files, use [this](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#notebook/2107924853724368/command/2107924853724374)

# COMMAND ----------

# MAGIC %pip install mosaicml-streaming==0.8.0 datasets
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from streaming import MDSWriter
from streaming.base.storage import S3Uploader, download_from_databricks_unity_catalog
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, contains, when, collect_list, current_timestamp
#import hashlib
from datasets import Dataset
import os
import shutil
#from typing import Any, Sequence, Dict, Iterable, Optional
import pandas as pd
import numpy as np
from tempfile import mkdtemp
# import datasets as hf_datasets
# from transformers import AutoTokenizer, PreTrainedTokenizerBase
from streaming.base.converters import dataframe_to_mds

# COMMAND ----------

##Settings to change
local_dir = "/Volumes/datasets/mds/c4"
columns = {
    "text": "str",
}

# COMMAND ----------

source_df = spark.sql("""
    SELECT text FROM main.redpajama_v1.c4 
    where language='en'
    and split='train'
    """)
display(source_df.limit(5))

# COMMAND ----------

# Empty the MDS output directory
out_path = os.path.join(local_dir, 'mds')
shutil.rmtree(out_path, ignore_errors=True)

# Convert the dataset to an MDS format. It divides the dataframe into 4 parts, one parts per worker and merge the `index.json` from 4 sub-parts into one in a parent directory.
dataframe_to_mds(source_df.repartition(32), 
                 merge_index=True, 
                 mds_kwargs={'out': out_path, 
                             'compression': 'zstd',
                             'columns': columns})

# COMMAND ----------

# MAGIC %ls {out_path}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transfer from Databricks to S3

# COMMAND ----------

# Set AWS credentials
AWS_ACCESS_KEY_ID="redacted"
AWS_SECRET_ACCESS_KEY="redacted"
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.access.key", AWS_ACCESS_KEY_ID)
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)

# View S3 from databricks
bucket_name = "yen-mcli"
s3_out = f"s3://{bucket_name}/merck-bert/c4/train"
dbutils.fs.ls(f"s3a://{bucket_name}/")

# COMMAND ----------

# To download file locally
#os.getcwd() #/Workspace/Users/yen.low@databricks.com
#download_from_databricks_unity_catalog(f"dbfs:{out_path}/11054/index.json", "tmp")

# COMMAND ----------

local_mnt = f"/mnt/s3/{bucket_name}"
uc_shards = f"dbfs:{out_path}"

# Mount S3 onto DBFS
#dbutils.fs.mount(f"s3a://{bucket_name}/", local_mnt)
dbutils.fs.ls(local_mnt)

# COMMAND ----------

dbutils.fs.cp(uc_shards, f"{local_mnt}/merck-bert/train", True)

# COMMAND ----------

dbutils.fs.cp('dbfs:/mnt/s3/yen-mcli/merck-bert/train/index.json', f"s3a://{bucket_name}/tmp")

# COMMAND ----------

# Check if file in mounted s3
dbutils.fs.ls(f"{local_mnt}/merck-bert/train")

# COMMAND ----------

# Check if file in s3
dbutils.fs.ls(f"s3a://{bucket_name}/merck-bert/train")

# COMMAND ----------

dbutils.fs.unmount(local_mnt)
