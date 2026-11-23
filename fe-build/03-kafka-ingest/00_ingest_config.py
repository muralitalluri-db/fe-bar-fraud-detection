# Databricks notebook source
# MAGIC %md
# MAGIC # Kafka ingest config
# MAGIC
# MAGIC Shared by the notebooks in this folder through `%run ./00_ingest_config`. Pulls in the data
# MAGIC generation config so table names and payload columns have a single source of truth.

# COMMAND ----------

# MAGIC %run ../01-data-generation/00_config

# COMMAND ----------

KAFKA_SECRET_SCOPE = "dais-2025-json-parsing-demo"
KAFKA_SECRET_KEY = "prime-schema-from-json"

TXN_TOPIC = "transactions"

CHECKPOINT_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints"
