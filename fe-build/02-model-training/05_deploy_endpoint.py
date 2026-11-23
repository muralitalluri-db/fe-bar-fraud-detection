# Databricks notebook source
# MAGIC %md
# MAGIC # 5. Deploy the model to a serving endpoint
# MAGIC
# MAGIC 1. **Find** the version the `champion` alias points to.
# MAGIC 2. **Create or update** the Model Serving endpoint `fe-bar-fraud-scoring` with that version. The
# MAGIC    endpoint follows the model's feature spec: looks up `card_profile_online` in Lakebase by
# MAGIC    `card_id`, runs the 7 feature functions, then calls `FraudModel`.
# MAGIC 3. **Inference table**: every request and response is logged to `fraud_scoring_payload`.
# MAGIC 4. **Test** with known fraud and normal transactions from `transactions_live`.
# MAGIC
# MAGIC Run after `04_lakebase_online_store`; the endpoint needs the online table to exist. To serve a new
# MAGIC version later, move the `champion` alias and re-run this notebook.

# COMMAND ----------

# MAGIC %run ./00_ml_config

# COMMAND ----------

import time
from datetime import timedelta

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from mlflow.tracking import MlflowClient

mlflow_client = MlflowClient(registry_uri="databricks-uc")
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Find the champion version

# COMMAND ----------

champion = mlflow_client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
threshold = float(champion.tags[THRESHOLD_TAG])
print(f"{MODEL_NAME}@{CHAMPION_ALIAS} -> version {champion.version}, {THRESHOLD_TAG} = {threshold}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2–3. Create or update the endpoint
# MAGIC
# MAGIC `Medium` with scale-to-zero off keeps it warm for the live demo and handles the many small requests
# MAGIC the real-time stream sends at ~1k transactions per second. Creating it can take 20–40 minutes.

# COMMAND ----------

served_entities = [
    ServedEntityInput(
        entity_name=MODEL_NAME,
        entity_version=champion.version,
        workload_size="Medium",
        scale_to_zero_enabled=False,
    )
]

existing = next((e for e in w.serving_endpoints.list() if e.name == ENDPOINT_NAME), None)

if existing is None:
    print(f"Creating endpoint {ENDPOINT_NAME} with version {champion.version}...")
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(served_entities=served_entities),
        ai_gateway=AiGatewayConfig(
            inference_table_config=AiGatewayInferenceTableConfig(
                catalog_name=CATALOG,
                schema_name=SCHEMA,
                table_name_prefix=INFERENCE_TABLE_PREFIX,
                enabled=True,
            )
        ),
        timeout=timedelta(minutes=60),
    )
else:
    print(f"Updating endpoint {ENDPOINT_NAME} to version {champion.version}...")
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=served_entities,
        timeout=timedelta(minutes=60),
    )

endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
print(f"Endpoint {endpoint.name}: ready = {endpoint.state.ready}, config update = {endpoint.state.config_update}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test with known transactions
# MAGIC
# MAGIC One fraud transaction of each type plus five normal ones from the live window. Only the raw fields
# MAGIC in `SCORING_REQUEST_COLUMNS` are sent, exactly like the stream will. `is_fraud` and `fraud_type`
# MAGIC stay here for comparing.

# COMMAND ----------

live = spark.table(fq("transactions_live"))

fraud_samples = (
    live.filter("is_fraud")
    .withColumn("rn", F.row_number().over(Window.partitionBy("fraud_type").orderBy("transaction_id")))
    .filter("rn = 1")
    .drop("rn")
)
normal_samples = live.filter("not is_fraud").orderBy("transaction_id").limit(5)

samples = (
    fraud_samples.unionByName(normal_samples)
    .select("transaction_id", *SCORING_REQUEST_COLUMNS, LABEL, "fraud_type")
    .toPandas()
)

request_records = (
    samples[SCORING_REQUEST_COLUMNS]
    .assign(transaction_ts=lambda d: d["transaction_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f"))
    .to_dict("records")
)

started = time.time()
response = w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=request_records)
elapsed_ms = (time.time() - started) * 1000

results = samples[["transaction_id", "fraud_type", LABEL, "amount_usd", "merchant_category", "channel"]].assign(
    score=[round(float(s), 4) for s in response.predictions],
)
results["flagged"] = results["score"] >= threshold
print(f"Scored {len(results)} transactions in {elapsed_ms:.0f} ms (one request), threshold {threshold}")
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference table
# MAGIC
# MAGIC Requests and responses land in `fraud_scoring_payload`, usually within an hour of the call.

# COMMAND ----------

payload_table = fq(f"{INFERENCE_TABLE_PREFIX}_payload")
if spark.catalog.tableExists(payload_table):
    display(spark.table(payload_table).orderBy(F.col("request_time").desc()).limit(10))
else:
    print(f"{payload_table} doesn't exist yet; check back after the first requests are logged")
