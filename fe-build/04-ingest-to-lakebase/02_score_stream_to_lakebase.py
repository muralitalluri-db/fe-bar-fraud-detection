# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Score the Kafka stream and write to Lakebase (real-time mode)
# MAGIC
# MAGIC 1. **Read** the `transactions` Kafka topic and parse the JSON payload.
# MAGIC 2. **Score** each transaction by calling the `fe-bar-fraud-scoring` endpoint from a pandas UDF.
# MAGIC    Only the 7 raw fields are sent; the endpoint looks up the card profile and computes the rest.
# MAGIC 3. **Flag** transactions whose score is at or above the champion version's threshold (0.9).
# MAGIC 4. **Write** to `fe_bar_mt_lakebase.fe_bar.transactions` (Lakebase, registered in Unity Catalog) with
# MAGIC    the native Lakebase sink in real-time mode, upserting on `transaction_id`.
# MAGIC
# MAGIC In real-time mode the UDF runs on small batches (~10 ms), so the endpoint receives many small
# MAGIC requests. If it can't keep up at 1k transactions/second, scale the endpoint up.
# MAGIC
# MAGIC Requires classic compute (Databricks Runtime 18 LTS or above). Run `01_create_lakebase_tables`
# MAGIC first, start this stream, then start `03-kafka-ingest/02_stream_to_kafka`.

# COMMAND ----------

# MAGIC %run ./00_stream_config

# COMMAND ----------

import json
from datetime import datetime, timezone

import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql.streaming import StreamingQueryListener
from pyspark.sql.types import TimestampType

dbutils.widgets.text("clean_checkpoint", "yes")
if dbutils.widgets.get("clean_checkpoint") == "yes":
    dbutils.fs.rm(SCORING_CHECKPOINT, True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model version, threshold, and endpoint access

# COMMAND ----------

champion = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
model_version = champion.version
threshold = float(champion.tags[THRESHOLD_TAG])

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_url = f"https://{workspace_host}/serving-endpoints/{ENDPOINT_NAME}/invocations"
# The notebook's token is passed to the executors so the UDF can call the endpoint.
api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

print(f"Scoring with {MODEL_NAME} version {model_version}, threshold {threshold}")
print(f"Endpoint: {endpoint_url}")

# COMMAND ----------

class CustomStreamingQueryListener(StreamingQueryListener):
    def onQueryStarted(self, event):
        print(f"Query started: id={event.id}, name={event.name}")

    def onQueryProgress(self, event):
        row = event.progress
        print("****************************************** batchId ***********************")
        print(
            f"batchId = {row.batchId} "
            f"timestamp = {row.timestamp} "
            f"numInputRows = {row.numInputRows} "
            f"batchDuration = {row.batchDuration}"
        )
        progress_json = json.loads(row.json)
        latencies = progress_json.get("latencies", {})
        print(json.dumps(latencies, indent=2))

    def onQueryTerminated(self, event):
        print(f"Query terminated: id={event.id}, runId={event.runId}")

try:
    spark.streams.removeListener(CustomStreamingQueryListener())
except Exception:
    pass
spark.streams.addListener(CustomStreamingQueryListener())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read and parse the Kafka topic
# MAGIC
# MAGIC The JSON schema is taken from `transactions_live_stream`, the table the producer publishes from.

# COMMAND ----------

payload_schema = spark.table(fq(LIVE_STREAM_TABLE)).select(*LIVE_PAYLOAD_COLUMNS).schema

transactions = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", dbutils.secrets.get(KAFKA_SECRET_SCOPE, KAFKA_SECRET_KEY))
    .option("subscribe", TXN_TOPIC)
    .option("startingOffsets", "earliest")
    .load()
    .select(
        F.from_json(F.col("value").cast("string"), payload_schema).alias("txn"),
        F.col("timestamp").alias("kafka_ts"),
    )
    .select("txn.*", "kafka_ts")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2–3. Score and flag
# MAGIC
# MAGIC The pandas UDF receives a batch of rows, sends them to the endpoint in one request, and returns
# MAGIC one score per row. Transient endpoint errors are retried. `scored_at` comes from a UDF because
# MAGIC `current_timestamp()` is fixed for the whole (long-running) real-time batch.

# COMMAND ----------

def score_model(dataset):
    import requests

    url = endpoint_url
    headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()


@F.pandas_udf("double")
def fraud_score(card_id: pd.Series, amount_usd: pd.Series, merchant_category: pd.Series, channel: pd.Series,
                latitude: pd.Series, longitude: pd.Series, transaction_ts: pd.Series) -> pd.Series:
    import time

    dataset = pd.DataFrame({
        "card_id": card_id,
        "amount_usd": amount_usd,
        "merchant_category": merchant_category,
        "channel": channel,
        "latitude": latitude,
        "longitude": longitude,
        "transaction_ts": transaction_ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%f"),
    })

    # Retry when the endpoint is throttling or briefly unavailable (common under real-time load).
    for attempt in range(5):
        try:
            return pd.Series(score_model(dataset)["predictions"], dtype="float64")
        except Exception as e:
            if attempt == 4 or not any(f"status {code}" in str(e) for code in (429, 500, 502, 503, 504)):
                raise
            time.sleep(2 ** attempt)


@F.udf(returnType=TimestampType())
def current_timestamp_udf():
    return datetime.now(timezone.utc)


scored = (
    transactions
    .withColumn("fraud_score", fraud_score(*[F.col(c) for c in SCORING_REQUEST_COLUMNS]))
    .withColumn("is_flagged", F.col("fraud_score") >= F.lit(threshold))
    .withColumn("model_version", F.lit(str(model_version)))
    .withColumn("scored_at", current_timestamp_udf())
    .select(*SCORED_COLUMNS)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write to the Unity Catalog-registered Lakebase table

# COMMAND ----------

query = (
    scored
    .writeStream
    .queryName(SCORING_QUERY_NAME)
    .option("upsertkey", "transaction_id")
    .option("batchinterval", "10 milliseconds")
    .option("checkpointLocation", SCORING_CHECKPOINT)
    .trigger(realTime="5 minutes")
    .outputMode("update")
    .toTable(TRANSACTIONS_UC_TABLE)
)
