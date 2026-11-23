# Databricks notebook source
# MAGIC %md
# MAGIC # Stream the live stream table into Kafka
# MAGIC
# MAGIC Reads `transactions_live_stream` one file (one second, ~1k transactions) per micro-batch on a
# MAGIC 1-second trigger and publishes each transaction as JSON to the `transactions` topic.
# MAGIC The message key is `card_id`, so each card's transactions stay in order on one partition.

# COMMAND ----------

# MAGIC %run ./00_ingest_config

# COMMAND ----------

from pyspark.sql.streaming import StreamingQueryListener

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.checkpoints")

kafka_bootstrap_servers_plaintext = dbutils.secrets.get(KAFKA_SECRET_SCOPE, KAFKA_SECRET_KEY)
checkpoint_path = f"{CHECKPOINT_VOLUME}/kafka_ingest_{TXN_TOPIC}"

dbutils.widgets.text("clean_checkpoint", "yes")
clean_checkpoint = dbutils.widgets.get("clean_checkpoint")
if clean_checkpoint == "yes":
    dbutils.fs.rm(checkpoint_path, True)

# COMMAND ----------

class MyStreamingListener(StreamingQueryListener):
    def onQueryStarted(self, event):
        print(f"'{event.name}' [{event.id}] got started!")
    def onQueryProgress(self, event):
        row = event.progress
        print(f"****************************************** batchId ***********************")
        print(f"batchId = {row.batchId} timestamp = {row.timestamp} numInputRows = {row.numInputRows} batchDuration = {row.batchDuration}")
    def onQueryTerminated(self, event):
        print(f"{event.id} got terminated!")
try:
    spark.streams.removeListener(MyStreamingListener())
except:
    pass
spark.streams.addListener(MyStreamingListener())

# COMMAND ----------

txn_stream_df = (
    spark.readStream
    .format("delta")
    .option("maxFilesPerTrigger", 1)
    .table(fq(LIVE_STREAM_TABLE))
    .select(
        F.col("card_id").cast("binary").alias("key"),
        F.to_json(F.struct(*LIVE_PAYLOAD_COLUMNS)).cast("binary").alias("value"),
    )
)

# COMMAND ----------

(
    txn_stream_df
    .writeStream
    .queryName("transactions_to_kafka")
    .format("kafka")
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers_plaintext)
    .option("topic", TXN_TOPIC)
    .option("checkpointLocation", checkpoint_path)
    .trigger(processingTime="1 seconds")
    .start()
)
