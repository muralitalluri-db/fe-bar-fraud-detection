# Databricks notebook source
# MAGIC %md
# MAGIC # 5. Build the live stream table (one file per second)
# MAGIC
# MAGIC Copies `transactions_live` into `transactions_live_stream` one second at a time, writing each
# MAGIC second as a single file (~1k rows). The Kafka ingest stream then reads it with
# MAGIC `maxFilesPerTrigger=1`, so each micro-batch publishes one second of transactions.
# MAGIC
# MAGIC Only the payload columns are copied; the fraud labels stay in `transactions_live`.
# MAGIC This makes ~3,600 small appends, so it takes a while. It only needs to run once per data set.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {fq(LIVE_STREAM_TABLE)}")

# COMMAND ----------

# MAGIC %sql
# MAGIC set spark.databricks.delta.autoCompact.enabled = false;
# MAGIC set spark.databricks.delta.optimizeWrite.enabled = false;

# COMMAND ----------

stream_table = fq(LIVE_STREAM_TABLE)
src = spark.table(fq("transactions_live")).select(*LIVE_PAYLOAD_COLUMNS)

src.limit(0).write.saveAsTable(stream_table)
spark.sql(f"""
  ALTER TABLE {stream_table} SET TBLPROPERTIES (
    'delta.autoOptimize.autoCompact' = 'false',
    'delta.autoOptimize.optimizeWrite' = 'false'
  )
""")
spark.sql(f"ALTER TABLE {stream_table} DISABLE PREDICTIVE OPTIMIZATION")

# COMMAND ----------

bounds = src.agg(F.min("transaction_ts").alias("min_ts"), F.max("transaction_ts").alias("max_ts")).first()
cursor = bounds.min_ts.replace(microsecond=0)
end_time = bounds.max_ts
total_intervals = int((end_time.replace(microsecond=0) - cursor).total_seconds()) + 1

idx = 0
while cursor <= end_time:
    idx += 1
    interval_end = cursor + timedelta(seconds=1) - timedelta(microseconds=1)
    if idx % 60 == 1:
        print(f"Interval {idx}/{total_intervals}: {cursor} — {interval_end}")

    (
        src.filter(F.col("transaction_ts").between(F.lit(cursor), F.lit(interval_end)))
        .repartition(1)
        .write.format("delta").mode("append").saveAsTable(stream_table)
    )
    cursor += timedelta(seconds=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

display(spark.sql(f"""
  select
    (select count(*) from {fq('transactions_live')}) as live_rows,
    (select count(*) from {stream_table})            as stream_rows
"""))

# COMMAND ----------

# numFiles should be about 3,600 (one per second).
display(spark.sql(f"DESCRIBE DETAIL {stream_table}").select("numFiles", "sizeInBytes"))
