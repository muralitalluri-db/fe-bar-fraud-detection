# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Generate the 30-day labeled transaction history
# MAGIC
# MAGIC Writes `transactions_history`: roughly 500M+ normal transactions plus fraud episodes on about 3%
# MAGIC of cards, between `HISTORY_START` and `HISTORY_END`. This is the model's training data.
# MAGIC
# MAGIC Run `01_generate_dimensions` first. This is the heaviest step; use a multi-node cluster or
# MAGIC serverless compute.

# COMMAND ----------

# MAGIC %run ./_transaction_generator

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {fq('transactions_history')}")

# COMMAND ----------

card_ctx = load_card_context()

legit = generate_legit_transactions(card_ctx, HISTORY_START, HISTORY_END, salt="hist")
fraud = generate_fraud_transactions(
    card_ctx, HISTORY_START, HISTORY_END, salt="hist_fraud",
    compromised_rate=HISTORY_COMPROMISED_CARD_RATE,
)

legit.unionByName(fraud).write.saveAsTable(fq("transactions_history"))
add_primary_key("transactions_history", "transaction_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

display(spark.sql(f"""
  select
    count(*)                                   as total_txns,
    sum(cast(is_fraud as int))                 as fraud_txns,
    round(avg(cast(is_fraud as double)) * 100, 3) as fraud_pct,
    count(distinct case when is_fraud then card_id end) as compromised_cards,
    min(transaction_ts)                        as first_ts,
    max(transaction_ts)                        as last_ts
  from {fq('transactions_history')}
"""))

# COMMAND ----------

display(spark.sql(f"""
  select fraud_type, count(*) as txns, count(distinct card_id) as cards,
         round(avg(amount_usd), 2) as avg_amount,
         round(avg(case when channel = 'ONLINE' then 1.0 else 0.0 end), 2) as online_ratio
  from {fq('transactions_history')}
  where is_fraud
  group by 1 order by 2 desc
"""))

# COMMAND ----------

# Daily volume: weekends and paydays (1st, 15th) should be a little higher.
display(spark.sql(f"""
  select transaction_date, date_format(transaction_date, 'E') as dow, count(*) as txns
  from {fq('transactions_history')}
  group by 1, 2 order by 1
"""))

# COMMAND ----------

# Hour-of-day profile: normal activity peaks in the day; fraud is spread across all hours.
display(spark.sql(f"""
  select hour(transaction_ts) as hour, is_fraud, count(*) as txns
  from {fq('transactions_history')}
  group by 1, 2 order by 1, 2
"""))

# COMMAND ----------

# Normal transactions by category and channel.
display(spark.sql(f"""
  select merchant_category, channel, count(*) as txns, round(percentile_approx(amount_usd, 0.5), 2) as median_amount
  from {fq('transactions_history')}
  where not is_fraud
  group by 1, 2 order by 3 desc
"""))
