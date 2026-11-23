# Databricks notebook source
# MAGIC %md
# MAGIC # 3. Generate the 1-hour live window
# MAGIC
# MAGIC Writes `transactions_live`: about 3.6M transactions (1k per second) for the hour right after
# MAGIC the training history ends, using the same customers, cards, merchants, behavior and fraud
# MAGIC patterns. This table gets streamed into Kafka for the demo.
# MAGIC
# MAGIC `is_fraud` and `fraud_type` are kept as hidden ground truth so the app can show live model accuracy.
# MAGIC They must not be sent to the model.

# COMMAND ----------

# MAGIC %run ./_transaction_generator

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {fq('transactions_live')}")

# COMMAND ----------

card_ctx = load_card_context()
num_cards = spark.table(fq("cards")).count()

# Choose how many cards to compromise so fraud is about LIVE_FRAUD_TXN_RATE of live transactions.
target_fraud_txns = LIVE_TARGET_TXNS * LIVE_FRAUD_TXN_RATE
live_compromised_rate = target_fraud_txns / expected_fraud_txns_per_episode() / num_cards
print(f"Compromising ~{live_compromised_rate * num_cards:,.0f} of {num_cards:,} cards during the live window")

legit = generate_legit_transactions(card_ctx, LIVE_START, LIVE_END, salt="live", target_count=LIVE_TARGET_TXNS)
fraud = generate_fraud_transactions(
    card_ctx, LIVE_START, LIVE_END, salt="live_fraud",
    compromised_rate=live_compromised_rate, lead_seconds=10 * 60,
)

legit.unionByName(fraud).write.saveAsTable(fq("transactions_live"))
add_primary_key("transactions_live", "transaction_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

display(spark.sql(f"""
  select
    count(*)                                      as total_txns,
    sum(cast(is_fraud as int))                    as fraud_txns,
    round(avg(cast(is_fraud as double)) * 100, 3) as fraud_pct,
    count(distinct case when is_fraud then card_id end) as compromised_cards,
    min(transaction_ts)                           as first_ts,
    max(transaction_ts)                           as last_ts
  from {fq('transactions_live')}
"""))

# COMMAND ----------

# Per-second throughput should center around 1,000.
display(spark.sql(f"""
  with per_sec as (
    select date_trunc('SECOND', transaction_ts) as sec, count(*) as c
    from {fq('transactions_live')} group by 1
  )
  select min(c) as min_per_sec, round(avg(c)) as avg_per_sec,
         percentile(c, 0.5) as p50_per_sec, percentile(c, 0.99) as p99_per_sec, max(c) as max_per_sec,
         count(*) as seconds
  from per_sec
"""))

# COMMAND ----------

display(spark.sql(f"""
  select fraud_type, count(*) as txns, count(distinct card_id) as cards
  from {fq('transactions_live')} where is_fraud
  group by 1 order by 2 desc
"""))

# COMMAND ----------

# Every live card and merchant must exist in the training dimensions.
display(spark.sql(f"""
  select
    sum(case when c.card_id is null then 1 else 0 end)     as unknown_cards,
    sum(case when m.merchant_id is null then 1 else 0 end) as unknown_merchants
  from {fq('transactions_live')} t
  left join {fq('cards')} c using (card_id)
  left join {fq('merchants')} m using (merchant_id)
"""))
