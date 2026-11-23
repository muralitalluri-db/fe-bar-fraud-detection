# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Business impact: fraud losses, baseline to target
# MAGIC
# MAGIC Measures the live hour in the KPIs the business owners are judged on, using the scored transactions in
# MAGIC Lakebase and the hidden fraud labels in Delta.
# MAGIC
# MAGIC | Owner | KPI |
# MAGIC |---|---|
# MAGIC | Chief Risk Officer (funds the program, owns the fraud-loss line) | Fraud losses in USD |
# MAGIC | Head of Card Fraud Strategy (runs fraud operations) | Good customers wrongly blocked, good spend declined |
# MAGIC
# MAGIC Three scenarios over the same hour of traffic:
# MAGIC
# MAGIC | Scenario | What gets declined |
# MAGIC |---|---|
# MAGIC | **Baseline: batch detection** | Nothing during the hour; fraud is found after the fact, so every fraud dollar is lost |
# MAGIC | **Built: real-time blocking** | Every transaction on a card *after* its first flagged transaction (the trigger has blocked the card) |
# MAGIC | **Phase 2: score at authorization** | The flagged transaction itself as well, if the same model is called inside the authorization decision |
# MAGIC
# MAGIC The data is synthetic and this is a one-hour replay; the numbers are not annualized.
# MAGIC
# MAGIC Run on **serverless** compute: Lakebase catalogs in Unity Catalog can only be queried from serverless.

# COMMAND ----------

# MAGIC %run ../05-genie-agents/00_genie_config

# COMMAND ----------

impact = spark.sql(f"""
    WITH scored AS (
        SELECT l.card_id, l.transaction_ts, l.amount_usd, l.is_fraud, s.is_flagged
        FROM {LIVE_UC_TABLE} l
        JOIN {SCORED_TRANSACTIONS_UC_TABLE} s USING (transaction_id)
    ),
    first_flag AS (
        SELECT card_id, min(transaction_ts) AS first_flag_ts
        FROM scored
        WHERE is_flagged
        GROUP BY card_id
    ),
    labeled AS (
        SELECT s.*,
               f.first_flag_ts,
               coalesce(s.transaction_ts >  f.first_flag_ts, false) AS declined_after_block,
               coalesce(s.transaction_ts >= f.first_flag_ts, false) AS declined_at_authorization
        FROM scored s
        LEFT JOIN first_flag f USING (card_id)
    )
    SELECT
        sum(amount_usd)                                                                  AS purchase_volume_usd,
        sum(CASE WHEN is_fraud THEN amount_usd ELSE 0 END)                               AS fraud_usd,
        sum(CASE WHEN is_fraud AND declined_after_block THEN amount_usd ELSE 0 END)      AS fraud_stopped_built,
        sum(CASE WHEN is_fraud AND declined_at_authorization THEN amount_usd ELSE 0 END) AS fraud_stopped_phase2,
        sum(CASE WHEN NOT is_fraud AND declined_after_block THEN amount_usd ELSE 0 END)      AS good_declined_built,
        sum(CASE WHEN NOT is_fraud AND declined_at_authorization THEN amount_usd ELSE 0 END) AS good_declined_phase2,
        count(DISTINCT CASE WHEN is_fraud THEN card_id END)                                  AS cards_with_fraud,
        count(DISTINCT CASE WHEN first_flag_ts IS NOT NULL THEN card_id END)                 AS cards_blocked,
        count(DISTINCT CASE WHEN is_fraud AND first_flag_ts IS NOT NULL THEN card_id END)    AS fraud_cards_blocked
    FROM labeled
""").first()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline to target

# COMMAND ----------

import pandas as pd

volume = impact["purchase_volume_usd"]
fraud = impact["fraud_usd"]
lost_built = fraud - impact["fraud_stopped_built"]
lost_phase2 = fraud - impact["fraud_stopped_phase2"]
good_cards_blocked = impact["cards_blocked"] - impact["fraud_cards_blocked"]

kpis = pd.DataFrame(
    [
        ("Fraud losses (USD)", f"${fraud:,.0f}", f"${lost_built:,.0f}", f"${lost_phase2:,.0f}"),
        ("Fraud losses vs. baseline", "-", f"-{1 - lost_built / fraud:.0%}", f"-{1 - lost_phase2 / fraud:.0%}"),
        ("Cards with fraud that were blocked",
         f"0 of {impact['cards_with_fraud']:,}",
         f"{impact['fraud_cards_blocked']:,} of {impact['cards_with_fraud']:,}",
         f"{impact['fraud_cards_blocked']:,} of {impact['cards_with_fraud']:,}"),
        ("Good customers' cards blocked", "0", f"{good_cards_blocked:,}", f"{good_cards_blocked:,}"),
        ("Good spend declined (USD)", "$0",
         f"${impact['good_declined_built']:,.0f} ({impact['good_declined_built'] / volume:.2%} of volume)",
         f"${impact['good_declined_phase2']:,.0f} ({impact['good_declined_phase2'] / volume:.2%} of volume)"),
    ],
    columns=["KPI", "Baseline: batch detection", "Built: real-time blocking", "Phase 2: score at authorization"],
)

print(f"Purchase volume in the hour: ${volume:,.0f}")
display(kpis)
