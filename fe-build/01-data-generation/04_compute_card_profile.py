# Databricks notebook source
# MAGIC %md
# MAGIC # 4. Compute `card_profile` from the history
# MAGIC
# MAGIC One row per card (~17.5M) describing that card's normal behavior. Only non-fraud transactions
# MAGIC count toward the baseline, the way a bank excludes confirmed fraud.
# MAGIC
# MAGIC The `profile_window` widget picks the window. Run it twice:
# MAGIC
# MAGIC | Run | `profile_window` | Window | Used for |
# MAGIC |---|---|---|---|
# MAGIC | Before model training | `training` | Oct 1 – Oct 21 (20 days) | Training and validation features |
# MAGIC | After model training | `serving` | Oct 1 – Oct 31 (30 days) | Published to Lakebase for live scoring |
# MAGIC
# MAGIC Same table name and columns both times, so the model (which remembers the table name) reads the
# MAGIC 20-day version during training and the endpoint reads the 30-day version live.
# MAGIC
# MAGIC Window-length-dependent columns (`txn_count`, `cnt_<category>`) are raw counts; use
# MAGIC `avg_daily_txn` or `> 0` checks in the model so 20-day and 30-day profiles mean the same thing.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

dbutils.widgets.dropdown("profile_window", "training", ["training", "serving"])
profile_window = dbutils.widgets.get("profile_window")

profile_start = HISTORY_START
profile_end = TRAINING_PROFILE_END if profile_window == "training" else HISTORY_END
profile_days = (profile_end - profile_start).days
print(f"Building card_profile for '{profile_window}': {profile_start} to {profile_end} ({profile_days} days)")

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {fq('card_profile')}")

# COMMAND ----------

as_of = F.lit(profile_end).cast("timestamp")
customers = spark.table(fq("customers"))
cards = spark.table(fq("cards"))

txns = (
    spark.table(fq("transactions_history"))
    .filter(~F.col("is_fraud"))
    .filter((F.col("transaction_ts") >= F.lit(profile_start)) & (F.col("transaction_ts") < F.lit(profile_end)))
    .join(customers.select("customer_id", "home_lat", "home_lon"), "customer_id")
    .withColumn("hour_of_day", F.hour("transaction_ts"))
    .withColumn("is_online", (F.col("channel") == "ONLINE").cast("double"))
    .withColumn("is_night", (F.col("hour_of_day") < 6).cast("double"))
    .withColumn("is_high_risk", F.col("merchant_category").isin(HIGH_RISK_CATEGORIES).cast("double"))
    .withColumn(
        "instore_distance_km",
        F.when(F.col("channel") == "IN_STORE", haversine_km("latitude", "longitude", "home_lat", "home_lon")),
    )
)

card_agg = txns.groupBy("card_id").agg(
    F.count("*").alias("txn_count"),
    F.round(F.avg("amount_usd"), 2).alias("avg_amount"),
    F.round(F.stddev("amount_usd"), 2).alias("stddev_amount"),
    F.round(F.percentile_approx("amount_usd", 0.95), 2).alias("p95_amount"),
    F.max("amount_usd").alias("max_amount"),
    F.round(F.avg("is_online"), 4).alias("online_ratio"),
    F.round(F.avg("is_night"), 4).alias("night_ratio"),
    F.round(F.avg("hour_of_day"), 2).alias("avg_hour_of_day"),
    F.round(F.stddev("hour_of_day"), 2).alias("stddev_hour_of_day"),
    F.round(F.avg("is_high_risk"), 4).alias("high_risk_ratio"),
    F.approx_count_distinct("merchant_id").alias("distinct_merchants"),
    F.approx_count_distinct("merchant_category").alias("distinct_categories"),
    F.round(F.avg("instore_distance_km"), 1).alias("avg_instore_distance_km"),
    F.round(F.max("instore_distance_km"), 1).alias("max_instore_distance_km"),
    F.round(F.avg((F.col("instore_distance_km") > 100).cast("double")), 4).alias("away_from_home_ratio"),
    *[F.sum((F.col("merchant_category") == c).cast("int")).alias(f"cnt_{c}") for c in CATEGORY_NAMES],
)

customer_agg = txns.groupBy("customer_id").agg(
    F.count("*").alias("cust_txn_count"),
    F.round(F.avg("amount_usd"), 2).alias("cust_avg_amount"),
    F.round(F.percentile_approx("amount_usd", 0.95), 2).alias("cust_p95_amount"),
)

count_columns = ["txn_count", "distinct_merchants", "distinct_categories", "cust_txn_count"] + [f"cnt_{c}" for c in CATEGORY_NAMES]

card_profile = (
    cards.select("card_id", "customer_id", "card_type", "card_network", "credit_limit", "is_primary", "issue_date")
    .join(customers.select("customer_id", "customer_segment", "home_city_id", "home_lat", "home_lon", "signup_date"),
          "customer_id")
    .join(card_agg, "card_id", "left")
    .join(customer_agg, "customer_id", "left")
    .fillna(0, subset=count_columns)
    .withColumn("avg_daily_txn", F.round(F.col("txn_count") / profile_days, 3))
    .withColumn("cust_avg_daily_txn", F.round(F.col("cust_txn_count") / profile_days, 3))
    .withColumn("card_age_days", F.datediff(F.to_date(as_of), F.col("issue_date")))
    .withColumn("customer_tenure_days", F.datediff(F.to_date(as_of), F.col("signup_date")))
    .withColumn("profile_window", F.lit(profile_window))
    .withColumn("profile_start_ts", F.lit(profile_start).cast("timestamp"))
    .withColumn("profile_as_of_ts", as_of)
    .withColumn("profile_days", F.lit(profile_days))
    .drop("issue_date", "signup_date")
)

card_profile.write.saveAsTable(fq("card_profile"))
add_primary_key("card_profile", "card_id")
spark.sql(f"ALTER TABLE {fq('card_profile')} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

display(spark.sql(f"""
  select
    (select count(*) from {fq('cards')})        as cards,
    (select count(*) from {fq('card_profile')}) as card_profiles,
    (select count(*) from {fq('card_profile')} where txn_count = 0) as cards_without_history,
    (select first(profile_window) from {fq('card_profile')})   as profile_window,
    (select first(profile_as_of_ts) from {fq('card_profile')}) as profile_as_of_ts
"""))

# COMMAND ----------

display(spark.sql(f"""
  select customer_segment,
         round(avg(avg_daily_txn), 3)        as avg_daily_txns_per_card,
         round(avg(avg_amount), 2)           as avg_amount,
         round(avg(online_ratio), 3)         as online_ratio,
         round(avg(away_from_home_ratio), 3) as away_ratio
  from {fq('card_profile')}
  group by 1 order by 1
"""))

# COMMAND ----------

display(spark.table(fq("card_profile")).limit(20))
