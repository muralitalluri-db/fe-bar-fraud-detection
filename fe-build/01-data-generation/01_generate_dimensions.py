# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Generate customers, merchants and cards
# MAGIC
# MAGIC | Table | Rows |
# MAGIC |---|---|
# MAGIC | `customers` | 5M |
# MAGIC | `merchants` | ~100k |
# MAGIC | `cards` | ~17.5M (2 to 5 per customer) |
# MAGIC
# MAGIC These tables are shared by the 30-day training history and the 1-hour live window.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
for t in ["cards", "merchants", "customers"]:
    spark.sql(f"DROP TABLE IF EXISTS {fq(t)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customers

# COMMAND ----------

cities = cities_df()
home_city = cities.select(
    F.col("city_id").alias("home_city_id"),
    F.col("city").alias("home_city"),
    F.col("state").alias("home_state"),
    F.col("latitude").alias("_city_lat"),
    F.col("longitude").alias("_city_lon"),
)

signup_span_days = (datetime(2026, 9, 1) - datetime(2015, 1, 1)).days

customers = (
    spark.range(NUM_CUSTOMERS)
    .withColumnRenamed("id", "customer_num")
    .withColumn("customer_id", F.format_string("cust_%08d", "customer_num"))
    .withColumn("home_city_id", pick_weighted(uniform("home_city", "customer_id"), CITY_WEIGHTS))
    .join(F.broadcast(home_city), "home_city_id")
    .withColumn("home_lat", F.round(F.col("_city_lat") + (uniform("home_lat", "customer_id") - 0.5) * 0.5, 6))
    .withColumn("home_lon", F.round(F.col("_city_lon") + (uniform("home_lon", "customer_id") - 0.5) * 0.6, 6))
    .withColumn(
        "customer_segment",
        pick_weighted(uniform("segment", "customer_id"), [(s, c["weight"]) for s, c in SEGMENTS.items()]),
    )
    .withColumn("first_name", pick_from(FIRST_NAMES, uniform("first_name", "customer_id")))
    .withColumn("last_name", pick_from(LAST_NAMES, uniform("last_name", "customer_id")))
    .withColumn(
        "email",
        F.lower(F.format_string("%s.%s%d@example.com", "first_name", "last_name", F.col("customer_num") % 1000)),
    )
    .withColumn(
        "phone_number",
        F.format_string(
            "+1-%03d-555-%04d",
            (F.lit(200) + F.floor(uniform("phone_area", "customer_id") * 799)).cast("int"),
            F.floor(uniform("phone_line", "customer_id") * 10000).cast("int"),
        ),
    )
    .withColumn(
        "date_of_birth",
        F.date_sub(
            F.lit("2026-10-01").cast("date"),
            (F.lit(18 * 365) + uniform("dob", "customer_id") * (67 * 365)).cast("int"),
        ),
    )
    .withColumn(
        "signup_date",
        F.date_add(F.lit("2015-01-01").cast("date"), (uniform("signup", "customer_id") * signup_span_days).cast("int")),
    )
    .withColumn(
        "num_cards",
        (F.lit(MIN_CARDS_PER_CUSTOMER)
         + F.floor(uniform("num_cards", "customer_id") * (MAX_CARDS_PER_CUSTOMER - MIN_CARDS_PER_CUSTOMER + 1))).cast("int"),
    )
    .select(
        "customer_id", "first_name", "last_name", "email", "phone_number", "date_of_birth",
        "home_city_id", "home_city", "home_state", "home_lat", "home_lon",
        "customer_segment", "num_cards", "signup_date",
    )
)

customers.write.saveAsTable(fq("customers"))
add_primary_key("customers", "customer_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merchants
# MAGIC
# MAGIC Merchant counts are split per category, then per city by population weight. Every city has at
# MAGIC least one in-store merchant for each in-store category, and every category with online sales has
# MAGIC at least one online merchant (`city_id` is null for online merchants).

# COMMAND ----------

total_share = sum(c["merchant_share"] for c in CATEGORIES)
total_city_weight = sum(w for _, w in CITY_WEIGHTS)

bucket_rows = []
for cat in CATEGORIES:
    n_cat = NUM_MERCHANTS * cat["merchant_share"] / total_share
    if cat["online_share"] > 0:
        bucket_rows.append((cat["name"], None, max(1, round(n_cat * cat["online_share"]))))
    if cat["online_share"] < 1:
        n_store = n_cat * (1 - cat["online_share"])
        for city_id, w in CITY_WEIGHTS:
            bucket_rows.append((cat["name"], city_id, max(1, round(n_store * w / total_city_weight))))

offset, buckets = 0, []
for name, city_id, count in bucket_rows:
    buckets.append((name, city_id, count, offset))
    offset += count

merchant_buckets = spark.createDataFrame(
    buckets, "merchant_category string, city_id int, bucket_count int, id_offset int"
)

merchant_names = F.create_map(*itertools.chain.from_iterable(
    (F.lit(c["name"]), F.array(*[F.lit(n) for n in c["merchant_names"]])) for c in CATEGORIES
))

merchants = (
    merchant_buckets
    .withColumn("local_idx", F.explode(F.sequence(F.lit(0), F.col("bucket_count") - 1)))
    .withColumn("merchant_num", F.col("id_offset") + F.col("local_idx"))
    .withColumn("merchant_id", F.format_string("mrc_%06d", "merchant_num"))
    .join(F.broadcast(cities.withColumnRenamed("latitude", "_city_lat").withColumnRenamed("longitude", "_city_lon")),
          "city_id", "left")
    .withColumn("channel", F.when(F.col("city_id").isNull(), "ONLINE").otherwise("IN_STORE"))
    .withColumn(
        "merchant_name",
        F.format_string(
            "%s #%d",
            F.element_at(
                F.element_at(merchant_names, F.col("merchant_category")),
                (F.floor(uniform("merchant_name", "merchant_id") * 5) + 1).cast("int"),
            ),
            F.col("merchant_num") % 10000,
        ),
    )
    .withColumn("mcc", map_lookup({c["name"]: c["mcc"] for c in CATEGORIES}, F.col("merchant_category")))
    .withColumn("risk_level", map_lookup({c["name"]: c["risk_level"] for c in CATEGORIES}, F.col("merchant_category")))
    .withColumn("latitude", F.round(F.col("_city_lat") + (uniform("m_lat", "merchant_id") - 0.5) * 0.4, 6))
    .withColumn("longitude", F.round(F.col("_city_lon") + (uniform("m_lon", "merchant_id") - 0.5) * 0.5, 6))
    .select(
        "merchant_id", "merchant_name", "merchant_category", "mcc", "risk_level", "channel",
        "city_id", "city", "state", "latitude", "longitude",
    )
)

merchants.write.saveAsTable(fq("merchants"))
add_primary_key("merchants", "merchant_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cards
# MAGIC
# MAGIC `card_id` is `card_<customer number>_<card_seq>`. `card_seq = 1` is the primary card.
# MAGIC `card_number` is a fake PAN (not Luhn-valid) kept for the Unity Catalog masking demo.

# COMMAND ----------

issue_floor = F.lit("2019-01-01").cast("date")
issue_ceiling = F.lit("2026-09-01").cast("date")
network = F.col("card_network")
pan_body = F.lpad(F.pmod(F.xxhash64("card_id", F.lit("pan")), F.lit(10**15)).cast("string"), 15, "0")

cards = (
    spark.table(fq("customers"))
    .select("customer_id", "num_cards", "signup_date")
    .withColumn("card_seq", F.explode(F.sequence(F.lit(1), F.col("num_cards"))))
    .withColumn("card_id", F.format_string("card_%s_%d", F.substring("customer_id", 6, 8), "card_seq"))
    .withColumn("card_network", pick_weighted(uniform("network", "card_id"), CARD_NETWORKS))
    .withColumn("card_type", F.when(uniform("card_type", "card_id") < 0.6, "CREDIT").otherwise("DEBIT"))
    .withColumn(
        "card_number",
        F.when(network == "VISA", F.concat(F.lit("4"), F.substring(pan_body, 1, 15)))
        .when(network == "MASTERCARD", F.concat(
            F.lit("5"),
            (F.lit(1) + F.floor(uniform("mc_prefix", "card_id") * 5)).cast("int").cast("string"),
            F.substring(pan_body, 1, 14),
        ))
        .when(network == "AMEX", F.concat(F.lit("37"), F.substring(pan_body, 1, 13)))
        .otherwise(F.concat(F.lit("6011"), F.substring(pan_body, 1, 12))),
    )
    .withColumn("card_last4", F.substring("card_number", -4, 4))
    .withColumn("_issue_from", F.greatest(F.col("signup_date"), issue_floor))
    .withColumn(
        "issue_date",
        F.date_add(
            F.col("_issue_from"),
            (uniform("issue", "card_id") * F.greatest(F.datediff(issue_ceiling, F.col("_issue_from")), F.lit(0))).cast("int"),
        ),
    )
    .withColumn("expiry_date", F.last_day(F.add_months(F.col("issue_date"), 48)))
    .withColumn(
        "credit_limit",
        F.when(
            F.col("card_type") == "CREDIT",
            F.round((F.lit(1500.0) + uniform("limit", "card_id") * 23500.0) / 100) * 100,
        ),
    )
    .withColumn("is_primary", F.col("card_seq") == 1)
    .withColumn("status", F.lit("ACTIVE"))
    .select(
        "card_id", "customer_id", "card_seq", "is_primary", "card_network", "card_type",
        "card_number", "card_last4", "issue_date", "expiry_date", "credit_limit", "status",
    )
)

cards.write.saveAsTable(fq("cards"))
add_primary_key("cards", "card_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

display(spark.sql(f"""
  select 'customers' as table_name, count(*) as row_count from {fq('customers')}
  union all select 'merchants', count(*) from {fq('merchants')}
  union all select 'cards', count(*) from {fq('cards')}
"""))

# COMMAND ----------

display(spark.sql(f"""
  select customer_segment, count(*) as customers, round(avg(num_cards), 2) as avg_cards
  from {fq('customers')} group by 1 order by 2 desc
"""))

# COMMAND ----------

display(spark.sql(f"""
  select merchant_category, channel, count(*) as merchants
  from {fq('merchants')} group by 1, 2 order by 1, 2
"""))
