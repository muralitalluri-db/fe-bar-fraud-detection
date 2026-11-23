# Databricks notebook source
# MAGIC %md
# MAGIC # Transaction generator (shared)
# MAGIC
# MAGIC Used by `02_generate_transactions_history` and `03_generate_transactions_live` through
# MAGIC `%run ./_transaction_generator`, so both windows use identical behavior and fraud logic.
# MAGIC
# MAGIC - **Normal transactions** follow each customer's habits: home city, a few regular merchants per
# MAGIC   category, active hours, spend level, online preference, and occasional legitimate travel.
# MAGIC - **Fraud** is injected as short episodes on compromised cards (see `FRAUD_TYPES` in `00_config`).

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

TXN_COLUMNS = [
    "transaction_id", "card_id", "customer_id", "merchant_id", "merchant_category", "mcc", "channel",
    "amount_usd", "currency", "transaction_ts", "transaction_date", "latitude", "longitude",
    "is_fraud", "fraud_type",
]

_CATEGORY = {c["name"]: c for c in CATEGORIES}


def _category_attr(attr: str, category: Column) -> Column:
    return map_lookup({name: float(c[attr]) if attr != "mcc" else c[attr] for name, c in _CATEGORY.items()}, category)


def with_customer_behavior(df: DataFrame) -> DataFrame:
    """Hidden behavior parameters, derived deterministically from customer_id and segment."""
    seg = F.col("customer_segment")

    def seg_value(attr):
        return map_lookup({s: float(c[attr]) for s, c in SEGMENTS.items()}, seg)

    return (
        df
        .withColumn("daily_txn_rate", seg_value("daily_rate") * F.exp(F.lit(0.3) * std_normal("rate", "customer_id")))
        .withColumn("spend_multiplier", seg_value("spend_scale") * F.exp(F.lit(0.35) * std_normal("spend", "customer_id")))
        .withColumn("active_hour_center",
                    F.least(F.lit(20.0), F.greatest(F.lit(9.0), F.lit(14.5) + F.lit(2.0) * std_normal("hour_center", "customer_id"))))
        .withColumn("active_hour_spread", F.lit(2.5) + F.lit(1.5) * uniform("hour_spread", "customer_id"))
        .withColumn("online_factor", seg_value("online_factor") * (F.lit(0.8) + F.lit(0.4) * uniform("online", "customer_id")))
        .withColumn("uses_high_risk", uniform("high_risk", "customer_id") < seg_value("high_risk_prob"))
        .withColumn("travel_prob", seg_value("travel_prob"))
    )


def load_card_context() -> DataFrame:
    """One row per card with the owning customer's profile and hidden behavior."""
    customers = spark.table(fq("customers")).select(
        "customer_id", "customer_segment", "home_city_id", "home_lat", "home_lon", "num_cards"
    )
    cards = spark.table(fq("cards")).select("card_id", "customer_id", "card_seq")

    weight_rows = []
    for n in range(MIN_CARDS_PER_CUSTOMER, MAX_CARDS_PER_CUSTOMER + 1):
        total = sum(CARD_USAGE_WEIGHTS[:n])
        weight_rows += [(n, seq + 1, CARD_USAGE_WEIGHTS[seq] * n / total) for seq in range(n)]
    usage = spark.createDataFrame(weight_rows, "num_cards int, card_seq int, usage_weight double")

    return (
        cards.join(customers, "customer_id")
        .join(F.broadcast(usage), ["num_cards", "card_seq"])
        .transform(with_customer_behavior)
    )


def _merchant_buckets():
    w = Window.partitionBy("bucket_city_id", "merchant_category").orderBy("merchant_id")
    merchants = (
        spark.table(fq("merchants"))
        .withColumn("bucket_city_id", F.coalesce(F.col("city_id"), F.lit(0)))
        .withColumn("bucket_idx", F.row_number().over(w) - 1)
        .select(
            "merchant_id", "merchant_category", "bucket_city_id", "bucket_idx",
            F.col("latitude").alias("merchant_lat"), F.col("longitude").alias("merchant_lon"),
        )
    )
    sizes = merchants.groupBy("bucket_city_id", "merchant_category").agg(F.count("*").alias("bucket_size"))
    return merchants, sizes


def resolve_merchants(df: DataFrame) -> DataFrame:
    """Map (bucket_city_id, merchant_category, merchant_hash) to a concrete merchant.

    bucket_city_id is the merchant's city for in-store purchases, or 0 for online.
    """
    merchants, sizes = _merchant_buckets()
    return (
        df.join(F.broadcast(sizes), ["bucket_city_id", "merchant_category"])
        .withColumn("bucket_idx", F.pmod(F.col("merchant_hash"), F.col("bucket_size")).cast("int"))
        .join(F.broadcast(merchants), ["bucket_city_id", "merchant_category", "bucket_idx"])
        .drop("bucket_size", "bucket_idx", "merchant_hash")
    )


def _city_centers(prefix: str) -> DataFrame:
    return cities_df().select(
        F.col("city_id").alias(f"{prefix}_city_id"),
        F.col("latitude").alias(f"{prefix}_city_lat"),
        F.col("longitude").alias(f"{prefix}_city_lon"),
    )


def _finalize(df: DataFrame, salt: str, is_fraud: bool) -> DataFrame:
    return (
        df
        .withColumn("transaction_id", F.concat(
            F.lit("txn_"),
            F.substring(F.sha2(F.concat_ws("|", F.lit(salt), F.col("card_id"), F.col("txn_seq").cast("string")), 256), 1, 24),
        ))
        .withColumn("transaction_ts", F.timestamp_seconds(F.col("_ts_epoch")))
        .withColumn("transaction_date", F.to_date("transaction_ts"))
        .withColumn("mcc", _category_attr("mcc", F.col("merchant_category")))
        .withColumn("currency", F.lit("USD"))
        .withColumn("latitude", F.round("latitude", 6))
        .withColumn("longitude", F.round("longitude", 6))
        .withColumn("is_fraud", F.lit(is_fraud))
        .select(*TXN_COLUMNS)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normal transactions

# COMMAND ----------

def _day_weight(d) -> float:
    w = 1.0
    if d.weekday() >= 5:
        w *= 1.15
    elif d.weekday() == 4:
        w *= 1.10
    if d.day in (1, 15):
        w *= 1.25
    return w


def generate_legit_transactions(card_ctx: DataFrame, start: datetime, end: datetime, salt: str,
                                target_count: int = None) -> DataFrame:
    """Normal transactions in [start, end).

    target_count=None: calendar timing. Each card's volume follows its own daily rate, with
    weekday/weekend/payday effects and the customer's active hours (used for the 30-day history).

    target_count=N: uniform timing within the window, with card volumes scaled so the expected total
    is N (used for the live window to hit a steady transactions-per-second rate).
    """
    s, e = epoch(start), epoch(end)
    card_rate = F.col("daily_txn_rate") * F.col("usage_weight")

    if target_count is None:
        first_day = start.date()
        last_day = (end - timedelta(microseconds=1)).date()
        days = [first_day + timedelta(days=i) for i in range((last_day - first_day).days + 1)]
        lam = card_rate * F.lit(float(len(days)))
    else:
        total_rate = card_ctx.agg(F.sum(card_rate)).first()[0]
        lam = card_rate * F.lit(target_count / total_rate)

    keys = ("card_id", "txn_seq")

    def U(name):
        return uniform(f"{salt}_{name}", *keys)

    def Z(name):
        return std_normal(f"{salt}_{name}", *keys)

    df = (
        card_ctx
        .withColumn("_n", poisson(lam, f"{salt}_count", "card_id"))
        .filter(F.col("_n") > 0)
        .withColumn("txn_seq", F.explode(F.sequence(F.lit(0), F.col("_n") - 1)))
    )

    if target_count is None:
        day_idx = pick_weighted(U("day"), [(i, _day_weight(d)) for i, d in enumerate(days)])
        day_start = F.lit(float(epoch(datetime.combine(first_day, datetime.min.time())))) + day_idx * F.lit(86400.0)
        hour = F.when(U("night_owl") < 0.05, U("hour_any") * 24.0).otherwise(
            F.pmod(F.col("active_hour_center") + F.col("active_hour_spread") * Z("hour"), F.lit(24.0))
        )
        df = df.withColumn("_ts_epoch", day_start + hour * F.lit(3600.0))
    else:
        df = df.withColumn("_ts_epoch", F.lit(float(s)) + U("ts") * F.lit(float(e - s)))

    df = df.filter((F.col("_ts_epoch") >= s) & (F.col("_ts_epoch") < e))

    # Legitimate travel: a customer is either home or in one other city for a whole 3-day block.
    # Keyed on customer and calendar block only, so history and live agree.
    trip_block = F.floor(F.floor(F.col("_ts_epoch") / 86400) / 3)
    travel_city = pick_weighted(uniform("trip_city", "customer_id", trip_block), CITY_WEIGHTS)
    df = (
        df
        .withColumn("_traveling", uniform("trip", "customer_id", trip_block) < F.col("travel_prob"))
        .withColumn("_travel_city", F.when(travel_city == F.col("home_city_id"),
                                           F.pmod(F.col("home_city_id"), F.lit(NUM_CITIES)) + 1).otherwise(travel_city))
        .withColumn("current_city_id", F.when(F.col("_traveling"), F.col("_travel_city")).otherwise(F.col("home_city_id")))
    )

    all_categories = [(c["name"], c["txn_weight"]) for c in CATEGORIES]
    safe_categories = [(c["name"], c["txn_weight"]) for c in CATEGORIES if c["name"] not in HIGH_RISK_CATEGORIES]
    df = df.withColumn(
        "merchant_category",
        F.when(F.col("uses_high_risk"), pick_weighted(U("category"), all_categories))
        .otherwise(pick_weighted(U("category"), safe_categories)),
    )

    online_share = _category_attr("online_share", F.col("merchant_category"))
    p_online = (
        F.when(online_share >= 1.0, F.lit(1.0))
        .when(online_share <= 0.0, F.lit(0.0))
        .otherwise(F.least(F.lit(0.95), online_share * F.col("online_factor")))
    )
    df = (
        df
        .withColumn("_online", U("channel") < p_online)
        .withColumn("channel", F.when(F.col("_online"), "ONLINE").otherwise("IN_STORE"))
        .withColumn("bucket_city_id", F.when(F.col("_online"), F.lit(0)).otherwise(F.col("current_city_id")))
    )

    # Habitual merchants: 3 regular merchants per (customer, category, city); 15% of the time a random one.
    explore = U("explore") < 0.15
    slot = F.when(explore, F.floor(U("explore_slot") * 1e9)).otherwise(F.floor(U("slot") * 3))
    df = df.withColumn(
        "merchant_hash",
        F.xxhash64(F.col("customer_id"), F.col("merchant_category"), F.col("bucket_city_id"), slot, explore),
    )

    median = _category_attr("median_amount", F.col("merchant_category"))
    sigma = _category_attr("amount_sigma", F.col("merchant_category"))
    amount = F.exp(F.log(median) + sigma * Z("amount")) * F.col("spend_multiplier")
    # Occasional legitimate big purchase.
    amount = F.when(U("big") < 0.004, amount * (F.lit(3.0) + F.lit(5.0) * U("big_mult"))).otherwise(amount)
    df = df.withColumn("amount_usd", F.round(F.least(F.greatest(amount, F.lit(0.5)), F.lit(25000.0)), 2))

    df = (
        resolve_merchants(df)
        .join(F.broadcast(_city_centers("current")), "current_city_id")
        .withColumn(
            "latitude",
            F.when(~F.col("_online"), F.col("merchant_lat"))
            .when(F.col("_traveling"), F.col("current_city_lat") + (U("geo_lat") - 0.5) * 0.3)
            .otherwise(F.col("home_lat") + (U("geo_lat") - 0.5) * 0.1),
        )
        .withColumn(
            "longitude",
            F.when(~F.col("_online"), F.col("merchant_lon"))
            .when(F.col("_traveling"), F.col("current_city_lon") + (U("geo_lon") - 0.5) * 0.3)
            .otherwise(F.col("home_lon") + (U("geo_lon") - 0.5) * 0.1),
        )
        .withColumn("fraud_type", F.lit(None).cast("string"))
    )
    return _finalize(df, salt, is_fraud=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fraud episodes

# COMMAND ----------

def expected_fraud_txns_per_episode() -> float:
    total = sum(cfg["weight"] for cfg in FRAUD_TYPES.values())
    return sum(cfg["weight"] * (cfg["n_min"] + cfg["n_max"]) / 2 for cfg in FRAUD_TYPES.values()) / total


def generate_fraud_transactions(card_ctx: DataFrame, start: datetime, end: datetime, salt: str,
                                compromised_rate: float, lead_seconds: int = 0) -> DataFrame:
    """Fraud transactions in [start, end).

    A compromised_rate share of cards get one episode. Episodes start uniformly in
    [start - lead_seconds, end); transactions outside the window are dropped.
    """
    s, e = epoch(start), epoch(end)
    ftype = F.col("fraud_type")

    def type_attr(attr, default=0.0):
        return map_lookup({t: float(cfg.get(attr, default)) for t, cfg in FRAUD_TYPES.items()}, ftype)

    far = far_cities_df()
    far_counts = far.select("home_city_id", "far_count").distinct()

    episodes = (
        card_ctx
        .filter(uniform(f"{salt}_compromised", "card_id") < compromised_rate)
        .withColumn("fraud_type", pick_weighted(uniform(f"{salt}_type", "card_id"),
                                                [(t, cfg["weight"]) for t, cfg in FRAUD_TYPES.items()]))
        .withColumn("_n", (type_attr("n_min")
                           + F.floor(uniform(f"{salt}_n", "card_id") * (type_attr("n_max") - type_attr("n_min") + 1))).cast("int"))
        .withColumn("_gap", type_attr("gap_min_sec")
                    + uniform(f"{salt}_gap", "card_id") * (type_attr("gap_max_sec") - type_attr("gap_min_sec")))
        .withColumn("_episode_start", F.lit(float(s - lead_seconds))
                    + uniform(f"{salt}_start", "card_id") * F.lit(float(e - s + lead_seconds)))
        .join(F.broadcast(far_counts), "home_city_id")
        .withColumn("far_rank", F.pmod(F.xxhash64("card_id", F.lit(f"{salt}_far")), F.col("far_count")).cast("int"))
        .join(F.broadcast(far.select("home_city_id", "far_rank", "far_city_id")), ["home_city_id", "far_rank"])
        .join(F.broadcast(_city_centers("far")), "far_city_id")
    )

    keys = ("card_id", "txn_seq")

    def U(name):
        return uniform(f"{salt}_{name}", *keys)

    df = (
        episodes
        .withColumn("txn_seq", F.explode(F.sequence(F.lit(0), F.col("_n") - 1)))
        .withColumn("_ts_epoch", F.col("_episode_start") + F.col("txn_seq") * F.col("_gap") + U("jitter") * 0.5 * F.col("_gap"))
        .filter((F.col("_ts_epoch") >= s) & (F.col("_ts_epoch") < e))
        .withColumn("_probe", (ftype == "CARD_TESTING") & (F.col("txn_seq") < F.col("_n") - 1))
    )

    category = F.when(F.col("_probe"), pick_weighted(U("category"), CARD_TESTING_PROBE["categories"]))
    for t, cfg in FRAUD_TYPES.items():
        category = category.when(ftype == t, pick_weighted(U("category"), cfg["categories"]))
    df = df.withColumn("merchant_category", category)

    online_share = _category_attr("online_share", F.col("merchant_category"))
    channel_rule = map_lookup({t: cfg["channel"] for t, cfg in FRAUD_TYPES.items()}, ftype)
    df = (
        df
        .withColumn("_online",
                    F.when(channel_rule == "ONLINE", F.lit(True))
                    .when(channel_rule == "IN_STORE", F.lit(False))
                    .otherwise(U("channel") < online_share))
        .withColumn("channel", F.when(F.col("_online"), "ONLINE").otherwise("IN_STORE"))
        .withColumn("bucket_city_id",
                    F.when(F.col("_online"), F.lit(0))
                    .when(ftype == "IMPOSSIBLE_TRAVEL", F.col("far_city_id"))
                    .otherwise(F.col("home_city_id")))
        .withColumn("merchant_hash", F.xxhash64(F.col("card_id"), F.col("txn_seq"), F.lit(f"{salt}_merchant")))
    )

    typical = _category_attr("median_amount", F.col("merchant_category")) * F.col("spend_multiplier")
    amount = (
        F.when(F.col("_probe"), log_uniform(CARD_TESTING_PROBE["amount_min"], CARD_TESTING_PROBE["amount_max"], U("amount")))
        .when(ftype == "AMOUNT_ANOMALY", typical * log_uniform(
            FRAUD_TYPES["AMOUNT_ANOMALY"]["amount_multiplier_min"],
            FRAUD_TYPES["AMOUNT_ANOMALY"]["amount_multiplier_max"], U("amount")))
        .otherwise(log_uniform(type_attr("amount_min", 1.0), type_attr("amount_max", 1.0), U("amount")))
    )
    df = df.withColumn("amount_usd", F.round(F.least(amount, F.lit(25000.0)), 2))

    # Online fraud from card testing / account takeover appears to come from the fraudster's far-away location.
    remote_online = F.col("_online") & ftype.isin("CARD_TESTING", "ACCOUNT_TAKEOVER")
    df = (
        resolve_merchants(df)
        .withColumn(
            "latitude",
            F.when(~F.col("_online"), F.col("merchant_lat"))
            .when(remote_online, F.col("far_city_lat") + (U("geo_lat") - 0.5) * 0.3)
            .otherwise(F.col("home_lat") + (U("geo_lat") - 0.5) * 0.1),
        )
        .withColumn(
            "longitude",
            F.when(~F.col("_online"), F.col("merchant_lon"))
            .when(remote_online, F.col("far_city_lon") + (U("geo_lon") - 0.5) * 0.3)
            .otherwise(F.col("home_lon") + (U("geo_lon") - 0.5) * 0.1),
        )
    )
    return _finalize(df, salt, is_fraud=True)
