# Databricks notebook source
# MAGIC %md
# MAGIC # ML config
# MAGIC
# MAGIC Shared by every notebook in this folder through `%run ./00_ml_config`. Pulls in the data
# MAGIC generation config so table names and time windows have a single source of truth.
# MAGIC
# MAGIC | Period | Window | Used for |
# MAGIC |---|---|---|
# MAGIC | Profile | Oct 1 – Oct 21 | `card_profile` built with `profile_window = training` |
# MAGIC | Train | Oct 21 – Oct 28 | Model learns from these transactions |
# MAGIC | Validate | Oct 28 – Oct 31 | Model is tested on these transactions |

# COMMAND ----------

# MAGIC %run ../01-data-generation/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Date splits and sampling

# COMMAND ----------

TRAIN_START = TRAINING_PROFILE_END                      # Oct 21 12:00
TRAIN_END = HISTORY_START + timedelta(days=27)          # Oct 28 12:00
VALIDATION_START = TRAIN_END
VALIDATION_END = HISTORY_END                            # Oct 31 12:00

# Training keeps every fraud row and this share of normal rows, so fraud isn't drowned out.
TRAIN_NORMAL_SAMPLE_RATE = 0.20
# Validation samples fraud and normal rows at the same rate, so it keeps the real fraud mix.
VALIDATION_SAMPLE_RATE = 0.20

LABEL = "is_fraud"
MODEL_NAME = fq("fraud_model")
# The validated version gets this alias; serving points at it. The blocking threshold is stored
# as a tag on that version.
CHAMPION_ALIAS = "champion"
THRESHOLD_TAG = "fraud_threshold"

# Lakebase Autoscaling project that backs the online feature store.
LAKEBASE_PROJECT = "fe-bar-lakebase"
LAKEBASE_CAPACITY = "CU_2"
# Online copy of card_profile. Its catalog name is also the Postgres database name in Lakebase.
ONLINE_PROFILE_TABLE = fq("card_profile_online")

# Model Serving endpoint. Requests and responses are logged to <INFERENCE_TABLE_PREFIX>_payload.
ENDPOINT_NAME = "fe-bar-fraud-scoring"
INFERENCE_TABLE_PREFIX = "fraud_scoring"
# Raw transaction fields a scoring request must contain; the endpoint looks up and computes the rest.
SCORING_REQUEST_COLUMNS = [
    "card_id", "amount_usd", "merchant_category", "channel", "latitude", "longitude", "transaction_ts",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features
# MAGIC
# MAGIC Three sources, all assembled by Feature Engineering so the serving endpoint repeats the same steps:
# MAGIC
# MAGIC 1. **Transaction fields** sent with each transaction.
# MAGIC 2. **Profile fields** looked up from `card_profile` by `card_id` (Lakebase online store when serving).
# MAGIC 3. **Comparison features** computed by Unity Catalog Python functions from 1 and 2.

# COMMAND ----------

# Raw transaction fields the model uses directly.
TXN_MODEL_FEATURES = ["amount_usd", "merchant_category", "channel"]

# Profile fields the model uses directly. Raw counts (txn_count, cnt_<category>, distinct_*) are left
# out because they grow with the profile window (20 days in training, 30 days live).
PROFILE_MODEL_FEATURES = [
    "card_type", "card_network", "credit_limit", "is_primary", "customer_segment",
    "avg_amount", "stddev_amount", "p95_amount", "max_amount",
    "online_ratio", "night_ratio", "avg_hour_of_day", "stddev_hour_of_day", "high_risk_ratio",
    "avg_instore_distance_km", "max_instore_distance_km", "away_from_home_ratio",
    "avg_daily_txn", "cust_avg_amount", "cust_p95_amount", "cust_avg_daily_txn",
    "card_age_days", "customer_tenure_days",
]

# Profile fields looked up only as inputs to the comparison functions, not given to the model.
PROFILE_HELPER_COLUMNS = ["home_lat", "home_lon"] + [f"cnt_{c}" for c in CATEGORY_NAMES]

PROFILE_LOOKUP_COLUMNS = PROFILE_MODEL_FEATURES + PROFILE_HELPER_COLUMNS

# Unity Catalog Python functions: (function name, output column, {function parameter: input column}).
FEATURE_FUNCTIONS = [
    ("fn_amount_ratio", "amount_ratio",
     {"amount_usd": "amount_usd", "avg_amount": "avg_amount", "cust_avg_amount": "cust_avg_amount"}),
    ("fn_amount_to_max_ratio", "amount_to_max_ratio",
     {"amount_usd": "amount_usd", "max_amount": "max_amount"}),
    ("fn_distance_from_home_km", "distance_from_home_km",
     {"latitude": "latitude", "longitude": "longitude", "home_lat": "home_lat", "home_lon": "home_lon"}),
    ("fn_is_new_category", "is_new_category",
     {"merchant_category": "merchant_category", **{f"cnt_{c}": f"cnt_{c}" for c in CATEGORY_NAMES}}),
    ("fn_hour_of_day", "hour_of_day",
     {"transaction_ts": "transaction_ts"}),
    ("fn_hour_diff_from_usual", "hour_diff_from_usual",
     {"transaction_ts": "transaction_ts", "avg_hour_of_day": "avg_hour_of_day"}),
    ("fn_is_online", "is_online",
     {"channel": "channel"}),
]
FUNCTION_FEATURES = [output for _, output, _ in FEATURE_FUNCTIONS]

MODEL_FEATURES = TXN_MODEL_FEATURES + PROFILE_MODEL_FEATURES + FUNCTION_FEATURES


def feature_lookups():
    """The lookup recipe: profile by card_id, then the comparison functions."""
    from databricks.feature_engineering import FeatureFunction, FeatureLookup

    return [
        FeatureLookup(table_name=fq("card_profile"), lookup_key="card_id", feature_names=PROFILE_LOOKUP_COLUMNS),
        *[
            FeatureFunction(udf_name=fq(name), output_name=output, input_bindings=bindings)
            for name, output, bindings in FEATURE_FUNCTIONS
        ],
    ]
