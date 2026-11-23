# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Register the comparison feature functions
# MAGIC
# MAGIC Registers small Python functions in Unity Catalog under `murali_talluri.fe_bar`. Each one compares
# MAGIC a transaction with its card's profile. Because they live in Unity Catalog (not just this
# MAGIC notebook), the model remembers them and the serving endpoint runs the exact same code live.
# MAGIC
# MAGIC | Function | Output | Example |
# MAGIC |---|---|---|
# MAGIC | `fn_amount_ratio` | amount ÷ card's average (customer's average if the card has no history) | $1,900 ÷ $48 = 39.6 |
# MAGIC | `fn_amount_to_max_ratio` | amount ÷ card's largest normal purchase | $1,900 ÷ $900 = 2.1 |
# MAGIC | `fn_distance_from_home_km` | distance from the transaction to the card holder's home | Miami → Chicago ≈ 1,900 km |
# MAGIC | `fn_is_new_category` | 1 if the card has never been used in this category, else 0 | jewelry count 0 → 1 |
# MAGIC | `fn_hour_of_day` | hour of the transaction (0–23) | 3 AM → 3 |
# MAGIC | `fn_hour_diff_from_usual` | hours between this transaction and the card's usual hour | 3 vs 14.2 → 11.2 |
# MAGIC | `fn_is_online` | 1 if online, else 0 | ONLINE → 1 |

# COMMAND ----------

# MAGIC %run ./00_ml_config

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_amount_ratio')}(amount_usd DOUBLE, avg_amount DOUBLE, cust_avg_amount DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON
COMMENT 'Transaction amount divided by the card average, falling back to the customer average for cards without history.'
AS $$
def missing(v):
    return v is None or v != v
base = cust_avg_amount if missing(avg_amount) or avg_amount == 0 else avg_amount
if missing(amount_usd) or missing(base) or base == 0:
    return None
return amount_usd / base
$$
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_amount_to_max_ratio')}(amount_usd DOUBLE, max_amount DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON
COMMENT 'Transaction amount divided by the card largest normal purchase. Above 1 means bigger than ever.'
AS $$
def missing(v):
    return v is None or v != v
if missing(amount_usd) or missing(max_amount) or max_amount == 0:
    return None
return amount_usd / max_amount
$$
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_distance_from_home_km')}(latitude DOUBLE, longitude DOUBLE, home_lat DOUBLE, home_lon DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON
COMMENT 'Great-circle distance in km between the transaction location and the card holder home.'
AS $$
import math
def missing(v):
    return v is None or v != v
if any(missing(v) for v in (latitude, longitude, home_lat, home_lon)):
    return None
p1, p2 = math.radians(latitude), math.radians(home_lat)
dp, dl = p2 - p1, math.radians(home_lon - longitude)
a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
return 2 * 6371.0 * math.asin(math.sqrt(a))
$$
""")

# COMMAND ----------

count_params = ", ".join(f"cnt_{c} BIGINT" for c in CATEGORY_NAMES)
count_map = ", ".join(f"'{c}': cnt_{c}" for c in CATEGORY_NAMES)

spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_is_new_category')}(merchant_category STRING, {count_params})
RETURNS INT
LANGUAGE PYTHON
COMMENT '1 if the card has no normal purchases in this merchant category in its profile (or no profile at all), else 0.'
AS $$
def missing(v):
    return v is None or v != v
counts = {{{count_map}}}
if merchant_category not in counts:
    return None
count = counts[merchant_category]
return 1 if missing(count) or count == 0 else 0
$$
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_hour_of_day')}(transaction_ts TIMESTAMP)
RETURNS INT
LANGUAGE PYTHON
COMMENT 'Hour of day (0-23) of the transaction.'
AS $$
if transaction_ts is None:
    return None
return transaction_ts.hour
$$
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_hour_diff_from_usual')}(transaction_ts TIMESTAMP, avg_hour_of_day DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON
COMMENT 'Hours between the transaction time and the card usual hour, wrapping around midnight (0-12).'
AS $$
def missing(v):
    return v is None or v != v
if transaction_ts is None or missing(avg_hour_of_day):
    return None
hour = transaction_ts.hour + transaction_ts.minute / 60.0
diff = abs(hour - avg_hour_of_day)
return min(diff, 24 - diff)
$$
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_is_online')}(channel STRING)
RETURNS INT
LANGUAGE PYTHON
COMMENT '1 if the transaction is online, else 0.'
AS $$
if channel is None:
    return None
return 1 if channel == 'ONLINE' else 0
$$
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check the functions on real transactions
# MAGIC
# MAGIC Five fraud and five normal transactions from the training window, with every function applied.
# MAGIC Fraud rows should tend to show larger ratios, bigger distances, new categories and odd hours.

# COMMAND ----------

count_args = ", ".join(f"p.cnt_{c}" for c in CATEGORY_NAMES)

display(spark.sql(f"""
  with sample as (
    (select * from {fq('transactions_history')}
     where transaction_ts >= '{TRAIN_START}' and transaction_ts < '{TRAIN_END}' and is_fraud limit 5)
    union all
    (select * from {fq('transactions_history')}
     where transaction_ts >= '{TRAIN_START}' and transaction_ts < '{TRAIN_END}' and not is_fraud limit 5)
  )
  select
    t.is_fraud, t.fraud_type, t.card_id, t.merchant_category, t.channel, t.amount_usd, t.transaction_ts,
    p.avg_amount, p.max_amount, p.avg_hour_of_day,
    round({fq('fn_amount_ratio')}(t.amount_usd, p.avg_amount, p.cust_avg_amount), 2)              as amount_ratio,
    round({fq('fn_amount_to_max_ratio')}(t.amount_usd, p.max_amount), 2)                           as amount_to_max_ratio,
    round({fq('fn_distance_from_home_km')}(t.latitude, t.longitude, p.home_lat, p.home_lon), 1)    as distance_from_home_km,
    {fq('fn_is_new_category')}(t.merchant_category, {count_args})                                  as is_new_category,
    {fq('fn_hour_of_day')}(t.transaction_ts)                                                       as hour_of_day,
    round({fq('fn_hour_diff_from_usual')}(t.transaction_ts, p.avg_hour_of_day), 1)                 as hour_diff_from_usual,
    {fq('fn_is_online')}(t.channel)                                                                as is_online
  from sample t
  join {fq('card_profile')} p using (card_id)
  order by t.is_fraud desc
"""))
