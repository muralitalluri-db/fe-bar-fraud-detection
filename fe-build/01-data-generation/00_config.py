# Databricks notebook source
# MAGIC %md
# MAGIC # Config and shared helpers
# MAGIC
# MAGIC Shared by every notebook in this folder through `%run ./00_config`.
# MAGIC
# MAGIC All randomness is hash-based (`xxhash64` of stable keys plus a salt), so re-running any
# MAGIC notebook produces the same customers, cards, merchants and transactions.

# COMMAND ----------

import calendar
import itertools
import math
from datetime import datetime, timedelta

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scale and time windows

# COMMAND ----------

CATALOG = "murali_talluri"
SCHEMA = "fe_bar"


def fq(table: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{table}"


NUM_CUSTOMERS = 5_000_000
MIN_CARDS_PER_CUSTOMER = 2
MAX_CARDS_PER_CUSTOMER = 5
NUM_MERCHANTS = 100_000

HISTORY_START = datetime(2026, 10, 1, 12, 0, 0)
HISTORY_DAYS = 30
HISTORY_END = HISTORY_START + timedelta(days=HISTORY_DAYS)

# card_profile is built twice: from the first 20 days for model training (so training transactions
# after this point are only compared with earlier behavior), then from all 30 days for live serving.
TRAINING_PROFILE_DAYS = 20
TRAINING_PROFILE_END = HISTORY_START + timedelta(days=TRAINING_PROFILE_DAYS)

# The live window starts exactly where the history ends.
LIVE_START = HISTORY_END
LIVE_MINUTES = 60
LIVE_END = LIVE_START + timedelta(minutes=LIVE_MINUTES)
LIVE_TARGET_TPS = 1_000
LIVE_TARGET_TXNS = LIVE_TARGET_TPS * LIVE_MINUTES * 60

# Share of cards that get one fraud episode during the 30-day history.
HISTORY_COMPROMISED_CARD_RATE = 0.03
# Target share of live transactions that are fraudulent.
LIVE_FRAUD_TXN_RATE = 0.005

# Live transactions written one file per second, then streamed to Kafka.
LIVE_STREAM_TABLE = "transactions_live_stream"
# Fields that leave the lakehouse with each live transaction. is_fraud and fraud_type stay behind
# in transactions_live as ground truth.
LIVE_PAYLOAD_COLUMNS = [
    "transaction_id", "card_id", "customer_id", "merchant_id", "merchant_category", "mcc", "channel",
    "amount_usd", "currency", "transaction_ts", "latitude", "longitude",
]


def epoch(dt: datetime) -> int:
    return calendar.timegm(dt.timetuple())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reference data

# COMMAND ----------

# (city, state, latitude, longitude, relative population weight). city_id = position + 1; 0 means online.
CITIES = [
    ("New York", "NY", 40.7128, -74.0060, 19.5),
    ("Los Angeles", "CA", 34.0522, -118.2437, 13.0),
    ("Chicago", "IL", 41.8781, -87.6298, 9.4),
    ("Dallas", "TX", 32.7767, -96.7970, 7.6),
    ("Houston", "TX", 29.7604, -95.3698, 7.1),
    ("Washington", "DC", 38.9072, -77.0369, 6.3),
    ("Philadelphia", "PA", 39.9526, -75.1652, 6.2),
    ("Miami", "FL", 25.7617, -80.1918, 6.1),
    ("Atlanta", "GA", 33.7490, -84.3880, 6.1),
    ("Boston", "MA", 42.3601, -71.0589, 4.9),
    ("Phoenix", "AZ", 33.4484, -112.0740, 4.9),
    ("San Francisco", "CA", 37.7749, -122.4194, 4.7),
    ("Detroit", "MI", 42.3314, -83.0458, 4.3),
    ("Seattle", "WA", 47.6062, -122.3321, 4.0),
    ("Minneapolis", "MN", 44.9778, -93.2650, 3.7),
    ("San Diego", "CA", 32.7157, -117.1611, 3.3),
    ("Tampa", "FL", 27.9506, -82.4572, 3.2),
    ("Denver", "CO", 39.7392, -104.9903, 3.0),
    ("Baltimore", "MD", 39.2904, -76.6122, 2.8),
    ("St. Louis", "MO", 38.6270, -90.1994, 2.8),
    ("Orlando", "FL", 28.5383, -81.3792, 2.7),
    ("Charlotte", "NC", 35.2271, -80.8431, 2.7),
    ("San Antonio", "TX", 29.4241, -98.4936, 2.6),
    ("Portland", "OR", 45.5152, -122.6784, 2.5),
    ("Sacramento", "CA", 38.5816, -121.4944, 2.4),
    ("Pittsburgh", "PA", 40.4406, -79.9959, 2.4),
    ("Austin", "TX", 30.2672, -97.7431, 2.4),
    ("Las Vegas", "NV", 36.1699, -115.1398, 2.3),
    ("Cincinnati", "OH", 39.1031, -84.5120, 2.3),
    ("Kansas City", "MO", 39.0997, -94.5786, 2.2),
    ("Columbus", "OH", 39.9612, -82.9988, 2.1),
    ("Indianapolis", "IN", 39.7684, -86.1581, 2.1),
    ("Cleveland", "OH", 41.4993, -81.6944, 2.1),
    ("Nashville", "TN", 36.1627, -86.7816, 2.0),
    ("San Jose", "CA", 37.3382, -121.8863, 2.0),
    ("Jacksonville", "FL", 30.3322, -81.6557, 1.7),
    ("Raleigh", "NC", 35.7796, -78.6382, 1.5),
    ("Salt Lake City", "UT", 40.7608, -111.8910, 1.3),
    ("New Orleans", "LA", 29.9511, -90.0715, 1.3),
    ("Omaha", "NE", 41.2565, -95.9345, 1.0),
    ("Honolulu", "HI", 21.3069, -157.8583, 1.0),
    ("Albuquerque", "NM", 35.0844, -106.6504, 0.9),
    ("Boise", "ID", 43.6150, -116.2023, 0.8),
    ("Anchorage", "AK", 61.2181, -149.9003, 0.4),
]
CITY_WEIGHTS = [(i + 1, c[4]) for i, c in enumerate(CITIES)]
NUM_CITIES = len(CITIES)

# Fraud "impossible travel" picks a city at least this far from the cardholder's home.
FAR_CITY_MIN_KM = 1200

# merchant_share: share of merchants; txn_weight: share of everyday transactions;
# median_amount / amount_sigma: lognormal amount; online_share: share of merchants/txns that are online.
CATEGORIES = [
    dict(name="grocery",               mcc="5411", merchant_share=0.14,  txn_weight=0.200, median_amount=45,  amount_sigma=0.60, online_share=0.08, risk_level="LOW",
         merchant_names=["FreshMart", "Green Basket", "Daily Grocer", "Harvest Foods", "Corner Market"]),
    dict(name="gas",                   mcc="5541", merchant_share=0.08,  txn_weight=0.120, median_amount=40,  amount_sigma=0.35, online_share=0.00, risk_level="LOW",
         merchant_names=["QuickFuel", "Roadway Gas", "Sunoco Express", "PetroStop", "FillUp"]),
    dict(name="restaurants",           mcc="5812", merchant_share=0.16,  txn_weight=0.150, median_amount=32,  amount_sigma=0.60, online_share=0.10, risk_level="LOW",
         merchant_names=["Bistro Lane", "Urban Grill", "Taco Town", "Pasta House", "Golden Wok"]),
    dict(name="coffee_shops",          mcc="5814", merchant_share=0.06,  txn_weight=0.090, median_amount=7,   amount_sigma=0.35, online_share=0.05, risk_level="LOW",
         merchant_names=["Bean There", "Morning Brew", "Roast Co", "Daily Grind", "Cup & Saucer"]),
    dict(name="pharmacy",              mcc="5912", merchant_share=0.04,  txn_weight=0.050, median_amount=25,  amount_sigma=0.60, online_share=0.10, risk_level="LOW",
         merchant_names=["CareRx", "HealthPlus Pharmacy", "MedCorner", "WellPoint Drug", "CityRx"]),
    dict(name="retail_apparel",        mcc="5651", merchant_share=0.10,  txn_weight=0.070, median_amount=65,  amount_sigma=0.70, online_share=0.40, risk_level="LOW",
         merchant_names=["Threadline", "Urban Outfit", "StyleHub", "Denim & Co", "The Closet"]),
    dict(name="department_stores",     mcc="5311", merchant_share=0.03,  txn_weight=0.040, median_amount=80,  amount_sigma=0.70, online_share=0.35, risk_level="LOW",
         merchant_names=["Grand Dept", "Macroy's", "Northgate", "Hudson & Pike", "ValueMart"]),
    dict(name="home_improvement",      mcc="5200", merchant_share=0.04,  txn_weight=0.030, median_amount=90,  amount_sigma=0.80, online_share=0.15, risk_level="LOW",
         merchant_names=["BuildRight", "HomeBase Supply", "ToolShed", "FixIt Depot", "Lumber Yard"]),
    dict(name="electronics",           mcc="5732", merchant_share=0.04,  txn_weight=0.030, median_amount=220, amount_sigma=0.90, online_share=0.55, risk_level="MEDIUM",
         merchant_names=["VoltTech", "Circuit City", "GadgetZone", "ByteBuy", "PixelPlus"]),
    dict(name="online_marketplace",    mcc="5399", merchant_share=0.03,  txn_weight=0.080, median_amount=45,  amount_sigma=0.90, online_share=1.00, risk_level="LOW",
         merchant_names=["ShopSphere", "MegaMarket Online", "CartNow", "BuyAll", "Bazaarly"]),
    dict(name="digital_subscriptions", mcc="4899", merchant_share=0.02,  txn_weight=0.040, median_amount=14,  amount_sigma=0.40, online_share=1.00, risk_level="LOW",
         merchant_names=["StreamFlix", "TuneBox", "CloudVault", "NewsPlus", "GameStream"]),
    dict(name="travel_airlines",       mcc="4511", merchant_share=0.01,  txn_weight=0.010, median_amount=380, amount_sigma=0.60, online_share=0.90, risk_level="MEDIUM",
         merchant_names=["SkyWay Air", "BlueJet", "Horizon Airlines", "AeroLink", "Coastal Air"]),
    dict(name="hotels_lodging",        mcc="7011", merchant_share=0.03,  txn_weight=0.010, median_amount=240, amount_sigma=0.60, online_share=0.50, risk_level="MEDIUM",
         merchant_names=["Grandview Hotel", "StayInn", "Harbor Suites", "Parkside Lodge", "Metro Hotel"]),
    dict(name="entertainment",         mcc="7999", merchant_share=0.05,  txn_weight=0.030, median_amount=35,  amount_sigma=0.60, online_share=0.30, risk_level="LOW",
         merchant_names=["CineMax", "FunZone", "Bowl-O-Rama", "LiveStage", "Arcade City"]),
    dict(name="jewelry",               mcc="5944", merchant_share=0.01,  txn_weight=0.004, median_amount=300, amount_sigma=0.90, online_share=0.30, risk_level="HIGH",
         merchant_names=["Diamond Row", "Gold & Gem", "Luxe Jewelers", "Sparkle Co", "Carat House"]),
    dict(name="gift_cards",            mcc="5947", merchant_share=0.01,  txn_weight=0.004, median_amount=80,  amount_sigma=0.60, online_share=0.60, risk_level="HIGH",
         merchant_names=["GiftHub", "CardCorner", "PrepaidPlus", "GiftWorld", "eGift Express"]),
    dict(name="crypto_exchange",       mcc="6051", merchant_share=0.005, txn_weight=0.002, median_amount=400, amount_sigma=1.00, online_share=1.00, risk_level="HIGH",
         merchant_names=["CoinVault", "BitTrade", "CryptoNow", "BlockEx", "HashMarket"]),
]
CATEGORY_NAMES = [c["name"] for c in CATEGORIES]
HIGH_RISK_CATEGORIES = [c["name"] for c in CATEGORIES if c["risk_level"] == "HIGH"]

# Customer segments drive "normal" behavior. daily_rate is transactions per card per day (before
# primary/secondary card weighting); spend_scale multiplies amounts; high_risk_prob is the chance a
# customer ever uses high-risk categories; travel_prob is the chance of being away on a given 3-day block.
SEGMENTS = {
    "LIGHT":          dict(weight=0.40, daily_rate=0.5, spend_scale=0.7, online_factor=0.8, high_risk_prob=0.08, travel_prob=0.03),
    "AVERAGE":        dict(weight=0.35, daily_rate=1.0, spend_scale=1.0, online_factor=1.0, high_risk_prob=0.10, travel_prob=0.04),
    "HEAVY":          dict(weight=0.15, daily_rate=2.0, spend_scale=1.4, online_factor=1.0, high_risk_prob=0.15, travel_prob=0.06),
    "TRAVELER":       dict(weight=0.05, daily_rate=1.4, spend_scale=1.3, online_factor=1.1, high_risk_prob=0.12, travel_prob=0.45),
    "ONLINE_SHOPPER": dict(weight=0.05, daily_rate=1.3, spend_scale=1.0, online_factor=2.2, high_risk_prob=0.25, travel_prob=0.04),
}

# Primary card gets the most use; weights are normalized per customer so the total rate is unchanged.
CARD_USAGE_WEIGHTS = [1.0, 0.6, 0.35, 0.2, 0.12]

CARD_NETWORKS = [("VISA", 0.50), ("MASTERCARD", 0.30), ("AMEX", 0.12), ("DISCOVER", 0.08)]

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth",
    "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen",
    "Daniel", "Nancy", "Matthew", "Lisa", "Anthony", "Betty", "Mark", "Margaret", "Steven", "Sandra",
    "Priya", "Wei", "Carlos", "Fatima", "Hiroshi", "Aisha", "Diego", "Mei", "Raj", "Sofia",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Patel", "Chen", "Nguyen", "Kim", "Singh", "Tanaka", "Khan", "Rossi", "Murphy", "Cohen",
]

# Fraud episodes. Each compromised card gets one episode of one type.
# gap_*_sec: time between consecutive fraud transactions in the episode.
# channel: ONLINE, IN_STORE, or CATEGORY (use the category's online share).
FRAUD_TYPES = {
    "CARD_TESTING": dict(
        weight=0.25, n_min=4, n_max=9, gap_min_sec=30, gap_max_sec=120, channel="ONLINE",
        categories=[("electronics", 0.5), ("gift_cards", 0.3), ("online_marketplace", 0.2)],
        amount_min=300, amount_max=2500,
    ),
    "ACCOUNT_TAKEOVER": dict(
        weight=0.25, n_min=2, n_max=5, gap_min_sec=300, gap_max_sec=900, channel="ONLINE",
        categories=[("electronics", 0.40), ("gift_cards", 0.25), ("crypto_exchange", 0.15), ("jewelry", 0.10), ("travel_airlines", 0.10)],
        amount_min=400, amount_max=3500,
    ),
    "IMPOSSIBLE_TRAVEL": dict(
        weight=0.20, n_min=1, n_max=3, gap_min_sec=600, gap_max_sec=1800, channel="IN_STORE",
        categories=[("electronics", 0.30), ("jewelry", 0.20), ("retail_apparel", 0.20), ("department_stores", 0.15), ("restaurants", 0.15)],
        amount_min=150, amount_max=2000,
    ),
    "HIGH_RISK_MERCHANT": dict(
        weight=0.15, n_min=1, n_max=4, gap_min_sec=300, gap_max_sec=1200, channel="CATEGORY",
        categories=[("gift_cards", 0.5), ("crypto_exchange", 0.5)],
        amount_min=150, amount_max=1500,
    ),
    "AMOUNT_ANOMALY": dict(
        weight=0.15, n_min=1, n_max=2, gap_min_sec=300, gap_max_sec=1800, channel="CATEGORY",
        categories=[("electronics", 0.35), ("jewelry", 0.20), ("retail_apparel", 0.15), ("home_improvement", 0.15), ("department_stores", 0.15)],
        # Amount is the customer's typical spend for the category times this multiplier range.
        amount_multiplier_min=8, amount_multiplier_max=25,
    ),
}
# Card testing starts with tiny "probe" charges; only the last transaction of the episode is large.
CARD_TESTING_PROBE = dict(
    categories=[("digital_subscriptions", 0.5), ("online_marketplace", 0.5)],
    amount_min=0.5, amount_max=5.0,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helpers

# COMMAND ----------

_HASH_MOD = 2_147_483_647


def _col(x) -> Column:
    return x if isinstance(x, Column) else F.col(x)


def uniform(salt: str, *keys) -> Column:
    """Deterministic uniform [0, 1) from the given key columns and salt."""
    h = F.xxhash64(*[_col(k) for k in keys], F.lit(salt))
    return F.pmod(h, F.lit(_HASH_MOD)).cast("double") / F.lit(float(_HASH_MOD))


def std_normal(salt: str, *keys) -> Column:
    """Deterministic standard normal via Box-Muller."""
    u1 = F.greatest(uniform(f"{salt}_a", *keys), F.lit(1e-12))
    u2 = uniform(f"{salt}_b", *keys)
    return F.sqrt(F.lit(-2.0) * F.log(u1)) * F.cos(F.lit(2 * math.pi) * u2)


def log_uniform(lo, hi, u: Column) -> Column:
    lo, hi = _as_col(lo), _as_col(hi)
    return F.exp(F.log(lo) + u * (F.log(hi) - F.log(lo)))


def _as_col(x) -> Column:
    return x if isinstance(x, Column) else F.lit(float(x))


def pick_weighted(u: Column, items) -> Column:
    """Pick a value from [(value, weight), ...] using a uniform column."""
    total = float(sum(w for _, w in items))
    expr, cum = None, 0.0
    for value, w in items[:-1]:
        cum += w / total
        expr = F.when(u < cum, F.lit(value)) if expr is None else expr.when(u < cum, F.lit(value))
    last = F.lit(items[-1][0])
    return last if expr is None else expr.otherwise(last)


def pick_from(values, u: Column) -> Column:
    arr = F.array(*[F.lit(v) for v in values])
    return F.element_at(arr, (F.floor(u * len(values)) + 1).cast("int"))


def map_lookup(mapping: dict, key: Column) -> Column:
    m = F.create_map(*itertools.chain.from_iterable((F.lit(k), F.lit(v)) for k, v in mapping.items()))
    return F.element_at(m, key)


def poisson(lam: Column, salt: str, *keys) -> Column:
    """Poisson sample: exact inverse CDF for small lambda, normal approximation otherwise."""
    u = uniform(f"{salt}_u", *keys)
    small, cdf, term = None, None, None
    for k in range(0, 9):
        term = F.exp(-lam) if k == 0 else term * lam / F.lit(float(k))
        cdf = term if k == 0 else cdf + term
        small = F.when(u < cdf, F.lit(k)) if small is None else small.when(u < cdf, F.lit(k))
    small = small.otherwise(F.lit(9))
    large = F.greatest(F.lit(0), F.round(lam + F.sqrt(lam) * std_normal(f"{salt}_z", *keys))).cast("int")
    return F.when(lam < 2.0, small).otherwise(large)


def haversine_km(lat1, lon1, lat2, lon2) -> Column:
    lat1, lon1, lat2, lon2 = (_col(c) for c in (lat1, lon1, lat2, lon2))
    dlat = F.radians(lat2 - lat1)
    dlon = F.radians(lon2 - lon1)
    a = F.pow(F.sin(dlat / 2), 2) + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.pow(F.sin(dlon / 2), 2)
    return F.lit(2 * 6371.0) * F.asin(F.sqrt(a))


def _haversine_py(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def cities_df() -> DataFrame:
    rows = [(i + 1, c[0], c[1], c[2], c[3]) for i, c in enumerate(CITIES)]
    return spark.createDataFrame(rows, "city_id int, city string, state string, latitude double, longitude double")


def far_cities_df() -> DataFrame:
    """(home_city_id, far_rank, far_city_id, far_count) for cities at least FAR_CITY_MIN_KM apart."""
    rows = []
    for i, a in enumerate(CITIES):
        far = [j + 1 for j, b in enumerate(CITIES) if j != i and _haversine_py(a[2], a[3], b[2], b[3]) >= FAR_CITY_MIN_KM]
        rows += [(i + 1, rank, city_id, len(far)) for rank, city_id in enumerate(far)]
    return spark.createDataFrame(rows, "home_city_id int, far_rank int, far_city_id int, far_count int")


def add_primary_key(table: str, column: str) -> None:
    spark.sql(f"ALTER TABLE {fq(table)} ALTER COLUMN {column} SET NOT NULL")
    spark.sql(f"ALTER TABLE {fq(table)} ADD CONSTRAINT {table}_pk PRIMARY KEY ({column})")
