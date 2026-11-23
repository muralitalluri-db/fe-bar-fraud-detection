# Databricks notebook source
# MAGIC %md
# MAGIC # Scoring stream config
# MAGIC
# MAGIC Shared by the notebooks in this folder through `%run ./00_stream_config`. Pulls in the ML config
# MAGIC (model, endpoint, Lakebase project) and the Kafka ingest config (topic, brokers, checkpoints).

# COMMAND ----------

# MAGIC %run ../02-model-training/00_ml_config

# COMMAND ----------

# MAGIC %run ../03-kafka-ingest/00_ingest_config

# COMMAND ----------

# Lakebase compute endpoint inside the fe-bar-lakebase project.
LAKEBASE_BRANCH = "production"
LAKEBASE_ENDPOINT_ID = "primary"
LAKEBASE_BRANCH_PATH = f"projects/{LAKEBASE_PROJECT}/branches/{LAKEBASE_BRANCH}"
LAKEBASE_ENDPOINT_PATH = f"{LAKEBASE_BRANCH_PATH}/endpoints/{LAKEBASE_ENDPOINT_ID}"

# Operational tables live in their own Postgres database/schema, separate from card_profile_online.
OPS_DATABASE = "databricks_postgres"
OPS_SCHEMA = "fe_bar"
TRANSACTIONS_TABLE = "transactions"

# Unity Catalog catalog registered on top of OPS_DATABASE, so the Lakebase sink can write with
# .toTable() and Unity Catalog manages the credentials.
OPS_CATALOG = "fe_bar_mt_lakebase"
TRANSACTIONS_UC_TABLE = f"{OPS_CATALOG}.{OPS_SCHEMA}.{TRANSACTIONS_TABLE}"

# One row per card; the blocking trigger sets status to BLOCKED when a flagged transaction lands.
CARDS_TABLE = "cards"
CARDS_UC_TABLE = f"{OPS_CATALOG}.{OPS_SCHEMA}.{CARDS_TABLE}"
CARDS_LOAD_CHECKPOINT = f"{CHECKPOINT_VOLUME}/load_cards_to_lakebase"

SCORING_QUERY_NAME = "score_to_lakebase"
SCORING_CHECKPOINT = f"{CHECKPOINT_VOLUME}/{SCORING_QUERY_NAME}"

# Columns written to the Lakebase transactions table, in order.
SCORED_COLUMNS = [
    "transaction_id", "card_id", "customer_id", "merchant_id", "merchant_category", "mcc", "channel",
    "amount_usd", "currency", "latitude", "longitude", "transaction_ts",
    "fraud_score", "is_flagged", "model_version", "kafka_ts", "scored_at",
]


def lakebase_connect(dbname=OPS_DATABASE):
    """psycopg connection to the Lakebase project with a fresh one-hour credential."""
    import psycopg
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    endpoint = w.postgres.get_endpoint(name=LAKEBASE_ENDPOINT_PATH)
    credential = w.postgres.generate_database_credential(endpoint=LAKEBASE_ENDPOINT_PATH)
    return psycopg.connect(
        host=endpoint.status.hosts.host,
        port=5432,
        dbname=dbname,
        user=w.current_user.me().user_name,
        password=credential.token,
        sslmode="require",
    )
