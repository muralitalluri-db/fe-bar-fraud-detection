# Databricks notebook source
# MAGIC %md
# MAGIC # Genie Agent config
# MAGIC
# MAGIC Shared by the notebooks in this folder through `%run ./00_genie_config`. Pulls in the scoring
# MAGIC stream config, which already includes the data generation, ML, Kafka and Lakebase settings.

# COMMAND ----------

# MAGIC %run ../04-ingest-to-lakebase/00_stream_config

# COMMAND ----------

GENIE_WAREHOUSE_ID = "84d8f89ea697d456"
GENIE_PARENT_PATH = f"/Workspace/Users/{spark.sql('SELECT current_user()').first()[0]}/ASQs/fe-bar"
GENIE_TITLE = "FE-BAR Fraud Detection"
GENIE_DESCRIPTION = (
    "Card fraud analytics: customers, cards, merchants, 30 days of labeled transaction history, "
    "the live hour scored by the fraud model, and real-time card blocks from Lakebase."
)

# Lakehouse (Delta) tables in murali_talluri.fe_bar
CUSTOMERS_UC_TABLE = fq("customers")
CARDS_DELTA_UC_TABLE = fq("cards")
MERCHANTS_UC_TABLE = fq("merchants")
HISTORY_UC_TABLE = fq("transactions_history")
LIVE_UC_TABLE = fq("transactions_live")

# Operational (Lakebase) tables, registered in Unity Catalog as fe_bar_mt_lakebase
SCORED_TRANSACTIONS_UC_TABLE = TRANSACTIONS_UC_TABLE
CARD_STATUS_UC_TABLE = CARDS_UC_TABLE


def find_genie_space(w, title=GENIE_TITLE):
    """The Genie Agent with this title, or None. Reads every page of the list API."""
    page_token = None
    while True:
        query = {"page_size": 100, **({"page_token": page_token} if page_token else {})}
        response = w.api_client.do("GET", "/api/2.0/genie/spaces", query=query)
        for space in response.get("spaces", []):
            if space.get("title") == title:
                return space
        page_token = response.get("next_page_token")
        if not page_token:
            return None
