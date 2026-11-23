# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Create the Lakebase operational tables
# MAGIC
# MAGIC In the `fe-bar-lakebase` project (database `databricks_postgres`), registered in Unity Catalog as
# MAGIC `fe_bar_mt_lakebase`:
# MAGIC
# MAGIC | Table | What it holds |
# MAGIC |---|---|
# MAGIC | `fe_bar.transactions` | every scored transaction, written by the stream (recreated empty on each run) |
# MAGIC | `fe_bar.cards` | one row per card (~17.5M), `ACTIVE` until a flagged transaction blocks it |
# MAGIC
# MAGIC A trigger on `fe_bar.transactions` sets the card to `BLOCKED` whenever a flagged transaction is
# MAGIC written. The stream doesn't change; Lakebase applies the rule.
# MAGIC
# MAGIC Re-running this notebook resets the demo: `transactions` is recreated empty and every card goes
# MAGIC back to `ACTIVE` (the 17.5M cards are only loaded the first time).
# MAGIC Timestamp columns are `timestamptz` because the streaming sink maps Spark timestamps to it.

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk psycopg

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_stream_config

# COMMAND ----------

transactions_table = f"{OPS_SCHEMA}.{TRANSACTIONS_TABLE}"
cards_table = f"{OPS_SCHEMA}.{CARDS_TABLE}"

with lakebase_connect() as conn, conn.cursor() as cur:
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {OPS_SCHEMA}")

    cur.execute(f"DROP TABLE IF EXISTS {transactions_table}")
    cur.execute(f"""
        CREATE TABLE {transactions_table} (
            transaction_id       TEXT PRIMARY KEY,
            card_id              TEXT NOT NULL,
            customer_id          TEXT,
            merchant_id          TEXT,
            merchant_category    TEXT,
            mcc                  TEXT,
            channel              TEXT,
            amount_usd           DOUBLE PRECISION,
            currency             TEXT,
            latitude             DOUBLE PRECISION,
            longitude            DOUBLE PRECISION,
            transaction_ts       TIMESTAMPTZ,
            fraud_score          DOUBLE PRECISION,
            is_flagged           BOOLEAN,
            model_version        TEXT,
            kafka_ts             TIMESTAMPTZ,
            scored_at            TIMESTAMPTZ,
            lakebase_written_at  TIMESTAMPTZ DEFAULT now()
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {cards_table} (
            card_id                 TEXT PRIMARY KEY,
            customer_id             TEXT,
            card_last4              TEXT,
            card_network            TEXT,
            card_type               TEXT,
            status                  TEXT NOT NULL DEFAULT 'ACTIVE',
            blocked_at              TIMESTAMPTZ,
            blocked_reason          TEXT,
            blocked_by_transaction  TEXT,
            updated_at              TIMESTAMPTZ DEFAULT now()
        )
    """)
    conn.commit()

print(f"Created {OPS_DATABASE}.{transactions_table} (empty)")
print(f"Ensured {OPS_DATABASE}.{cards_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the Lakebase database in Unity Catalog
# MAGIC
# MAGIC Creates the catalog `fe_bar_mt_lakebase` on top of `databricks_postgres` in `fe-bar-lakebase`, so
# MAGIC the tables appear as `fe_bar_mt_lakebase.fe_bar.*` and Spark can write to them with `.toTable()`.
# MAGIC Skipped if the catalog already exists.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.postgres import Catalog, CatalogCatalogSpec

w = WorkspaceClient()

try:
    w.catalogs.get(OPS_CATALOG)
    print(f"Catalog already registered: {OPS_CATALOG}")
except NotFound:
    catalog = w.postgres.create_catalog(
        catalog=Catalog(spec=CatalogCatalogSpec(
            postgres_database=OPS_DATABASE,
            branch=LAKEBASE_BRANCH_PATH,
        )),
        catalog_id=OPS_CATALOG,
    ).wait()
    print(f"Catalog registered: {catalog.name}")

print(f"Stream target: {TRANSACTIONS_UC_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the cards (first run) or reset them to ACTIVE (later runs)
# MAGIC
# MAGIC The first load copies ~17.5M cards from the Delta `cards` table with the Lakebase sink, using a
# MAGIC one-time `availableNow` stream. Later runs just reset any blocked cards.

# COMMAND ----------

with lakebase_connect() as conn, conn.cursor() as cur:
    cur.execute(f"SELECT count(*) FROM {cards_table}")
    existing_cards = cur.fetchone()[0]

if existing_cards == 0:
    print("Loading cards into Lakebase...")
    dbutils.fs.rm(CARDS_LOAD_CHECKPOINT, True)
    (
        spark.readStream.table(fq("cards"))
        .select(
            "card_id", "customer_id", "card_last4", "card_network", "card_type",
            F.lit("ACTIVE").alias("status"),
        )
        .writeStream
        .queryName("load_cards_to_lakebase")
        .option("upsertkey", "card_id")
        .option("batchsize", 10000)
        .option("checkpointLocation", CARDS_LOAD_CHECKPOINT)
        .trigger(availableNow=True)
        .outputMode("update")
        .toTable(CARDS_UC_TABLE)
        .awaitTermination()
    )
else:
    with lakebase_connect() as conn, conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {cards_table}
            SET status = 'ACTIVE', blocked_at = NULL, blocked_reason = NULL,
                blocked_by_transaction = NULL, updated_at = now()
            WHERE status <> 'ACTIVE'
        """)
        reset = cur.rowcount
        conn.commit()
    print(f"{existing_cards:,} cards already loaded; reset {reset:,} blocked cards to ACTIVE")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Blocking trigger
# MAGIC
# MAGIC Fires for every transaction written with `is_flagged = true` (new rows and upserts). Sets that
# MAGIC card to `BLOCKED` with the time, the score, and the transaction, unless it's already blocked.

# COMMAND ----------

with lakebase_connect() as conn, conn.cursor() as cur:
    cur.execute(f"""
        CREATE OR REPLACE FUNCTION {OPS_SCHEMA}.block_card_on_flag() RETURNS trigger AS $$
        BEGIN
            UPDATE {cards_table}
            SET status = 'BLOCKED',
                blocked_at = now(),
                blocked_reason = 'fraud score ' || round(NEW.fraud_score::numeric, 4),
                blocked_by_transaction = NEW.transaction_id,
                updated_at = now()
            WHERE card_id = NEW.card_id AND status <> 'BLOCKED';
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    cur.execute(f"""
        CREATE TRIGGER block_card_on_flag
        AFTER INSERT OR UPDATE OF is_flagged ON {transactions_table}
        FOR EACH ROW
        WHEN (NEW.is_flagged)
        EXECUTE FUNCTION {OPS_SCHEMA}.block_card_on_flag()
    """)
    conn.commit()

print(f"Trigger block_card_on_flag created on {transactions_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Card status
# MAGIC
# MAGIC Re-run while the stream is running to watch blocked cards rise.

# COMMAND ----------

with lakebase_connect() as conn, conn.cursor() as cur:
    cur.execute(f"SELECT status, count(*) FROM {cards_table} GROUP BY status ORDER BY status")
    for status, count in cur.fetchall():
        print(f"{status:8} {count:,}")
