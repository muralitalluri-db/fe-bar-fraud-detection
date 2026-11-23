# Databricks notebook source
# MAGIC %md
# MAGIC # 4. Create the Lakebase online store and publish `card_profile`
# MAGIC
# MAGIC 1. **Create** the Lakebase Autoscaling project `fe-bar-lakebase` (skipped if it already exists).
# MAGIC 2. **Wait** until it's available.
# MAGIC 3. **Publish** `card_profile` to it as `card_profile_online`. Unity Catalog records the link, so
# MAGIC    the serving endpoint finds the online copy automatically from the model's feature spec.
# MAGIC 4. **Check** the online table with a direct Postgres query.
# MAGIC
# MAGIC `card_profile` must hold the `serving` window (Oct 1–31). After publishing, don't drop and
# MAGIC recreate `card_profile`; that breaks the endpoint's link to the online table.
# MAGIC Docs: https://docs.databricks.com/aws/en/machine-learning/feature-store/online-feature-store

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk psycopg

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_ml_config

# COMMAND ----------

import time

import psycopg
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient

fe = FeatureEngineeringClient()
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check `card_profile` holds the serving window

# COMMAND ----------

profile_meta = spark.table(fq("card_profile")).select("profile_window", "profile_as_of_ts").first()
assert profile_meta.profile_window == "serving", (
    f"card_profile holds the '{profile_meta.profile_window}' window; rebuild it with profile_window = serving "
    "before publishing"
)
print(f"card_profile window: {profile_meta.profile_window}, as of {profile_meta.profile_as_of_ts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create the Lakebase project

# COMMAND ----------

if fe.get_online_store(name=LAKEBASE_PROJECT):
    print(f"Online store '{LAKEBASE_PROJECT}' already exists")
else:
    fe.create_online_store(name=LAKEBASE_PROJECT, capacity=LAKEBASE_CAPACITY)
    print(f"Creating online store '{LAKEBASE_PROJECT}' ({LAKEBASE_CAPACITY})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Wait until it's available

# COMMAND ----------

while "AVAILABLE" not in str(fe.get_online_store(name=LAKEBASE_PROJECT).state):
    print("Waiting for the online store...")
    time.sleep(30)

online_store = fe.get_online_store(name=LAKEBASE_PROJECT)
print(f"Online store: {online_store.name}, state: {online_store.state}, capacity: {online_store.capacity}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Publish `card_profile`
# MAGIC
# MAGIC Creates `card_profile_online` in Lakebase, copies the rows, and links it to `card_profile` in
# MAGIC Unity Catalog. `SNAPSHOT` is a one-time full copy, since `card_profile` doesn't change after this;
# MAGIC re-running the cell copies everything again.

# COMMAND ----------

published = fe.publish_table(
    online_store=online_store,
    source_table_name=fq("card_profile"),
    online_table_name=ONLINE_PROFILE_TABLE,
    publish_mode="SNAPSHOT",
)
print(published)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Check the online table
# MAGIC
# MAGIC Run once `card_profile_online` shows **AVAILABLE** in Catalog Explorer (the first sync copies
# MAGIC ~17.5M rows, so it takes a while). Connects to Lakebase with a short-lived credential and queries
# MAGIC Postgres directly. The online table's catalog name is its Postgres database name.

# COMMAND ----------

online_db, online_schema, online_table = ONLINE_PROFILE_TABLE.split(".")

branches = list(w.postgres.list_branches(parent=f"projects/{LAKEBASE_PROJECT}"))
branch = next(
    (b for b in branches if getattr(getattr(b, "status", None), "default", False)),
    next((b for b in branches if b.name.endswith("/production")), branches[0]),
)
endpoint = list(w.postgres.list_endpoints(parent=branch.name))[0]
credential = w.postgres.generate_database_credential(endpoint=endpoint.name)

print(f"Branch: {branch.name}")
print(f"Endpoint: {endpoint.name}")

with psycopg.connect(
    host=endpoint.status.hosts.host,
    port=5432,
    dbname=online_db,
    user=w.current_user.me().user_name,
    password=credential.token,
    sslmode="require",
) as conn, conn.cursor() as cur:
    cur.execute(f'SELECT count(*) FROM "{online_schema}"."{online_table}"')
    online_rows = cur.fetchone()[0]
    cur.execute(
        f'SELECT card_id, avg_amount, online_ratio, home_lat, home_lon '
        f'FROM "{online_schema}"."{online_table}" LIMIT 5'
    )
    sample = cur.fetchall()

offline_rows = spark.table(fq("card_profile")).count()
print(f"Offline card_profile rows: {offline_rows:,}")
print(f"Online  card_profile rows: {online_rows:,}")
for row in sample:
    print(row)
