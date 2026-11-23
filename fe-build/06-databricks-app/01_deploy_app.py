# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Deploy the fraud operations app
# MAGIC
# MAGIC Streamlit app in `./app`, deployed as the Databricks App `fe-bar-fraud-ops`:
# MAGIC
# MAGIC | Panel | Reads from |
# MAGIC |---|---|
# MAGIC | Transactions scored, flagged, cards blocked | Lakebase `fe_bar.transactions` and `fe_bar.cards` (Postgres) |
# MAGIC | Blocked cards + unblock button | Lakebase `fe_bar.cards` (the button updates Postgres) |
# MAGIC | Catch rate, block accuracy | SQL warehouse: Lakebase `transactions` joined to Delta `transactions_live` |
# MAGIC | Ask Genie | Genie Conversation API on the `FE-BAR Fraud Detection` agent |
# MAGIC
# MAGIC Steps:
# MAGIC 1. **Create or update the app** with three resources: the Lakebase database, the SQL warehouse and the
# MAGIC    Genie agent. Adding the Lakebase resource creates a Postgres role for the app's service principal.
# MAGIC 2. **Grant** the service principal read access in Unity Catalog (for the warehouse and Genie queries)
# MAGIC    and in Postgres (read both tables, update `cards`).
# MAGIC 3. **Deploy** the code in `./app` and print the app URL.

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk psycopg

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../05-genie-agents/00_genie_config

# COMMAND ----------

import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.apps import (
    App,
    AppDeployment,
    AppDeploymentMode,
    AppResource,
    AppResourceGenieSpace,
    AppResourceGenieSpaceGenieSpacePermission,
    AppResourcePostgres,
    AppResourcePostgresPostgresPermission,
    AppResourceSqlWarehouse,
    AppResourceSqlWarehouseSqlWarehousePermission,
    ComputeState,
)

w = WorkspaceClient()

APP_NAME = "fe-bar-fraud-ops"
APP_DESCRIPTION = "FE-BAR fraud operations: live scoring numbers, blocked cards with unblock, and Genie."

notebook_dir = os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
APP_SOURCE_PATH = (notebook_dir if notebook_dir.startswith("/Workspace") else f"/Workspace{notebook_dir}") + "/app"
w.workspace.get_status(f"{APP_SOURCE_PATH}/app.yaml")
print(f"App source: {APP_SOURCE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create or update the app

# COMMAND ----------

genie_space = find_genie_space(w)
if genie_space is None:
    raise ValueError(f"Genie Agent '{GENIE_TITLE}' not found; run ../05-genie-agents/01_create_genie_agent first")

ops_database = next(
    d for d in w.postgres.list_databases(parent=LAKEBASE_BRANCH_PATH)
    if d.status.postgres_database == OPS_DATABASE
)

# The resource names are the keys app.yaml reads with valueFrom.
resources = [
    AppResource(
        name="postgres",
        postgres=AppResourcePostgres(
            branch=LAKEBASE_BRANCH_PATH,
            database=ops_database.name,
            permission=AppResourcePostgresPostgresPermission.CAN_CONNECT_AND_CREATE,
        ),
    ),
    AppResource(
        name="sql-warehouse",
        sql_warehouse=AppResourceSqlWarehouse(
            id=GENIE_WAREHOUSE_ID,
            permission=AppResourceSqlWarehouseSqlWarehousePermission.CAN_USE,
        ),
    ),
    AppResource(
        name="genie-space",
        genie_space=AppResourceGenieSpace(
            name=GENIE_TITLE,
            space_id=genie_space["space_id"],
            permission=AppResourceGenieSpaceGenieSpacePermission.CAN_RUN,
        ),
    ),
]

app_spec = App(name=APP_NAME, description=APP_DESCRIPTION, resources=resources)
try:
    w.apps.get(APP_NAME)
    app = w.apps.update(name=APP_NAME, app=app_spec)
    print(f"Updated app {APP_NAME}")
except NotFound:
    app = w.apps.create_and_wait(app=app_spec)
    print(f"Created app {APP_NAME}")

if app.compute_status is None or app.compute_status.state != ComputeState.ACTIVE:
    print("Starting app compute...")
    app = w.apps.start_and_wait(name=APP_NAME)

app_sp = app.service_principal_client_id
print(f"Service principal: {app.service_principal_name} ({app_sp})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Grant the app's service principal access
# MAGIC
# MAGIC **Unity Catalog**: the accuracy panel and Genie run SQL on the warehouse as the service principal, so it
# MAGIC needs `SELECT` on the Delta tables and on the Lakebase catalog schema.
# MAGIC
# MAGIC **Postgres**: the live panels connect to Lakebase directly as the service principal's role. It reads
# MAGIC both tables and updates `cards` for the unblock button. The default privilege covers `transactions`
# MAGIC being dropped and recreated when the demo is reset.
# MAGIC
# MAGIC The partial index covers only blocked cards, so the blocked list and count don't scan all 17.5M cards.

# COMMAND ----------

delta_tables = [CUSTOMERS_UC_TABLE, CARDS_DELTA_UC_TABLE, MERCHANTS_UC_TABLE, HISTORY_UC_TABLE, LIVE_UC_TABLE]

uc_grants = [
    f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `{app_sp}`",
    f"GRANT USE SCHEMA ON SCHEMA {CATALOG}.{SCHEMA} TO `{app_sp}`",
    *[f"GRANT SELECT ON TABLE {table} TO `{app_sp}`" for table in delta_tables],
    f"GRANT USE CATALOG ON CATALOG {OPS_CATALOG} TO `{app_sp}`",
    f"GRANT USE SCHEMA, SELECT ON SCHEMA {OPS_CATALOG}.{OPS_SCHEMA} TO `{app_sp}`",
]
for grant in uc_grants:
    spark.sql(grant)
    print(grant)

# COMMAND ----------

cards_table = f"{OPS_SCHEMA}.{CARDS_TABLE}"

pg_grants = [
    f'GRANT USAGE ON SCHEMA {OPS_SCHEMA} TO "{app_sp}"',
    f'GRANT SELECT ON ALL TABLES IN SCHEMA {OPS_SCHEMA} TO "{app_sp}"',
    f'ALTER DEFAULT PRIVILEGES IN SCHEMA {OPS_SCHEMA} GRANT SELECT ON TABLES TO "{app_sp}"',
    f'GRANT UPDATE ON {cards_table} TO "{app_sp}"',
    f"CREATE INDEX IF NOT EXISTS cards_blocked_idx ON {cards_table} (blocked_at DESC) WHERE status = 'BLOCKED'",
]
with lakebase_connect() as conn, conn.cursor() as cur:
    for statement in pg_grants:
        cur.execute(statement)
        print(statement)
    conn.commit()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Deploy

# COMMAND ----------

deployment = w.apps.deploy_and_wait(
    app_name=APP_NAME,
    app_deployment=AppDeployment(source_code_path=APP_SOURCE_PATH, mode=AppDeploymentMode.SNAPSHOT),
)
print(f"Deployment {deployment.deployment_id}: {deployment.status.state} - {deployment.status.message}")
print(f"Open the app: {w.apps.get(APP_NAME).url}")
