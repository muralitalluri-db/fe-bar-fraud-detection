# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Create the fraud Genie Agent
# MAGIC
# MAGIC 1. **Define** the agent: tables, instructions, joins, sample questions and example SQL.
# MAGIC 2. **Create** it with the Genie API (`POST /api/2.0/genie/spaces`), or update it if an agent with
# MAGIC    the same title already exists.
# MAGIC 3. **Verify**: read it back and ask two test questions, one over Delta tables and one over the
# MAGIC    Lakebase tables.
# MAGIC
# MAGIC Genie Agents were formerly Genie Spaces; the API still uses `/genie/spaces`.
# MAGIC Docs: https://docs.databricks.com/aws/en/genie-agents/conversation-api

# COMMAND ----------

# MAGIC %run ./00_genie_config

# COMMAND ----------

import json
from uuid import uuid4

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def new_id():
    return uuid4().hex


def lines(sql):
    return sql.strip().splitlines(keepends=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define the agent

# COMMAND ----------

def entity_columns(*names):
    return [{"column_name": n, "enable_format_assistance": True, "enable_entity_matching": True} for n in names]


tables = [
    {
        "identifier": CUSTOMERS_UC_TABLE,
        "description": ["One row per customer (5M): name, contact details, home city and state, segment, number of cards."],
        "column_configs": entity_columns("customer_segment", "home_city", "home_state"),
    },
    {
        "identifier": CARDS_DELTA_UC_TABLE,
        "description": ["One row per card (~17.5M), 2-5 per customer: network, credit or debit, last 4 digits, credit limit."],
        "column_configs": entity_columns("card_network", "card_type"),
    },
    {
        "identifier": MERCHANTS_UC_TABLE,
        "description": ["One row per merchant (100k): name, category, risk level, online or in-store, city and state."],
        "column_configs": entity_columns("merchant_category", "risk_level", "channel", "city", "state"),
    },
    {
        "identifier": HISTORY_UC_TABLE,
        "description": [
            "30 days of card transactions (Oct 1 12:00 to Oct 31 12:00 UTC, ~541M rows) with fraud labels. "
            "is_fraud marks confirmed fraud; fraud_type is the fraud pattern."
        ],
        "column_configs": entity_columns("merchant_category", "channel", "fraud_type"),
    },
    {
        "identifier": LIVE_UC_TABLE,
        "description": [
            "The live hour after the history (Oct 31 12:00-13:00 UTC, ~3.6M rows) with hidden fraud labels "
            "(is_fraud, fraud_type). This is what was streamed through the model."
        ],
        "column_configs": entity_columns("merchant_category", "channel", "fraud_type"),
    },
    {
        "identifier": SCORED_TRANSACTIONS_UC_TABLE,
        "description": [
            "Live transactions scored in real time by the fraud model and written to Lakebase. fraud_score is 0-1; "
            "is_flagged is true when fraud_score >= 0.9. No fraud labels here; join to transactions_live on "
            "transaction_id for is_fraud."
        ],
        "column_configs": entity_columns("merchant_category", "channel"),
    },
    {
        "identifier": CARD_STATUS_UC_TABLE,
        "description": [
            "Current status of every card in Lakebase. status is ACTIVE or BLOCKED; a card is BLOCKED automatically "
            "when one of its transactions is flagged. blocked_at, blocked_reason and blocked_by_transaction explain the block."
        ],
        "column_configs": entity_columns("status", "card_network", "card_type"),
    },
]

text_instructions = """
This agent covers card fraud for a card issuer.

- Lakehouse tables (murali_talluri.fe_bar): customers, cards, merchants, transactions_history (30 days, labeled),
  transactions_live (the live hour, labeled).
- Operational tables (fe_bar_mt_lakebase.fe_bar, Lakebase): transactions = live transactions scored by the model;
  cards = current card status (ACTIVE or BLOCKED).
- "Fraud" means is_fraud = true (the real answer). "Flagged" means the model flagged it: is_flagged = true,
  i.e. fraud_score >= 0.9. "Blocked" means cards.status = 'BLOCKED' in the Lakebase cards table.
- Catch rate = flagged real fraud / all real fraud. Block accuracy = flagged real fraud / all flagged.
  Compute both by joining fe_bar_mt_lakebase.fe_bar.transactions to murali_talluri.fe_bar.transactions_live
  on transaction_id.
- Fraud losses = SUM(amount_usd) of fraud transactions. Amounts are in USD.
- For "right now" or "live" questions, use the Lakebase tables. For trends over October, use transactions_history.
- All timestamps are UTC.
"""

join_specs = [
    ("transactions_history", HISTORY_UC_TABLE, "cards", CARDS_DELTA_UC_TABLE, "card_id", "card_id", "MANY_TO_ONE"),
    ("transactions_history", HISTORY_UC_TABLE, "merchants", MERCHANTS_UC_TABLE, "merchant_id", "merchant_id", "MANY_TO_ONE"),
    ("transactions_history", HISTORY_UC_TABLE, "customers", CUSTOMERS_UC_TABLE, "customer_id", "customer_id", "MANY_TO_ONE"),
    ("cards", CARDS_DELTA_UC_TABLE, "customers", CUSTOMERS_UC_TABLE, "customer_id", "customer_id", "MANY_TO_ONE"),
    ("transactions_live", LIVE_UC_TABLE, "merchants", MERCHANTS_UC_TABLE, "merchant_id", "merchant_id", "MANY_TO_ONE"),
    ("scored_transactions", SCORED_TRANSACTIONS_UC_TABLE, "transactions_live", LIVE_UC_TABLE, "transaction_id", "transaction_id", "ONE_TO_ONE"),
    ("scored_transactions", SCORED_TRANSACTIONS_UC_TABLE, "card_status", CARD_STATUS_UC_TABLE, "card_id", "card_id", "MANY_TO_ONE"),
    ("card_status", CARD_STATUS_UC_TABLE, "cards", CARDS_DELTA_UC_TABLE, "card_id", "card_id", "ONE_TO_ONE"),
    ("card_status", CARD_STATUS_UC_TABLE, "customers", CUSTOMERS_UC_TABLE, "customer_id", "customer_id", "MANY_TO_ONE"),
]

example_sqls = [
    (
        "How many cards are blocked right now?",
        f"""
SELECT count(*) AS blocked_cards
FROM {CARD_STATUS_UC_TABLE}
WHERE status = 'BLOCKED'
""",
    ),
    (
        "Which merchant categories have the most flagged transactions?",
        f"""
SELECT merchant_category, count(*) AS flagged_transactions
FROM {SCORED_TRANSACTIONS_UC_TABLE}
WHERE is_flagged
GROUP BY merchant_category
ORDER BY flagged_transactions DESC
""",
    ),
    (
        "How well did the model catch fraud in the live hour?",
        f"""
SELECT
  sum(CASE WHEN l.is_fraud THEN 1 ELSE 0 END)                       AS real_fraud,
  sum(CASE WHEN s.is_flagged AND l.is_fraud THEN 1 ELSE 0 END)      AS caught,
  sum(CASE WHEN s.is_flagged AND NOT l.is_fraud THEN 1 ELSE 0 END)  AS false_alarms,
  round(sum(CASE WHEN s.is_flagged AND l.is_fraud THEN 1 ELSE 0 END)
        / sum(CASE WHEN l.is_fraud THEN 1 ELSE 0 END), 4)           AS catch_rate,
  round(sum(CASE WHEN s.is_flagged AND l.is_fraud THEN 1 ELSE 0 END)
        / sum(CASE WHEN s.is_flagged THEN 1 ELSE 0 END), 4)         AS block_accuracy
FROM {SCORED_TRANSACTIONS_UC_TABLE} s
JOIN {LIVE_UC_TABLE} l USING (transaction_id)
""",
    ),
    (
        "What were fraud losses by fraud type in October?",
        f"""
SELECT fraud_type, count(*) AS fraud_transactions, round(sum(amount_usd), 2) AS fraud_losses_usd
FROM {HISTORY_UC_TABLE}
WHERE is_fraud
GROUP BY fraud_type
ORDER BY fraud_losses_usd DESC
""",
    ),
    (
        "Which customers have the most blocked cards?",
        f"""
SELECT c.customer_id, c.first_name, c.last_name, c.home_city, c.home_state, count(*) AS blocked_cards
FROM {CARD_STATUS_UC_TABLE} cs
JOIN {CUSTOMERS_UC_TABLE} c ON cs.customer_id = c.customer_id
WHERE cs.status = 'BLOCKED'
GROUP BY c.customer_id, c.first_name, c.last_name, c.home_city, c.home_state
ORDER BY blocked_cards DESC
LIMIT 10
""",
    ),
]

sample_questions = [
    "How many cards are blocked right now?",
    "Which merchant categories have the most flagged transactions?",
    "How well did the model catch fraud in the live hour?",
    "What were fraud losses by fraud type in October?",
    "Which customers have the most blocked cards?",
]

serialized_space = {
    "version": 2,
    "config": {
        "sample_questions": [{"id": new_id(), "question": [q]} for q in sample_questions],
    },
    "data_sources": {
        "tables": sorted(tables, key=lambda t: t["identifier"]),
    },
    "instructions": {
        "text_instructions": [{"id": new_id(), "content": [text_instructions.strip()]}],
        "example_question_sqls": [
            {"id": new_id(), "question": [question], "sql": lines(sql)} for question, sql in example_sqls
        ],
        "join_specs": [
            {
                "id": new_id(),
                "left": {"identifier": left_table, "alias": left_alias},
                "right": {"identifier": right_table, "alias": right_alias},
                "sql": [
                    f"`{left_alias}`.`{left_col}` = `{right_alias}`.`{right_col}`",
                    f"--rt=FROM_RELATIONSHIP_TYPE_{relationship}--",
                ],
            }
            for left_alias, left_table, right_alias, right_table, left_col, right_col, relationship in join_specs
        ],
    },
}

print(f"{len(tables)} tables, {len(example_sqls)} example queries, {len(join_specs)} joins, "
      f"{len(sample_questions)} sample questions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create or update the agent

# COMMAND ----------

for table in serialized_space["data_sources"]["tables"]:
    table["column_configs"] = sorted(table["column_configs"], key=lambda c: c["column_name"])

by_id = lambda items: sorted(items, key=lambda item: item["id"])
serialized_space["config"]["sample_questions"] = by_id(serialized_space["config"]["sample_questions"])
for key in ["text_instructions", "example_question_sqls", "join_specs"]:
    serialized_space["instructions"][key] = by_id(serialized_space["instructions"][key])

body = {
    "title": GENIE_TITLE,
    "description": GENIE_DESCRIPTION,
    "warehouse_id": GENIE_WAREHOUSE_ID,
    "serialized_space": json.dumps(serialized_space),
}

existing = find_genie_space(w)

if existing:
    space_id = existing["space_id"]
    w.api_client.do("PATCH", f"/api/2.0/genie/spaces/{space_id}", body=body)
    print(f"Updated Genie Agent {space_id}")
else:
    space_id = w.api_client.do("POST", "/api/2.0/genie/spaces", body={**body, "parent_path": GENIE_PARENT_PATH})["space_id"]
    print(f"Created Genie Agent {space_id}")

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
print(f"Open it: https://{workspace_host}/genie/rooms/{space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Verify
# MAGIC
# MAGIC Read the agent back, then ask one question over Delta tables and one over the Lakebase tables.

# COMMAND ----------

stored = w.api_client.do("GET", f"/api/2.0/genie/spaces/{space_id}", query={"include_serialized_space": "true"})
stored_tables = [t["identifier"] for t in json.loads(stored["serialized_space"])["data_sources"]["tables"]]
print(f"Title: {stored['title']}")
for identifier in stored_tables:
    print(f"  {identifier}")

# COMMAND ----------

test_questions = [
    "What were fraud losses by fraud type in October?",   # Delta: transactions_history
    "How many cards are blocked right now?",               # Lakebase: fe_bar_mt_lakebase.fe_bar.cards
]

get_query_result = getattr(w.genie, "get_message_attachment_query_result", None) or getattr(
    w.genie, "get_message_query_result_by_attachment"
)

for question in test_questions:
    print(f"\nQ: {question}")
    message = w.genie.start_conversation_and_wait(space_id, question)
    message_id = getattr(message, "message_id", None) or message.id
    print(f"Status: {message.status}")
    for attachment in message.attachments or []:
        if attachment.text:
            print(f"Answer: {attachment.text.content}")
        if attachment.query:
            print(f"SQL:\n{attachment.query.query}")
            result = get_query_result(space_id, message.conversation_id, message_id, attachment.attachment_id)
            print(f"Rows: {result.statement_response.result.data_array}")
