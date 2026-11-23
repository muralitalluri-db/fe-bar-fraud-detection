import os
import time

import pandas as pd
import psycopg
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

st.set_page_config(page_title="FE-BAR Fraud Operations", layout="wide")

LAKEBASE_ENDPOINT = os.environ["LAKEBASE_ENDPOINT"]
WAREHOUSE_ID = os.environ["WAREHOUSE_ID"]
GENIE_SPACE_ID = os.environ["GENIE_SPACE_ID"]
LIVE_TRANSACTIONS_TABLE = os.environ["LIVE_TRANSACTIONS_TABLE"]
SCORED_TRANSACTIONS_TABLE = os.environ["SCORED_TRANSACTIONS_TABLE"]
OPS_SCHEMA = "fe_bar"

w = WorkspaceClient()


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def lakebase_password():
    return w.postgres.generate_database_credential(endpoint=LAKEBASE_ENDPOINT).token


def lakebase_query(sql, params=None):
    # Host, port, database, user and sslmode come from the PG* variables set by the postgres app resource.
    with psycopg.connect(password=lakebase_password()) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def lakebase_execute(sql, params=None):
    with psycopg.connect(password=lakebase_password()) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def warehouse_query(sql):
    response = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=sql, wait_timeout="50s"
    )
    while response.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        response = w.statement_execution.get_statement(response.statement_id)
    if response.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(response.status.error.message if response.status.error else response.status.state)
    columns = [c.name for c in response.manifest.schema.columns]
    return pd.DataFrame(response.result.data_array or [], columns=columns)


@st.cache_data(ttl=60, show_spinner=False)
def model_accuracy():
    return warehouse_query(f"""
        SELECT
          sum(CASE WHEN l.is_fraud THEN 1 ELSE 0 END)                       AS real_fraud,
          sum(CASE WHEN s.is_flagged AND l.is_fraud THEN 1 ELSE 0 END)      AS caught,
          sum(CASE WHEN s.is_flagged AND NOT l.is_fraud THEN 1 ELSE 0 END)  AS false_alarms,
          try_divide(sum(CASE WHEN s.is_flagged AND l.is_fraud THEN 1 ELSE 0 END),
                     sum(CASE WHEN l.is_fraud THEN 1 ELSE 0 END))           AS catch_rate,
          try_divide(sum(CASE WHEN s.is_flagged AND l.is_fraud THEN 1 ELSE 0 END),
                     sum(CASE WHEN s.is_flagged THEN 1 ELSE 0 END))         AS block_accuracy
        FROM {SCORED_TRANSACTIONS_TABLE} s
        JOIN {LIVE_TRANSACTIONS_TABLE} l USING (transaction_id)
    """).iloc[0]


def ask_genie(question):
    if st.session_state.conversation_id is None:
        message = w.genie.start_conversation_and_wait(GENIE_SPACE_ID, question)
        st.session_state.conversation_id = message.conversation_id
    else:
        message = w.genie.create_message_and_wait(GENIE_SPACE_ID, st.session_state.conversation_id, question)

    reply = {"role": "assistant", "text": None, "sql": None, "table": None}
    if message.error:
        reply["text"] = f"Genie could not answer: {message.error.error}"
    for attachment in message.attachments or []:
        if attachment.text:
            reply["text"] = attachment.text.content
        if attachment.query:
            reply["sql"] = attachment.query.query
            result = w.genie.get_message_attachment_query_result(
                GENIE_SPACE_ID, message.conversation_id, message.message_id, attachment.attachment_id
            )
            statement = result.statement_response
            columns = [c.name for c in statement.manifest.schema.columns]
            reply["table"] = pd.DataFrame(statement.result.data_array or [], columns=columns)
    return reply


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("FE-BAR Fraud Operations")
live_tab, genie_tab = st.tabs(["Live operations", "Ask Genie"])

with live_tab:
    auto_refresh = st.toggle("Auto-refresh every 5 seconds", value=True)

    @st.fragment(run_every="5s" if auto_refresh else None)
    def live_numbers():
        txns = lakebase_query(f"""
            SELECT count(*) AS scored,
                   count(*) FILTER (WHERE is_flagged) AS flagged,
                   max(scored_at) AS last_scored
            FROM {OPS_SCHEMA}.transactions
        """).iloc[0]
        blocked = lakebase_query(
            f"SELECT count(*) AS blocked FROM {OPS_SCHEMA}.cards WHERE status = 'BLOCKED'"
        ).iloc[0]["blocked"]

        scored, flagged = int(txns["scored"]), int(txns["flagged"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions scored", f"{scored:,}")
        col2.metric("Flagged by the model", f"{flagged:,}", f"{flagged / scored:.2%} of scored" if scored else None,
                    delta_color="off")
        col3.metric("Cards blocked", f"{int(blocked):,}")
        st.caption(f"Last transaction scored at {txns['last_scored']} (UTC)" if scored else "No transactions scored yet")

        st.subheader("Blocked cards")
        cards = lakebase_query(f"""
            SELECT c.card_id, c.customer_id, c.card_network, c.card_type, c.card_last4,
                   c.blocked_at, c.blocked_reason,
                   t.amount_usd, t.merchant_category, t.channel, t.fraud_score,
                   c.blocked_by_transaction
            FROM {OPS_SCHEMA}.cards c
            LEFT JOIN {OPS_SCHEMA}.transactions t ON t.transaction_id = c.blocked_by_transaction
            WHERE c.status = 'BLOCKED'
            ORDER BY c.blocked_at DESC
            LIMIT 200
        """)
        st.dataframe(cards, hide_index=True)

    live_numbers()

    @st.fragment
    def unblock_card():
        st.subheader("Unblock a card")
        blocked_ids = lakebase_query(f"""
            SELECT card_id FROM {OPS_SCHEMA}.cards
            WHERE status = 'BLOCKED' ORDER BY blocked_at DESC LIMIT 500
        """)["card_id"].tolist()
        if not blocked_ids:
            st.info("No blocked cards.")
            return
        card_id = st.selectbox("Blocked card (most recent first)", blocked_ids)
        if st.button("Unblock", type="primary"):
            updated = lakebase_execute(f"""
                UPDATE {OPS_SCHEMA}.cards
                SET status = 'ACTIVE', blocked_at = NULL, blocked_reason = NULL,
                    blocked_by_transaction = NULL, updated_at = now()
                WHERE card_id = %s AND status = 'BLOCKED'
            """, (card_id,))
            if updated:
                st.success(f"Card {card_id} is ACTIVE again.")
            else:
                st.warning(f"Card {card_id} was not blocked.")

    unblock_card()

    @st.fragment(run_every="60s" if auto_refresh else None)
    def accuracy():
        st.subheader("Model accuracy in the live hour")
        st.caption(
            "Measured against the hidden fraud labels in transactions_live. A real issuer only learns "
            "these labels later, from customer disputes and chargebacks."
        )
        result = model_accuracy()
        if result["real_fraud"] is None:
            st.info("No scored transactions yet.")
            return
        real_fraud, caught, false_alarms = int(result["real_fraud"]), int(result["caught"]), int(result["false_alarms"])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Catch rate", f"{float(result['catch_rate'] or 0):.1%}",
                    f"{caught:,} of {real_fraud:,} real fraud caught", delta_color="off")
        col2.metric("Block accuracy", f"{float(result['block_accuracy'] or 0):.1%}",
                    f"{caught:,} of {caught + false_alarms:,} flags were real fraud", delta_color="off")
        col3.metric("Missed fraud", f"{real_fraud - caught:,}")
        col4.metric("False alarms", f"{false_alarms:,}")

    accuracy()

with genie_tab:
    st.link_button("Open in Genie", f"{w.config.host}/genie/rooms/{GENIE_SPACE_ID}")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.conversation_id = None

    def show(message):
        with st.chat_message(message["role"]):
            if message.get("text"):
                st.markdown(message["text"])
            if message.get("table") is not None:
                st.dataframe(message["table"], hide_index=True)
            if message.get("sql"):
                with st.expander("SQL"):
                    st.code(message["sql"], language="sql")

    for message in st.session_state.messages:
        show(message)

    if question := st.chat_input("Ask about fraud, flagged transactions or blocked cards"):
        user_message = {"role": "user", "text": question}
        st.session_state.messages.append(user_message)
        show(user_message)
        with st.spinner("Genie is thinking..."):
            reply = ask_genie(question)
        st.session_state.messages.append(reply)
        show(reply)
