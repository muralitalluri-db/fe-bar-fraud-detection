# FE-BAR: real-time card fraud detection on Databricks

**Industry:** Financial Services · consumer card issuing · card fraud and risk operations

## Business problem

Card fraud comes in bursts: a stolen card is tested with small charges and then drained, or a taken-over
account makes a run of purchases within minutes. When detection runs in batch, the issuer finds the burst
after it's over and absorbs the loss.

| Owner | Role | KPI |
|---|---|---|
| Chief Risk Officer | Funds the program; owns the fraud-loss line | Fraud losses (USD) |
| Head of Card Fraud Strategy | Runs fraud operations | Good customers wrongly blocked; time to restore a blocked card |

## Outcome

Measured on a one-hour replay of 3.6M synthetic transactions (`07-business-impact/01_business_impact`):

| KPI | Baseline: batch detection | Built: real-time blocking | Phase 2: score at authorization |
|---|---|---|---|
| Fraud losses | $10.0M | $5.2M (-48%) | $1.2M (-88%) |
| Cards with fraud blocked | 0 of 4,737 | 4,175 of 4,737 | 4,175 of 4,737 |
| Good customers' cards blocked | 0 | 1,628 of 17.5M (fewer than 1 in 10,000) | 1,628 |
| Good spend declined | $0 | $53K (0.02% of purchase volume) | $1.5M (0.54% of purchase volume) |
| Time to restore a wrongly blocked card | Call-center queue | One click in the fraud-ops app | One click in the fraud-ops app |

Real-time blocking (built) declines a card's transactions after its first flag. Phase 2 calls the same model
inside the authorization decision, so the flagged transaction is declined too: fraud losses fall further, but
false alarms are now declined as well, which is the trade-off the risk and fraud owners decide together.

## How it works

Card transactions stream from Kafka, get scored by an ML model in Spark real-time mode, and land in
Lakebase, where a Postgres trigger blocks the card when a transaction is flagged. A Genie Agent and a
Databricks App sit on top for analysts.

```
Delta (synthetic data) -> Kafka -> Spark real-time mode -> Model Serving -> Lakebase (trigger blocks card)
                                                                              |
                                                       Genie Agent + Databricks App
```

## What's used

- **Lakeflow**: Spark Structured Streaming real-time mode from Kafka to Lakebase
- **Lakebase**: online feature store, operational tables, card-blocking trigger
- **ML**: Feature Engineering (feature tables + on-demand UC functions), XGBoost, MLflow, Model Serving
- **Genie Agent**: created with the Genie API over Delta and Lakebase tables
- **Databricks App**: Streamlit; live numbers, blocked cards with unblock, model accuracy, Genie chat

## Data

5M customers, ~17.5M cards, 100k merchants. 30 days of labeled history (~541M transactions) with five
fraud patterns, plus one live hour (~3.6M transactions, ~1,000 per second) replayed through Kafka.

## Prerequisites

- Unity Catalog catalog and schema (set `CATALOG` and `SCHEMA` in `01-data-generation/00_config.py`)
- A Lakebase Autoscaling project (`LAKEBASE_PROJECT` in `02-model-training/00_ml_config.py`)
- A Kafka cluster reachable from the workspace, with the broker list stored in a secret
  (`KAFKA_SECRET_SCOPE` / `KAFKA_SECRET_KEY` in `03-kafka-ingest/00_ingest_config.py`)
- A Pro or Serverless SQL warehouse (`GENIE_WAREHOUSE_ID` in `05-genie-agents/00_genie_config.py`)
- A cluster with Databricks Runtime ML that supports real-time mode

## Run order

| Step | Notebook | What it does |
|---|---|---|
| 1 | `01-data-generation/01` to `03` | Customers, cards, merchants, history, live hour |
| 2 | `01-data-generation/04_compute_card_profile` (`profile_window=training`) | Card profiles for training |
| 3 | `01-data-generation/05_build_live_stream_table` | One file per second for the Kafka replay |
| 4 | `02-model-training/01` to `03` | Feature functions, train, validate, promote to `champion` |
| 5 | `01-data-generation/04_compute_card_profile` (`profile_window=serving`) | Card profiles for serving |
| 6 | `02-model-training/04_lakebase_online_store` | Publish card profiles to Lakebase |
| 7 | `02-model-training/05_deploy_endpoint` | Model Serving endpoint |
| 8 | `03-kafka-ingest/01_create_delete_topic` | Create the Kafka topic |
| 9 | `04-ingest-to-lakebase/01_create_lakebase_tables` | Lakebase tables, UC catalog, trigger (also resets the demo) |
| 10 | `04-ingest-to-lakebase/02_score_stream_to_lakebase` | Real-time scoring stream into Lakebase |
| 11 | `03-kafka-ingest/02_stream_to_kafka` | Replay the live hour into Kafka |
| 12 | `05-genie-agents/01_create_genie_agent` | Genie Agent |
| 13 | `06-databricks-app/01_deploy_app` | Databricks App |
| 14 | `07-business-impact/01_business_impact` (serverless) | Fraud losses and good-customer impact, baseline to target |

Start step 10 a minute or two before step 11; the first micro-batch includes warm-up time.

## Technical results from one run

| Metric | Value |
|---|---|
| Throughput | ~1,000 transactions per second for the full hour |
| End-to-end latency (Kafka to Lakebase, incl. model call) | P50 ~260 ms, P99 ~430 ms |
| Validation PR-AUC | 0.971 |
| Live catch rate / block accuracy (threshold 0.9) | 90.2% / 88.8% |
| Cards blocked by the trigger | 5,803 |
