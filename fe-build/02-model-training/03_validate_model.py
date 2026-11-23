# Databricks notebook source
# MAGIC %md
# MAGIC # 3. Validate the model and choose the blocking threshold
# MAGIC
# MAGIC The honest test: Oct 28–31 transactions the model has never seen, with the real fraud mix
# MAGIC (20% sample of all rows, ~0.35% fraud).
# MAGIC
# MAGIC 1. **Score** them with `fe.score_batch`, which runs the registered model's own recipe (look up
# MAGIC    `card_profile`, run the 7 functions, call `FraudModel`) — the same path serving will use.
# MAGIC 2. **Overall scores**: PR-AUC and ROC-AUC.
# MAGIC 3. **Threshold table**: for each candidate threshold, how much fraud is caught and how many
# MAGIC    normal transactions would be wrongly blocked.
# MAGIC 4. **Catch rate by fraud type** at the chosen threshold.
# MAGIC 5. **Log** the results to the model's training run.
# MAGIC 6. **Promote** (last cell, run after choosing the threshold): alias the version as `champion`
# MAGIC    and store the threshold on it.
# MAGIC
# MAGIC `card_profile` must still hold the `training` window (Oct 1–21). Run this **before** rebuilding
# MAGIC it with `serving`, otherwise the profiles would include the validation days.

# COMMAND ----------

# MAGIC %run ./00_ml_config

# COMMAND ----------

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()
client = MlflowClient()

dbutils.widgets.text("model_version", "", "Model version (blank = latest)")
dbutils.widgets.text("threshold", "0.5", "Blocking threshold")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks

# COMMAND ----------

profile_meta = spark.table(fq("card_profile")).select("profile_window", "profile_as_of_ts").first()
assert profile_meta.profile_window == "training", (
    f"card_profile holds the '{profile_meta.profile_window}' window; rebuild it with profile_window = training "
    "before validating"
)

model_version = dbutils.widgets.get("model_version").strip() or max(
    (v.version for v in client.search_model_versions(f"name='{MODEL_NAME}'")), key=int
)
threshold = float(dbutils.widgets.get("threshold"))
model_uri = f"models:/{MODEL_NAME}/{model_version}"

print(f"card_profile window: {profile_meta.profile_window}, as of {profile_meta.profile_as_of_ts}")
print(f"Validating {model_uri} at threshold {threshold}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Score the validation transactions
# MAGIC
# MAGIC Only raw transaction fields go in, exactly like a live request. `is_fraud` and `fraud_type` ride
# MAGIC along for measuring and are not model inputs.

# COMMAND ----------

ts = F.col("transaction_ts")

validation_rows = (
    spark.table(fq("transactions_history"))
    .filter((ts >= F.lit(VALIDATION_START)) & (ts < F.lit(VALIDATION_END)))
    .filter(uniform("validation_sample", "transaction_id") < VALIDATION_SAMPLE_RATE)
    .select("transaction_id", "card_id", "amount_usd", "merchant_category", "channel",
            "latitude", "longitude", "transaction_ts", LABEL, "fraud_type")
)

scored = fe.score_batch(model_uri=model_uri, df=validation_rows, result_type="double")

results = (
    scored.select("transaction_id", LABEL, "fraud_type", F.col("prediction").alias("score"))
    .toPandas()
)
y = results[LABEL].astype(bool).to_numpy()
score = results["score"].to_numpy()
print(f"Scored {len(results):,} transactions, {y.sum():,} fraud ({y.mean() * 100:.2f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Overall scores

# COMMAND ----------

validation_pr_auc = average_precision_score(y, score)
validation_roc_auc = roc_auc_score(y, score)
print(f"Validation PR-AUC:  {validation_pr_auc:.4f}")
print(f"Validation ROC-AUC: {validation_roc_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC **ROC-AUC 0.9999:** "Pick any real fraud and any normal transaction. 99.99% of the time, the model
# MAGIC gives the fraud the higher score."
# MAGIC
# MAGIC **PR-AUC 0.971:** "Block transactions from the highest score downwards. On average, about 97% of
# MAGIC what gets blocked is real fraud."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Threshold table
# MAGIC
# MAGIC Block a transaction when its score is at or above the threshold.
# MAGIC
# MAGIC | Column | Meaning |
# MAGIC |---|---|
# MAGIC | `catch_rate` | share of real fraud that gets blocked |
# MAGIC | `block_accuracy` | share of blocked transactions that are real fraud |
# MAGIC | `false_alarms_per_10k_normal` | normal transactions wrongly blocked, per 10,000 normal transactions |
# MAGIC | `est_false_blocks_per_live_hour` | the same rate applied to the 1-hour live demo (~3.6M transactions) |

# COMMAND ----------

# MAGIC %md
# MAGIC **Catch rate:** out of all the real fraud, how much did we catch?
# MAGIC
# MAGIC **Block accuracy:** out of everything we blocked, how much was really fraud?

# COMMAND ----------

def threshold_metrics(t):
    flagged = score >= t
    caught = int((flagged & y).sum())
    missed = int((~flagged & y).sum())
    false_alarms = int((flagged & ~y).sum())
    normal = int((~y).sum())
    return {
        "threshold": t,
        "fraud_caught": caught,
        "fraud_missed": missed,
        "false_alarms": false_alarms,
        "catch_rate": round(caught / (caught + missed), 4),
        "block_accuracy": round(caught / (caught + false_alarms), 4) if caught + false_alarms else None,
        "false_alarms_per_10k_normal": round(false_alarms / normal * 10_000, 2),
        "est_false_blocks_per_live_hour": round(false_alarms / normal * LIVE_TARGET_TXNS * (1 - LIVE_FRAUD_TXN_RATE)),
    }


candidate_thresholds = sorted({0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, threshold})
threshold_table = pd.DataFrame([threshold_metrics(t) for t in candidate_thresholds])
display(threshold_table)

# COMMAND ----------

precision, recall, curve_thresholds = precision_recall_curve(y, score)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(curve_thresholds, recall[:-1], label="catch rate (recall)")
ax.plot(curve_thresholds, precision[:-1], label="block accuracy (precision)")
ax.axvline(threshold, color="grey", linestyle="--", label=f"chosen threshold {threshold}")
ax.set_xlabel("threshold")
ax.set_ylim(0, 1.02)
ax.set_title("Catch rate and block accuracy by threshold (validation, Oct 28–31)")
ax.legend(loc="lower left")
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Catch rate by fraud type at the chosen threshold

# COMMAND ----------

fraud_rows = results[results[LABEL].astype(bool)]
by_type = (
    fraud_rows.assign(caught=fraud_rows["score"] >= threshold)
    .groupby("fraud_type")
    .agg(fraud_txns=("caught", "size"), caught=("caught", "sum"), median_score=("score", "median"))
    .assign(catch_rate=lambda d: (d["caught"] / d["fraud_txns"]).round(4))
    .sort_values("catch_rate")
    .reset_index()
)
display(by_type)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log the results to the model's training run

# COMMAND ----------

chosen = threshold_metrics(threshold)
training_run_id = client.get_model_version(MODEL_NAME, model_version).run_id

with mlflow.start_run(run_id=training_run_id):
    mlflow.log_metrics({
        "validation_pr_auc": validation_pr_auc,
        "validation_roc_auc": validation_roc_auc,
        "validation_catch_rate": chosen["catch_rate"],
        "validation_block_accuracy": chosen["block_accuracy"] or 0.0,
        "validation_false_alarms_per_10k_normal": chosen["false_alarms_per_10k_normal"],
    })
    mlflow.log_param("validation_threshold", threshold)
    mlflow.log_table(threshold_table, "validation_thresholds.json")
    mlflow.log_table(by_type, "validation_by_fraud_type.json")
    mlflow.log_figure(fig, "validation_catch_rate_vs_accuracy.png")

print(f"Logged validation results to run {training_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Promote (run after choosing the threshold)
# MAGIC
# MAGIC Set the `threshold` widget to your choice, re-run sections 3–5, then run this cell. The serving
# MAGIC endpoint will point at the `champion` alias, and the stream will read the threshold tag.

# COMMAND ----------

client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, model_version)
client.set_model_version_tag(MODEL_NAME, model_version, THRESHOLD_TAG, str(threshold))
print(f"{MODEL_NAME} version {model_version} is now '{CHAMPION_ALIAS}', {THRESHOLD_TAG} = {threshold}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Look up a version's details
# MAGIC
# MAGIC By version number, or by alias (whichever version the alias points to right now).

# COMMAND ----------

v = client.get_model_version(MODEL_NAME, "1")
print(v.version)   # '1'
print(v.aliases)   # ['champion']
print(v.tags)      # {'fraud_threshold': '0.9'}
print(v.run_id)    # the training run

# COMMAND ----------

v = client.get_model_version_by_alias(MODEL_NAME, "champion")
print(v.version)   # '1'
print(v.aliases)   # ['champion']
print(v.tags)      # {'fraud_threshold': '0.9'}
print(v.run_id)    # the training run
