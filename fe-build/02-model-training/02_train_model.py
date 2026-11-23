# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Build the training set and train the fraud model
# MAGIC
# MAGIC One `TrainingSet` object does both jobs:
# MAGIC
# MAGIC 1. **Build the training set**: the Oct 21–28 transactions (all fraud + 20% of normal), with each
# MAGIC    card's profile looked up from `card_profile` and the 7 on-demand features computed.
# MAGIC 2. **Train XGBoost** on it, holding back 10% of the rows for early stopping.
# MAGIC 3. **Log to MLflow**: settings, scores and feature importance.
# MAGIC 4. **Register** the model in Unity Catalog as `fraud_model`. `fe.log_model` saves the same
# MAGIC    `TrainingSet` recipe with the model, so the serving endpoint looks up `card_profile` and runs
# MAGIC    the 7 functions itself.
# MAGIC
# MAGIC `card_profile` must hold the `training` window (Oct 1–21). The validation days (Oct 28–31) are
# MAGIC not used here; they're for step 03. Requires a Databricks Runtime ML cluster.

# COMMAND ----------

# MAGIC %run ./00_ml_config

# COMMAND ----------

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check `card_profile` holds the training window
# MAGIC
# MAGIC If this fails, re-run `01-data-generation/04_compute_card_profile` with `profile_window = training`.

# COMMAND ----------

profile_meta = spark.table(fq("card_profile")).select("profile_window", "profile_as_of_ts").first()
assert profile_meta.profile_window == "training", (
    f"card_profile holds the '{profile_meta.profile_window}' window; rebuild it with profile_window = training"
)
print(f"card_profile window: {profile_meta.profile_window}, as of {profile_meta.profile_as_of_ts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training settings

# COMMAND ----------

# Fixed category lists so training and serving encode text columns identically.
CATEGORICAL_LEVELS = {
    "merchant_category": CATEGORY_NAMES,
    "channel": ["IN_STORE", "ONLINE"],
    "card_type": ["CREDIT", "DEBIT"],
    "card_network": [name for name, _ in CARD_NETWORKS],
    "customer_segment": list(SEGMENTS),
}

# Share of training rows held back to decide when to stop adding trees.
EARLY_STOPPING_SHARE = 0.10

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    # Training data is ~1.7% fraud after sampling; count each fraud row ~7x.
    scale_pos_weight=7,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="aucpr",
    early_stopping_rounds=30,
    n_jobs=-1,
    random_state=42,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model wrapper
# MAGIC
# MAGIC At serving time the endpoint hands the model the looked-up and computed features. The wrapper
# MAGIC puts them in the training column order, applies the same type conversions as training, and
# MAGIC returns a fraud score between 0 and 1 (not a yes/no).

# COMMAND ----------

def prepare_features(df, feature_names, categorical_levels):
    X = df[feature_names].copy()
    for col in feature_names:
        if col in categorical_levels:
            X[col] = pd.Categorical(X[col], categories=categorical_levels[col])
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")
    return X


class FraudModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model, feature_names, categorical_levels):
        self.model = model
        self.feature_names = feature_names
        self.categorical_levels = categorical_levels

    def predict(self, context, model_input):
        X = prepare_features(model_input, self.feature_names, self.categorical_levels)
        return self.model.predict_proba(X)[:, 1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build the training set
# MAGIC
# MAGIC Training rows: Oct 21–28, every fraud row plus 20% of normal rows (picked by a hash of
# MAGIC `transaction_id`, so the same rows every run). `exclude_columns` keeps the model's inputs to
# MAGIC `MODEL_FEATURES`; the helper columns are still looked up because the functions need them.

# COMMAND ----------

ts = F.col("transaction_ts")

train_labels = (
    spark.table(fq("transactions_history"))
    .filter((ts >= F.lit(TRAIN_START)) & (ts < F.lit(TRAIN_END)))
    .filter(F.col("is_fraud") | (uniform("train_sample", "transaction_id") < TRAIN_NORMAL_SAMPLE_RATE))
    .select("card_id", "amount_usd", "merchant_category", "channel",
            "latitude", "longitude", "transaction_ts", LABEL)
)

training_set = fe.create_training_set(
    df=train_labels,
    feature_lookups=feature_lookups(),
    label=LABEL,
    exclude_columns=["card_id", "latitude", "longitude", "transaction_ts", *PROFILE_HELPER_COLUMNS],
)

train_df = training_set.load_df()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load it onto the driver
# MAGIC
# MAGIC `load_df()` runs the lookup and the 7 functions. The ~25M result rows are more than the
# MAGIC driver's default result-size limit for `toPandas()`, so they're staged as Parquet in a Unity
# MAGIC Catalog volume and read straight into pandas.

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.ml_staging")
staging_path = f"/Volumes/{CATALOG}/{SCHEMA}/ml_staging/train_features"
dbutils.fs.rm(staging_path, True)

train_df.select(*MODEL_FEATURES, LABEL).write.parquet(staging_path)

train_pdf = pq.read_table(staging_path).to_pandas(strings_to_categorical=True)
print(f"Loaded {len(train_pdf):,} rows, {train_pdf[LABEL].sum():,} fraud "
      f"({train_pdf[LABEL].mean() * 100:.2f}%)")

# COMMAND ----------

X = prepare_features(train_pdf, MODEL_FEATURES, CATEGORICAL_LEVELS)
y = train_pdf[LABEL].astype("int8")
del train_pdf

held_back = np.random.default_rng(42).random(len(X)) < EARLY_STOPPING_SHARE
X_fit, y_fit = X[~held_back], y[~held_back]
X_eval, y_eval = X[held_back], y[held_back]
print(f"Fit on {len(X_fit):,} rows, early-stopping check on {len(X_eval):,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3–4. Train, log to MLflow, register in Unity Catalog

# COMMAND ----------

history_version = spark.sql(f"DESCRIBE HISTORY {fq('transactions_history')} LIMIT 1").first().version

with mlflow.start_run(run_name="xgboost_fraud") as run:
    mlflow.log_params({
        **{k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"},
        "train_start": str(TRAIN_START),
        "train_end": str(TRAIN_END),
        "train_normal_sample_rate": TRAIN_NORMAL_SAMPLE_RATE,
        "early_stopping_share": EARLY_STOPPING_SHARE,
        "num_features": len(MODEL_FEATURES),
        "profile_as_of": str(profile_meta.profile_as_of_ts),
    })
    mlflow.log_input(
        mlflow.data.load_delta(table_name=fq("transactions_history"), version=history_version),
        context="training",
    )

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_fit, y_fit, eval_set=[(X_eval, y_eval)], verbose=50)

    eval_scores = model.predict_proba(X_eval)[:, 1]
    mlflow.log_metrics({
        "best_iteration": model.best_iteration,
        "early_stopping_pr_auc": average_precision_score(y_eval, eval_scores),
        "early_stopping_roc_auc": roc_auc_score(y_eval, eval_scores),
    })

    importance = (
        pd.Series(model.get_booster().get_score(importance_type="gain"), name="gain")
        .rename_axis("feature").reset_index()
        .sort_values("gain", ascending=False)
    )
    mlflow.log_table(importance, "feature_importance.json")

    fig, ax = plt.subplots(figsize=(8, 10))
    importance.head(20).iloc[::-1].plot.barh(x="feature", y="gain", ax=ax, legend=False)
    ax.set_title("Top 20 features by gain")
    fig.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")

    fe.log_model(
        model=FraudModel(model, MODEL_FEATURES, CATEGORICAL_LEVELS),
        artifact_path="fraud_model",
        flavor=mlflow.pyfunc,
        training_set=training_set,
        registered_model_name=MODEL_NAME,
        extra_pip_requirements=[f"xgboost=={xgb.__version__}"],
    )

print(f"MLflow run: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

latest = max(MlflowClient().search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version))
print(f"Registered {MODEL_NAME} version {latest.version}")
print(f"Trees used: {model.best_iteration + 1} of {XGB_PARAMS['n_estimators']}")
print(f"Early-stopping check PR-AUC: {average_precision_score(y_eval, eval_scores):.4f}")
print(f"Early-stopping check ROC-AUC: {roc_auc_score(y_eval, eval_scores):.4f}")

# COMMAND ----------

# Which features the model relies on most.
display(importance.head(20))

# COMMAND ----------

display(fig)
