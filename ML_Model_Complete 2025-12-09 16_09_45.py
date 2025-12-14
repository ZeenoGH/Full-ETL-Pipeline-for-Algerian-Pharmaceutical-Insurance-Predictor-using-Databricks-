# Databricks notebook source
# MAGIC %md
# MAGIC # Algerian Pharmaceutical Market - ML Model
# MAGIC ## Insurance Reimbursement Prediction

# COMMAND ----------

# Cell 1: Load data and check structure
df = spark.table("medications")

print(f"Total records: {df.count()}")
print(f"\nColumns: {df.columns}")

# Check refundable column
df.groupBy("refundable").count().show()

# COMMAND ----------

# Cell 2: Handle NULL values (convert to False)
from pyspark.sql.functions import col, when

df = df.withColumn(
    "refundable_label",
    when(col("refundable").isNull(), False)
    .otherwise(col("refundable"))
)

print("After converting NULL to False:")
df.groupBy("refundable_label").count().show()

total = df.count()
covered = df.filter(col("refundable_label") == True).count()
not_covered = df.filter(col("refundable_label") == False).count()

print(f"\nCovered: {covered} ({covered/total*100:.1f}%)")
print(f"Not Covered: {not_covered} ({not_covered/total*100:.1f}%)")
print(f"Ratio: {covered/not_covered:.1f}:1")

# COMMAND ----------

# Cell 3: Select and clean features for ML
ml_data = df.select(
    "price",
    "form",
    "therapeutic_class",
    "pharmacological_class",
    "lab_name",
    "refundable_label"
).filter(
    (col("price").isNotNull()) &
    (col("form").isNotNull()) &
    (col("therapeutic_class").isNotNull()) &
    (col("pharmacological_class").isNotNull()) &
    (col("lab_name").isNotNull())
)

print(f"✅ ML-ready data: {ml_data.count()} records")
display(ml_data.limit(10))

# COMMAND ----------

# Cell 4: Encode categorical features (fixed initialization)
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Convert text to numbers - use proper initialization
indexer_form = StringIndexer(inputCol="form", outputCol="form_idx")
indexer_form.setHandleInvalid("keep")

indexer_therapeutic = StringIndexer(inputCol="therapeutic_class", outputCol="therapeutic_idx")
indexer_therapeutic.setHandleInvalid("keep")

indexer_pharmacological = StringIndexer(inputCol="pharmacological_class", outputCol="pharmacological_idx")
indexer_pharmacological.setHandleInvalid("keep")

indexer_lab = StringIndexer(inputCol="lab_name", outputCol="lab_idx")
indexer_lab.setHandleInvalid("keep")

indexer_label = StringIndexer(inputCol="refundable_label", outputCol="label")

# Apply all transformations
df_indexed = indexer_form.fit(ml_data).transform(ml_data)
df_indexed = indexer_therapeutic.fit(df_indexed).transform(df_indexed)
df_indexed = indexer_pharmacological.fit(df_indexed).transform(df_indexed)
df_indexed = indexer_lab.fit(df_indexed).transform(df_indexed)
df_indexed = indexer_label.fit(df_indexed).transform(df_indexed)


display(df_indexed.select("form", "form_idx", "therapeutic_class", "therapeutic_idx", "refundable_label", "label").limit(5))


# COMMAND ----------

# Cell 4: Encode categorical features (SQL-based workaround)
from pyspark.sql.functions import col, dense_rank
from pyspark.sql.window import Window

# Create indices using dense_rank (SQL-based encoding)
ml_data_encoded = ml_data

# Encode form
window_form = Window.orderBy("form")
ml_data_encoded = ml_data_encoded.withColumn("form_idx", dense_rank().over(window_form) - 1)

# Encode therapeutic_class
window_therapeutic = Window.orderBy("therapeutic_class")
ml_data_encoded = ml_data_encoded.withColumn("therapeutic_idx", dense_rank().over(window_therapeutic) - 1)

# Encode pharmacological_class
window_pharma = Window.orderBy("pharmacological_class")
ml_data_encoded = ml_data_encoded.withColumn("pharmacological_idx", dense_rank().over(window_pharma) - 1)

# Encode lab_name
window_lab = Window.orderBy("lab_name")
ml_data_encoded = ml_data_encoded.withColumn("lab_idx", dense_rank().over(window_lab) - 1)

# Encode label (refundable_label: True=0, False=1)
ml_data_encoded = ml_data_encoded.withColumn(
    "label", 
    when(col("refundable_label") == True, 0.0).otherwise(1.0)
)

df_indexed = ml_data_encoded

print("✅ Categorical features encoded using SQL")
display(df_indexed.select("form", "form_idx", "therapeutic_class", "therapeutic_idx", "refundable_label", "label").limit(5))


# COMMAND ----------

# Cell 5: Create feature vector manually (workaround)
from pyspark.sql.functions import array, col

# Manually create feature array instead of VectorAssembler
ml_final = df_indexed.withColumn(
    "features",
    array(
        col("price").cast("double"),
        col("form_idx").cast("double"),
        col("therapeutic_idx").cast("double"),
        col("pharmacological_idx").cast("double"),
        col("lab_idx").cast("double")
    )
).select("features", "label")

print(f"✅ Feature vector created: {ml_final.count()} records")

# Split 80/20
train, test = ml_final.randomSplit([0.8, 0.2], seed=42)
print(f"\nTraining: {train.count()} | Test: {test.count()}")


# COMMAND ----------

# Cell 6: Check class imbalance
train_covered = train.filter(col("label") == 0.0).count()
train_not_covered = train.filter(col("label") == 1.0).count()

print(f"Training set distribution:")
print(f"  Covered (label=0): {train_covered}")
print(f"  Not covered (label=1): {train_not_covered}")
print(f"  Imbalance ratio: {train_covered/train_not_covered:.1f}:1")

# COMMAND ----------

# MAGIC %md
# MAGIC ### PS: The dataset is not balanced, but i am avoiding to do undersampling, so i will use class weights in the model 

# COMMAND ----------

# Train model using sklearn (PySpark MLlib blocked in Community Edition)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

print("Converting Spark DataFrame to sklearn format...")

# Convert to pandas
train_pd = train.toPandas()
test_pd = test.toPandas()

# Extract features (arrays) and labels
X_train = np.array([list(x) for x in train_pd['features'].values])
y_train = train_pd['label'].values

X_test = np.array([list(x) for x in test_pd['features'].values])
y_test = test_pd['label'].values

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

# Train Random Forest with class weights
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight={0.0: 1, 1.0: 6.1},  # Weight minority class
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
print("Model trained successfully!")


# COMMAND ----------

# Evaluate model performance
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("Model Performance:")
print(f"  Accuracy: {accuracy*100:.1f}%")
print(f"  AUC-ROC: {auc:.3f}")

# Show predictions breakdown
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Covered', 'Not Covered']))


# COMMAND ----------

# Feature importance analysis
feature_names = ["price", "form", "therapeutic_class", "pharmacological_class", "lab_name"]
importances = rf.feature_importances_

import pandas as pd
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("Feature Importance (what drives insurance coverage):")
print(importance_df.to_string(index=False))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Testing it 

# COMMAND ----------

# Test with real examples from test set
test_sample = test_pd.head(10)

print("Sample Predictions:\n")
for i, row in test_sample.iterrows():
    features = np.array(list(row['features'])).reshape(1, -1)
    prediction = rf.predict(features)[0]
    probability = rf.predict_proba(features)[0]
    actual = row['label']
    
    pred_label = "Covered" if prediction == 0.0 else "Not Covered"
    actual_label = "Covered" if actual == 0.0 else "Not Covered"
    confidence = max(probability) * 100
    
    match = "✓" if prediction == actual else "✗"
    
    print(f"{match} Predicted: {pred_label} ({confidence:.1f}% confidence) | Actual: {actual_label}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Then we save it 

# COMMAND ----------

# Save model to temporary storage
import pickle

model_path = "/tmp/insurance_model.pkl"

# Save model
with open(model_path, 'wb') as f:
    pickle.dump(rf, f)

print(f"✓ Model saved to: {model_path}")
print("\nModel Details:")
print(f"  Algorithm: Random Forest")
print(f"  Trees: 100")
print(f"  Accuracy: {accuracy*100:.1f}%")
print(f"  AUC: {auc:.3f}")
print(f"  Features: 5 (price, form, therapeutic_class, pharmacological_class, lab_name)")


# COMMAND ----------

# Verify model can be loaded
with open("/tmp/insurance_model.pkl", 'rb') as f:
    loaded_model = pickle.load(f)

print("✓ Model loaded and verified!")
print("\nModel is trained and ready to use in this session.")
print("Note: In production, deploy to Databricks Model Registry or MLflow")


# COMMAND ----------

# Prepare model for download
import pickle

# Save model locally in notebook
model_filename = "insurance_model.pkl"

with open(f"/tmp/{model_filename}", 'wb') as f:
    pickle.dump(rf, f)

# Also save feature encodings (important!)
encodings = {
    'feature_names': ["price", "form", "therapeutic_class", "pharmacological_class", "lab_name"],
    'model_type': 'RandomForestClassifier',
    'accuracy': accuracy,
    'auc': auc,
    'class_weights': {0.0: 1, 1.0: 6.1}
}

with open("/tmp/model_info.pkl", 'wb') as f:
    pickle.dump(encodings, f)

print(f"✓ Model saved to /tmp/{model_filename}")
print("✓ Model info saved to /tmp/model_info.pkl")
print("\nTo download:")
print("1. Click 'File' menu in Databricks")
print("2. Navigate to /tmp/")
print("3. Right-click file → Download")


# COMMAND ----------

# List files in /tmp
import os
files = os.listdir("/tmp")
print("Files in /tmp:")
for f in files:
    if "model" in f.lower():
        print(f"  ✓ {f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This model predicts whether a medication will be covered by insurance in Algeria based on:
# MAGIC - Price
# MAGIC - Medication form (tablet, injection, etc.)
# MAGIC - Therapeutic category (disease area)
# MAGIC - Pharmacological class (mechanism of action)
# MAGIC - Manufacturer
# MAGIC
# MAGIC The model handles class imbalance (85.9% vs 14.1%) using weighted Random Forest.