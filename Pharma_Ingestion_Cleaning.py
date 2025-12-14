# Databricks notebook source
# ============================================
# ALGERIAN PHARMA DATA - INGESTION & CLEANING
# ============================================

import requests
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

print("📦 Starting data ingestion from DZ-Pharma-Data (Algerian medications)...\n")

# Correct URL - data is in the /data/ folder
url_meds = "https://raw.githubusercontent.com/fennecinspace/DZ-Pharma-Data/master/data/meds.json"
url_labs = "https://raw.githubusercontent.com/fennecinspace/DZ-Pharma-Data/master/data/labs.json"

# Fetch medications data
try:
    print(f"Fetching: {url_meds}")
    response = requests.get(url_meds, timeout=30)
    response.raise_for_status()
    meds_json = response.json()
    print(f"✅ Medications data loaded!\n")
except Exception as e:
    print(f"❌ Error: {e}")
    raise

# Flatten the nested structure (organized by first letter: A, B, C, ...)
all_medications = []

for letter, meds_list in meds_json.items():
    if isinstance(meds_list, list):
        for med in meds_list:
            med['first_letter'] = letter
            all_medications.append(med)
        print(f"✅ Letter {letter}: {len(meds_list)} medications")

print(f"\n📊 Total Algerian medications extracted: {len(all_medications)}")

# Create Spark DataFrame
df_raw = spark.createDataFrame(all_medications)

print("\n🔍 Raw data schema:")
df_raw.printSchema()


# COMMAND ----------

print("\n📋 Sample medications (first 5 rows):")
display(df_raw.limit(5))

print(f"\n✅ SUCCESS: {df_raw.count()} Algerian medications loaded!")


# COMMAND ----------

# MAGIC %md
# MAGIC ## # **Extract Nested Lab Fields**

# COMMAND ----------

from pyspark.sql.functions import col, when, regexp_extract, count, avg, min, max

# Flatten lab and class nested maps
df = df_raw \
    .withColumn("lab_name", col("lab.name")) \
    .withColumn("lab_address", col("lab.address")) \
    .withColumn("lab_tel", col("lab.tel")) \
    .withColumn("lab_web", col("lab.web")) \
    .withColumn("therapeutic_class", col("class.therapeutic")) \
    .withColumn("pharmacological_class", col("class.pharmacological"))

display(df.select("name", "lab_name", "therapeutic_class", "reference_rate").limit(20))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract numeric value from price string , using cast or try cast (khir)
# MAGIC just when not empty 

# COMMAND ----------

from pyspark.sql.functions import expr

# Use SQL try_cast via expr (works in all Databricks versions)
df = df.withColumn(
    "price",
    expr("try_cast(regexp_extract(reference_rate, '(\\\\d+)', 1) as int)")
)

# Check price distribution
df.filter(col("price").isNotNull()).select(min("price"), avg("price"), max("price")).show()
display(df.select("name", "reference_rate", "price").orderBy(col("price").desc_nulls_last()).limit(10))


# COMMAND ----------

# Count different price scenarios
total = df.count()
null_price = df.filter(col("price").isNull()).count()
zero_price = df.filter(col("price") == 0).count()
valid_price = df.filter((col("price").isNotNull()) & (col("price") > 0)).count()

print(f"Price Analysis:")
print(f"   Total medications: {total}")
print(f"   NULL price: {null_price}")
print(f"   Zero price: {zero_price}")
print(f"   Valid price (>0): {valid_price}")
print(f"   Will drop: {null_price + zero_price} medications")
print(f"   Will keep: {valid_price} medications ({(valid_price/total)*100:.1f}%)")


# COMMAND ----------

# MAGIC %md
# MAGIC ## **Delete non valid prices **

# COMMAND ----------

# Remove medications with null or zero prices
df = df.filter((col("price").isNotNull()) & (col("price") > 0))

print(f"✅ After cleaning: {df.count()} medications with valid prices")
df.select(min("price"), avg("price"), max("price")).show()
display(df.select("name", "price").orderBy("price").limit(40))


# COMMAND ----------

# Check if we have refundable data
print("📊 Refundable Column Analysis:\n")

# Check column exists
print("Columns in dataset:", df.columns)
print()

# Count refundable values
total = df.count()
refundable_true = df.filter(col("refundable") == True).count()
refundable_false = df.filter(col("refundable") == False).count()
refundable_null = df.filter(col("refundable").isNull()).count()

print(f"Total df: {total}")
print(f"Refundable (True): {refundable_true} ({refundable_true/total*100:.1f}%)")
print(f"Not Refundable (False): {refundable_false} ({refundable_false/total*100:.1f}%)")
print(f"Unknown (NULL): {refundable_null} ({refundable_null/total*100:.1f}%)")

# Show distribution by therapeutic class
print("\n📋 Refundable by Therapeutic Class:")
df.groupBy("therapeutic_class", "refundable").count().orderBy("count", ascending=False).show(20)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleaning & Transforming data also for the ML prediction task 

# COMMAND ----------

# NULL refundable means not covered (Verefied info from pharma experts in Algeria)
ml_data = df.withColumn(
    "refundable_clean",
    when(col("refundable").isNull(), False).otherwise(col("refundable"))
)

display(ml_data.select("name", "refundable", "refundable_clean").limit(10))

# COMMAND ----------

# Check class balance
total = ml_data.count()
refund_yes = ml_data.filter(col("refundable_clean") == True).count()
refund_no = ml_data.filter(col("refundable_clean") == False).count()

print(f"📊 Refundable Distribution:")
print(f"   YES (covered): {refund_yes} ({refund_yes/total*100:.1f}%)")
print(f"   NO (not covered): {refund_no} ({refund_no/total*100:.1f}%)")
print(f"   Ratio: {refund_yes/refund_no:.1f}:1")

# COMMAND ----------

# Check patterns by therapeutic class
ml_data.groupBy("therapeutic_class", "refundable_clean") \
       .count() \
       .orderBy("count", ascending=False) \
       .show(20)


# COMMAND ----------



print(f"\nTotal records: {df.count()}")

print("Current table:")
display(df.limit(5))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Good to gooo , save the delta tqble to catalog

# COMMAND ----------

# Save df (medications table) to catalog
# mode("overwrite") = replace if exists
# saveAsTable() = permanent storage in Databricks catalog
df.write.format("delta").mode("overwrite").saveAsTable("medications")



# COMMAND ----------

# Read from catalog to verify
saved_table = spark.table("medications")

print(f"✅ Table loaded from catalog")
print(f"   Total records: {saved_table.count()}")
print(f"   Columns: {saved_table.columns}")

display(saved_table.limit(10))
