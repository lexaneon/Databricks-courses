# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Data Management
# MAGIC 
# MAGIC Production machine learning solutions start with reproducible data management. Strategies that we'll cover in this notebook include [Delta Table Versioning](https://docs.databricks.com/delta/versioning.html), working with the [Feature Store](https://docs.databricks.com/applications/machine-learning/feature-store.html), and detecting data changes by [hashing](https://en.wikipedia.org/wiki/MD5) our data.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Version tables with Delta
# MAGIC  - Programmatically log Feature Tables
# MAGIC  - Hash data to detect changes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Management and Reproducibility
# MAGIC 
# MAGIC Managing the machine learning lifecycle means...<br><br>
# MAGIC 
# MAGIC * Reproducibility of data
# MAGIC * Reproducibility of code
# MAGIC * Reproducibility of models
# MAGIC * Automated integration with production systems
# MAGIC 
# MAGIC **We'll begin with data management,** which can be accomplished in a number of ways including:<br><br>
# MAGIC 
# MAGIC - Saving a snapshot of your data
# MAGIC - Table versioning and time travel using Delta
# MAGIC - Using a feature table
# MAGIC - Saving a hash of your data to detect changes

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC 
# MAGIC For each lesson to execute correctly, please make sure to run the **`Classroom-Setup`** cell at the start of each lesson.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md Let's load in our data and generate a unique ID for each listing.

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id 

airbnb_df = (spark.read
  .format("delta")
  .load("/mnt/training/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/")
  .withColumn("index", monotonically_increasing_id())
)

display(airbnb_df)

# COMMAND ----------

# MAGIC %md ## Versioning with Delta Tables
# MAGIC 
# MAGIC Let's start by writing to a new Delta Table.

# COMMAND ----------

delta_path = working_dir.replace("/dbfs", "dbfs:") + "/delta-example"
dbutils.fs.rm(delta_path, recurse=True)

airbnb_df.write.format("delta").save(delta_path)

# COMMAND ----------

# MAGIC %md Now let's read our Delta Table and modify it, dropping the `cancellation_policy` and `instant_bookable` columns.

# COMMAND ----------

delta_df = (spark.read
  .format("delta")
  .load(delta_path)
  .drop("cancellation_policy", "instant_bookable")
)

display(delta_df)

# COMMAND ----------

# MAGIC %md Now we can `overwrite` our Delta Table using the `mode` parameter.

# COMMAND ----------

delta_df.write.format("delta").mode("overwrite").save(delta_path)

# COMMAND ----------

# MAGIC %md Whoops! We actually wanted to keep the `cancellation_policy` column. Luckily we can use data versioning to return to an older version of this table. 
# MAGIC 
# MAGIC Start by using the `DESCRIBE HISTORY` SQL command.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY delta.`{delta_path}`"))

# COMMAND ----------

# MAGIC %md As we can see in the `operationParameters` column in version 1, we overwrote the table. We now need to travel back in time to load in version 0 to get all the original columns, then we can delete just the `instant_bookable` column.

# COMMAND ----------

delta_df = spark.read.format("delta").option("versionAsOf", 0).load(delta_path)

display(delta_df)

# COMMAND ----------

# MAGIC %md You can also query based upon timestamp.  **Note that the ability to query an older snapshot of a table (time travel) is lost after running [a VACUUM command.](https://docs.databricks.com/delta/delta-batch.html#deltatimetravel)**

# COMMAND ----------

timestamp = spark.sql(f"DESCRIBE HISTORY delta.`{delta_path}`").first().timestamp

display(spark.read
  .format("delta")
  .option("timestampAsOf", timestamp)
  .load(delta_path)
)

# COMMAND ----------

# MAGIC %md Now we can drop `instant_bookable` and overwrite the table.

# COMMAND ----------

delta_df.drop("instant_bookable").write.format("delta").mode("overwrite").save(delta_path)

# COMMAND ----------

# MAGIC %md Version 2 is our latest and most accurate table version.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY delta.`{delta_path}`"))

# COMMAND ----------

# MAGIC %md ## Feature Store
# MAGIC 
# MAGIC A [feature store](https://docs.databricks.com/applications/machine-learning/feature-store.html#databricks-feature-store) is a **centralized repository of features.** It enables feature **sharing and discovery across** your organization and also ensures that the same feature computation code is used for model training and inference.

# COMMAND ----------

# MAGIC %md %md Create a new database and unique table name (in case you re-run the notebook multiple times - currently no support for deleting feature tables or features).

# COMMAND ----------

import uuid
import re 

spark.sql(f"CREATE DATABASE IF NOT EXISTS {clean_username}")
table_name = f"{clean_username}.airbnb_" + str(uuid.uuid4())[:6]

print(table_name)

# COMMAND ----------

# MAGIC %md Let's start creating a [Feature Store Client](https://docs.databricks.com/applications/machine-learning/feature-store.html#create-a-feature-table-in-databricks-feature-store) so we can populate our feature store.

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

help(fs.create_feature_table)

# COMMAND ----------

# MAGIC %md #### Create Feature Table
# MAGIC 
# MAGIC Next, we can create the Feature Table using the `create_feature_table` method.
# MAGIC 
# MAGIC This method takes a few parameters as inputs:
# MAGIC * `name`- A feature table name of the form ``<database_name>.<table_name>``
# MAGIC * `keys`- The primary key(s). If multiple columns are required, specify a list of column names.
# MAGIC * `features_df`- Data to insert into this feature table.  The schema of `features_df` will be used as the feature table schema.
# MAGIC * `schema`- Feature table schema. Note that either `schema` or `features_df` must be provided.
# MAGIC * `description`- Description of the feature table
# MAGIC * `partition_columns`- Column(s) used to partition the feature table.

# COMMAND ----------

fs.create_feature_table(
    name=table_name,
    keys=["index"],
    features_df=airbnb_df,
    partition_columns=["neighbourhood_cleansed"],
    description="Original Airbnb data"
)

# COMMAND ----------

# MAGIC %md Now let's explore the UI and see how it tracks the tables that we created. Navigate to the UI by first ensuring that you are in the Machine Learning workspace, and then clicking on the Feature Store icon on the bottom-left of the navigation bar.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/FS_Nav.png" alt="step12" width="150"/>

# COMMAND ----------

# MAGIC %md Take a look at the feature table that you created.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/FS_Features.png" alt="step12" width="1000"/>

# COMMAND ----------

# MAGIC %md Here we can see more detail on the table. 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/FS_UI_Created.png" alt="step12" width="800"/>

# COMMAND ----------

# MAGIC %md Now let's try updating our feature table. We can begin to refine our table by filtering out some rows which don't match our specifications. We'll start by looking at some of the `bed_type` values.

# COMMAND ----------

display(airbnb_df.groupBy("bed_type").count())

# COMMAND ----------

# MAGIC %md Since we only want `real beds`, we can drop the other records from the DataFrame.

# COMMAND ----------

airbnb_df_real_beds = airbnb_df.filter("bed_type = 'Real Bed'")

display(airbnb_df_real_beds)

# COMMAND ----------

# MAGIC %md #### Merge Features
# MAGIC 
# MAGIC Now that we have filtered some of our data, we can `merge` the existing feature table in the Feature Store with the new table. Merging updates the feature table schema, and adds new feature values based on the primary key.

# COMMAND ----------

fs.write_table(
  name=table_name,
  df=airbnb_df_real_beds,
  mode="merge"
)

# COMMAND ----------

# MAGIC %md Lastly, we'll condense some of the review columns, we'll do this by finding the average review score for each listing.

# COMMAND ----------

from pyspark.sql.functions import lit, expr

reviewColumns = ["review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", 
                 "review_scores_communication", "review_scores_location", "review_scores_value"]

airbnb_df_short_reviews = (airbnb_df_real_beds
  .withColumn("average_review_score", expr("+".join(reviewColumns)) / lit(len(reviewColumns)))
  .drop(*reviewColumns)
)

display(airbnb_df_short_reviews)

# COMMAND ----------

# MAGIC %md #### Overwrite Features
# MAGIC 
# MAGIC Here we `overwrite` instead of `merge` since we have deleted some feature columns and want them to be removed from the feature table entirely.

# COMMAND ----------

fs.write_table(
  name=table_name,
  df=airbnb_df_short_reviews,
  mode="overwrite"
)

# COMMAND ----------

# MAGIC %md By navigating back to the UI, we can again see that the modified date has changed, and that a new column has been added to the feature list. However, note that the columns that we deleted are also still present, in the next command we can see how the data stored has changed, however the columns present in the original table will remain in the feature table.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/FS_New_Feature.png" alt="step12" width="800"/>

# COMMAND ----------

# MAGIC %md Now we can read in the feature data from the Feature Store into a new DataFrame. Optionally, we can use Delta Time Travel to read from a specific timestamp of the feature table. Note that the values of the deleted columns have been replaced by `null`s.

# COMMAND ----------

# Displays most recent table
feature_df = fs.read_table(
  name=table_name
)

display(feature_df)

# COMMAND ----------

# MAGIC %md If you have a use case to join features for real-time prediction, you can publish your features to an [online store](https://docs.databricks.com/applications/machine-learning/feature-store.html#publish-features-to-an-online-feature-store).
# MAGIC 
# MAGIC And finally, we can perform Access Control using built-in features in the Feature Store UI.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/FS_Access_Control.png" alt="step12" width="700"/>

# COMMAND ----------

# MAGIC %md ## MD5 Hash
# MAGIC 
# MAGIC The last data management technique we'll be looking at is the MD5 hash which allows you to confirm that the data has not been modified or corrupted, though this does not give you a full diff if your data does not match.

# COMMAND ----------

import hashlib
import pandas as pd

pd_df = airbnb_df.toPandas()

m1 = hashlib.md5(pd.util.hash_pandas_object(pd_df, index=True).values).hexdigest()
m1

# COMMAND ----------

# MAGIC %md Now that we have the `hexdigest` of the original file, any changes that are made to the underlying DataFrame will result in a different `hexdigest`.

# COMMAND ----------

pd_no_cancellation_df = pd_df.drop("cancellation_policy", axis=1)

m2 = hashlib.md5(pd.util.hash_pandas_object(pd_no_cancellation_df, index=True).values).hexdigest()
m2

# COMMAND ----------

# MAGIC %md As we can see here, the new hash is different stemming from the dropped column.

# COMMAND ----------

try:
  assert m1 == m2, "The datasets do not have the same hash"
  raise Exception("Expected failure")

except AssertionError:
  print("Failed as expected")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review
# MAGIC **Question:** Why do we care about Data Management?
# MAGIC **Answer:** Data Management is an oftentimes overlooked aspect of end-to-end reproducbility.
# MAGIC 
# MAGIC **Question:** How do we version data with Delta Tables?
# MAGIC **Answer:** Delta Tables are automatically versioned everytime a new data is written. Accessing a previous version of the table is as simple as using `display(spark.sql(f"DESCRIBE HISTORY delta.{delta_path}"))` to find the version to revert to and loading it in.  You can also revert to previous version using timestamps.
# MAGIC 
# MAGIC **Question:** What challenges does the Feature Store help solve?
# MAGIC **Answer:** A key issue many ML pipelines struggle with is feature reproducibility and data sharing. The Feature Store lets different users across the same organization utilize the same feature computation code.
# MAGIC 
# MAGIC **Question:** What does hashing a dataset help me do?
# MAGIC **Answer:** It can help confirm whether a dataset is or is not the same as another.  This is helpful in data reproducibility.  It cannot, however, tell you the full diff between two datasets.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Steps
# MAGIC 
# MAGIC Start the next lesson, [Experiment Tracking]($./02-Experiment-Tracking)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I learn more about Delta Tables?
# MAGIC **A:** Check out this <a href="https://databricks.com/session_na21/intro-to-delta-lake" target="_blank"> talk </a> by Himanshu Raj at the Data+AI Summit 2021.
# MAGIC 
# MAGIC **Q:** Where can I learn more about the Feature Store?
# MAGIC **A:** The <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html" target="_blank"> documentation </a> provides an in-depth look at what the Feature Store can do for your pipeline.
# MAGIC 
# MAGIC **Q:** Where can I learn more about reproducibility and its importance?
# MAGIC **A:** This [blog post](https://databricks.com/blog/2021/04/26/reproduce-anything-machine-learning-meets-data-lakehouse.html) by Mary Grace Moesta and Srijith Rajamohan provides a starting point for creating reproducible data and models

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
