# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Capstone Project: 3 Production Deployments
# MAGIC 
# MAGIC The goal of this project is to deploy a trained machine learning model into production using all three deployment paradigms: batch, streaming, and REST.  An optional exercise entails creating a monitoring and alerting infrastructure.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this capstone you:<br>
# MAGIC  - Apply a model trained in `sklearn` across a Spark DataFrame
# MAGIC  - Perform predictions on an incoming stream of data
# MAGIC  - Deploy a rest endpoint
# MAGIC  - _Optional:_ Create a monitoring and alerting solution
# MAGIC  
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> **Please be sure to delete any infrastructure you build after the course so you don't incur unexpected expenses.**

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md Define a name for the stream to be used later in this lesson.

# COMMAND ----------

myStreamName = "capstone_pi"

# COMMAND ----------

# MAGIC %md Import the AirBnB dataset.  Create the following objects:<br><br>
# MAGIC 
# MAGIC * `pandasDF`: a Pandas DataFrame of all the data
# MAGIC * `pandasX`: a Pandas DataFrame of the `X` values
# MAGIC * `pandasy`: a Pandas DataFrame of the `y` values
# MAGIC * `sparkDF`: a Spark DataFrame of all the data

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

pandasDF = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")
pandasX = pandasDF.drop(["price"], axis=1)
pandasy = pandasDF["price"]

sparkDF = spark.createDataFrame(pandasDF)
display(sparkDF)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Train an AdaBoost regressor.  AdaBoost is meta-estimator that works by fitting one regressor to a dataset and then fits many additional copies of that same regressor to the dataset but with different weights for different errors.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Knowing how this algorithm works is not necessary for this capstone.  To deploy a model, we just need to know its inputs and outputs.  To read more about AdaBoost, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor" target="_blank">see the `sklearn` documentation.</a>

# COMMAND ----------

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error

seed = 42

rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=seed)
ada = AdaBoostRegressor(base_estimator=rf, n_estimators=400)
ada.fit(pandasX, pandasy)

predictions = pandasX.copy()
predictions["prediction"] = ada.predict(pandasX)

mse = mean_squared_error(pandasy, predictions["prediction"]) # This is on the same data the model was trained

# COMMAND ----------

# MAGIC %md ### Apply a model trained in `sklearn` across a Spark DataFrame
# MAGIC 
# MAGIC Perform the following steps to apply the AdaBoost model to a Spark DataFrame:<br><br>
# MAGIC 
# MAGIC 1. Log the model `ada` and evaluation metric `mse` using `mlflow`
# MAGIC 2. Create a Spark UDF from this logged model
# MAGIC 3. Apply the UDF to `sparkDF` (you may have to drop columns)

# COMMAND ----------

# TODO
# Log the model `ada` and evaluation metric `mse` using `mlflow`

# COMMAND ----------

# TODO
# Create a Spark UDF from this logged model

# COMMAND ----------

# TODO
Apply the UDF to `sparkDF` (you may have to drop columns)

# COMMAND ----------

# MAGIC %md ### Perform predictions on an incoming stream of data
# MAGIC 
# MAGIC Perform the following steps to apply the AdaBoost model to a stream of incoming data:<br><br>
# MAGIC 
# MAGIC 1. Run the logic defined for you to create the schema and stream
# MAGIC 2. Apply the UDF to `sparkDF` defined above to the stream
# MAGIC 3. Write the results to a delta table

# COMMAND ----------

from pyspark.sql.types import DoubleType, IntegerType, StructType

schema = (StructType()
  .add("host_total_listings_count", DoubleType())
  .add("neighbourhood_cleansed", IntegerType())
  .add("zipcode", IntegerType())
  .add("latitude", DoubleType())
  .add("longitude", DoubleType())
  .add("property_type", IntegerType())
  .add("room_type", IntegerType())
  .add("accommodates", DoubleType())
  .add("bathrooms", DoubleType())
  .add("bedrooms", DoubleType())
  .add("beds", DoubleType())
  .add("bed_type", IntegerType())
  .add("minimum_nights", DoubleType())
  .add("number_of_reviews", DoubleType())
  .add("review_scores_rating", DoubleType())
  .add("review_scores_accuracy", DoubleType())
  .add("review_scores_cleanliness", DoubleType())
  .add("review_scores_checkin", DoubleType())
  .add("review_scores_communication", DoubleType())
  .add("review_scores_location", DoubleType())
  .add("review_scores_value", DoubleType())
  .add("price", DoubleType())
)

spark.conf.set("spark.sql.shuffle.partitions", "8")
streamingData = (spark
  .readStream
  .schema(schema)
  .option("maxFilesPerTrigger", 1)
  .parquet("/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
)

# COMMAND ----------

# TODO
Apply the UDF to sparkDF defined above to the stream

predictionsDF = FILL_IN

Preview our stream - you must stop the stream before resuming
display(predictionsDF, streamName = myStreamName)

# COMMAND ----------

# Wait until the stream is ready for processing
from time import sleep

sleep(15)

# COMMAND ----------

# Stop the stream once you are done previewing it
# and before moving on to the next step...
for stream in spark.streams.active:
  print(f"Stopping {stream.name}")
  stream.stop() # Stop the stream

# COMMAND ----------

# Delete any files that might exist from previous runs
dbutils.fs.rm(f"{working_dir}/ml-deployment-capstone", True)

# Write the results to a delta table location identified by:
writePath = f"{working_dir}/ml-deployment-capstone/predictions"

# Make sure to specify the following checkpoint-location as identified by:
checkpointLocation = f"{working_dir}/ml-deployment-capstone/stream.checkpoint"

# COMMAND ----------

# TODO
(predictionsDF
  .FILL_IN    # Write the stream
  .FILL_IN    # Name the stream
  .FILL_IN    # Use the delta format
  .FILL_IN    # Specify a feature to partition on
  .FILL_IN    # Specify where to log metadata
  .FILL_IN    # Specify the output path
  .FILL_IN    # Append new records to the output path
  .FILL_IN    # Start the operation
)

# COMMAND ----------

# Wait until the stream is ready for processing
for stream in spark.streams.active:
  print(f"Stopping {stream.name}")
  stream.stop() # Stop the stream

# COMMAND ----------

# MAGIC %md Check to see if your files are there

# COMMAND ----------

from pyspark.sql.utils import AnalysisException

try:
  print(spark.read.format("delta").load(writePath).count())
except AnalysisException:
  print("Files not found.  This could be because the stream hasn't initialized.  Try again in a moment.")

# COMMAND ----------

# MAGIC %md Stop the streams.

# COMMAND ----------

[q.stop() for q in spark.streams.active]

# COMMAND ----------

# MAGIC %md ### Deploy a rest endpoint
# MAGIC 
# MAGIC Perform the following steps to deploy the AdaBoost model as a REST endpoint:<br><br>
# MAGIC 
# MAGIC 1. Create a new Azure ML Workspace
# MAGIC 2. Build the model image and wait for its creation
# MAGIC 3. Deploy the model to either ACI or AKS
# MAGIC 4. Query the REST endpoint

# COMMAND ----------

# TODO
# Create a new Azure ML Workspace
workspace_name = ""
workspace_location = ""
resource_group = ""
subscription_id = ""

# COMMAND ----------

# TODO
Build the model image and wait for its creation


# COMMAND ----------

# TODO
Deploy the model to either ACI or AKS

# COMMAND ----------

# TODO
Query the REST endpoint

# COMMAND ----------

# MAGIC %md ### _Optional:_ Create a monitoring and alerting solution

# COMMAND ----------

# TODO

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> **Please be sure to delete any infrastructure you build after the course so you don't incur unexpected expenses.**

# COMMAND ----------

# MAGIC %md 
# MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> All done!</h2>
# MAGIC 
# MAGIC Thank you for your participation!

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
