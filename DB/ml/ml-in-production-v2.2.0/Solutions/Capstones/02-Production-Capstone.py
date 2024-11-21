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

# ANSWER
import mlflow.sklearn

with mlflow.start_run(run_name="Final RF Model") as run: 
  mlflow.sklearn.log_model(ada, "adaboost-model")
  mlflow.log_metric("Train data MSE", mse)
  
  run_info = run.info

# COMMAND ----------

# ANSWER
predict = mlflow.pyfunc.spark_udf(spark, run_info.artifact_uri + "/adaboost-model")

# COMMAND ----------

# ANSWER
predictionDF = (sparkDF
  .withColumn("prediction", predict(*sparkDF.drop("price").columns))
)

display(predictionDF)

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

# ANSWER
predictionsDF = streamingData.withColumn("prediction", predict(*streamingData.drop("price").columns))

# Preview our stream - you must stop the stream before resuming
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

# ANSWER
(predictionsDF
  .writeStream                                      # Write the stream
  .queryName(myStreamName)                          # Name the stream
  .format("delta")                                  # Use the delta format
  .partitionBy("zipcode")                           # Specify a feature to partition on
  .option("checkpointLocation", checkpointLocation) # Specify where to log metadata
  .option("path", writePath)                        # Specify the output path
  .outputMode("append")                             # Append new records to the output path
  .start()                                          # Start the operation
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

# ANSWER
# import azureml
# from azureml.core import Workspace

# workspace_name = ""
# workspace_location = ""
# resource_group = ""
# subscription_id = ""

# try:
#   workspace = Workspace.create(
#     name = workspace_name,
#     location = workspace_location,
#     resource_group = resource_group,
#     subscription_id = subscription_id,
#     exist_ok=True
#   )
# except Exception as e:
#   print("Error: Input the subscription id in the Azure-Credentials Notebook. \n")
#   print(e)

# COMMAND ----------

# ANSWER
# import mlflow.azureml

# model_uri = "/dbfs/databricks/mlflow/{}/{}/artifacts/{}".format(run_info.experiment_id, run_info.run_uuid, "adaboost-model")

# # Build the image
# model_image, azure_model = mlflow.azureml.build_image(
#   model_uri=model_uri, 
#   workspace=workspace, 
#   model_name="adaboostmodel",
#   image_name="adaboostmodel",
#   description="Sklearn AdaBoost image for predicting airbnb housing prices",
#   synchronous=False
# )

# # Wait for the image creation
# model_image.wait_for_creation(show_output=True)

# COMMAND ----------

# ANSWER
# Deploy to ACI
# from azureml.core.webservice import AciWebservice, Webservice

# # Create the deployment 
# dev_webservice_name = "adaboost-model"
# dev_webservice_deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# dev_webservice = Webservice.deploy_from_image(
#   name=dev_webservice_name, 
#   image=model_image, 
#   deployment_config=dev_webservice_deployment_config, 
#   workspace=workspace
# )

# # Wait for the image deployment
# dev_webservice.wait_for_deployment(show_output=True)

# COMMAND ----------

# ANSWER
# import requests
# import json

# sample = pandasX.iloc[[0]]

# query_input = sample.to_json(orient='split')
# query_input = eval(query_input)
# query_input.pop('index', None)

# def query_endpoint(scoring_uri, inputs, service_key=None):
#   headers = {
#     "Content-Type": "application/json",
#   }
#   if service_key is not None:
#     headers["Authorization"] = "Bearer {service_key}".format(service_key=service_key)
    
#   print("Sending batch prediction request with inputs: {}".format(inputs))
#   response = requests.post(scoring_uri, data=json.dumps(inputs), headers=headers)
#   preds = json.loads(response.text)
#   print("Received response: {}".format(preds))
#   return preds

# query_endpoint(scoring_uri=dev_webservice.scoring_uri, inputs=query_input)

# COMMAND ----------

# MAGIC %md ### _Optional:_ Create a monitoring and alerting solution

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
