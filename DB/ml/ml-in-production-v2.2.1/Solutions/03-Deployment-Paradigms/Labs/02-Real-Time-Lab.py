# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab: Deploying a Real-time Model with MLflow Model Serving
# MAGIC MLflow Model Serving offers a fast way of serving pre-calculated predictions or creating predictions in real time. In this lab, you'll deploy a model using MLflow Model Serving.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Enable MLflow Model Serving for your registered model
# MAGIC  - Compute predictions in real time for your registered model via a REST API request

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC To start this off, we will need to load the data, build a model, and register that model.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> We're building a random forest model to predict Airbnb listing prices.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from numpy import mean
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from pyspark.sql.functions import monotonically_increasing_id

# Load data
df = (spark
    .read
    .parquet("/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
    .select("price", "bathrooms", "bedrooms", "number_of_reviews", "neighbourhood_cleansed")
    .withColumn("listing_id", monotonically_increasing_id()))

# Split into modeling and inference sets
# We'll use the inference dataset later to emulate unseen data
modeling_df, inference_df = df.randomSplit([0.5, 0.5], seed=42)

# Start run
with mlflow.start_run(run_name="Random Forest Model") as run:
    
    # Split data
    modeling_pdf = modeling_df.toPandas()
    X_train, X_test, y_train, y_test = train_test_split(
        modeling_pdf.drop(["price", "neighbourhood_cleansed", "listing_id"], axis=1), 
        modeling_pdf[["price"]].values.ravel(), 
        random_state=42
    )
    
    # Train model
    regressor = RandomForestRegressor(n_estimators=10, max_depth=5)
    regressor.fit(X_train, y_train)
    
    # Evaluate model
    train_predictions = regressor.predict(X_train)
    train_rmse = mean_squared_error(train_predictions, y_train, squared = False)
    
    # Log params and metric
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("n_estimators", 10)
    mlflow.log_metric("train_rmse", train_rmse)
    
    # Log model
    mlflow.sklearn.log_model(regressor, "model")
    
# Register model
model_name = f"rfr_model_{clean_username}"
model_uri = f"{run.info.artifact_uri}/model"
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
model_version = model_details.version

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will transition to model to staging.

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version,
  stage="Staging",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable MLflow Model Serving for the Registered Model
# MAGIC 
# MAGIC Your first task is to enable MLflow Model Serving for the model that was just registered.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Check out the [documentation](https://docs.databricks.com/applications/mlflow/model-serving.html#enable-and-disable-model-serving) for a demo of how to enable model serving via the GUI.

# COMMAND ----------

# MAGIC %md
# MAGIC -- ANSWER
# MAGIC 
# MAGIC Users can head to the registered model's page in the Model Registry to enable Model Serving.
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/mlflow/demo_model_register.png" width="600" height="20"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Real-time Predictions
# MAGIC 
# MAGIC Now that your model is registered, you will query the model with inputs.
# MAGIC 
# MAGIC To do this, you'll first need the appropriate token and instance.

# COMMAND ----------

# We need both a token for the API, which we can get from the notebook.
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
# With the token, we can create our authorization header for our subsequent REST calls
headers = {'Authorization': f'Bearer {token}'}

# Next we need an enpoint at which to execute our request which we can get from the Notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
# This ojbect comes from the Java CM - Convert the Java Map opject to a Python dictionary
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)
# Lastly, extract the databricks instance (domain name) from the dictionary
instance = tags['browserHostName']

# COMMAND ----------

# MAGIC %md
# MAGIC Next, create a function that takes a single record as input and returns the predicted value from the endpoint.

# COMMAND ----------

# ANSWER
import pandas as pd
import requests

def score_model(dataset: pd.DataFrame, model_name: str, token: str, instance: str):
    url = f'https://{instance}/model/{model_name}/1/invocations'
    data_json = dataset.to_dict(orient='split')
    
    response = requests.request(method='POST', headers=headers, url=url, json=data_json)
    
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC Now, use that function to score a single row of a Pandas DataFrame.

# COMMAND ----------

# ANSWER
single_row_df = pd.DataFrame([[2, 2, 150]], columns=["bathrooms", "bedrooms", "number_of_reviews"])
score_model(single_row_df, model_name, token, instance)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, create a function that will score `n` total records from a Pandas DataFrame.

# COMMAND ----------

# ANSWER
inference_pdf = inference_df.toPandas().loc[:, ["bathrooms", "bedrooms", "number_of_reviews"]]

def score_n_records(n, df=df):
    sample_df = df.iloc[:n,:]
    return score_model(sample_df, model_name, token, instance)

score_n_records(100, inference_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Lesson<br>
# MAGIC 
# MAGIC Start the next lesson, [Streaming]($../03-Streaming )

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
