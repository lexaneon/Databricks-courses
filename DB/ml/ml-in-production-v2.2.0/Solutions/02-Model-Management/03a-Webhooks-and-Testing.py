# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # MLflow Webhooks & Testing
# MAGIC 
# MAGIC Webhooks trigger the execution of code (oftentimes tests) upon some event. This lesson explores how to employ webhooks to trigger automated tests against models in the model registry.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Explore the role of webhooks in ML pipelines
# MAGIC  - Create a job to test models in the model registry
# MAGIC  - Examine a job that imports a model and runs tests
# MAGIC  - Automate that job using MLflow webhooks

# COMMAND ----------

# MAGIC %md ## Automated Testing
# MAGIC 
# MAGIC The backbone of the continuous integration, continuous deployment (CI/CD) process is the automated building, testing, and deployment of code. A **webhook or trigger** causes the execution of code based upon some event.  This is commonly when new code is pushed to a code repository.  In the case of machine learning jobs, this could be the arrival of a new model in the model registry.
# MAGIC 
# MAGIC This lesson uses **MLflow webhooks** to trigger the execution of a job upon the arrival of a new model version with a given name in the model registry. The function of that job is to:<br><br>
# MAGIC 
# MAGIC - Import the new model version
# MAGIC - Test the schema of its inputs and outputs
# MAGIC - Pass example code through the model
# MAGIC 
# MAGIC This covers many of the desired tests for ML models.  However, throughput testing could also be performed using this paradigm. Also, the model could also be promoted to the production stage in an automated fashion.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md ## Creating a Job
# MAGIC 
# MAGIC The following steps will create a Databricks job using another notebook in this directory: `03b-Webhooks-Job-Demo`
# MAGIC 
# MAGIC **Note:** Ensure that you are an admin on this workspace and that you're not using Community Edition (which has jobs disabled)

# COMMAND ----------

# MAGIC %md Create a user access token using the following steps:<br><br>
# MAGIC 
# MAGIC 1. Click the user profile icon User Profile in the upper right corner of your Databricks workspace
# MAGIC 1. Click User Settings
# MAGIC 1. Go to the Access Tokens tab
# MAGIC 1. Click the Generate New Token button
# MAGIC 1. Optionally enter a description (comment) and expiration period
# MAGIC 1. Click the Generate button
# MAGIC 1. Copy the generated token **and paste it in the following cell**
# MAGIC 
# MAGIC **Note:** You can find details [about access tokens here](https://docs.databricks.com/dev-tools/api/latest/authentication.html)

# COMMAND ----------

# Paste your token below

token = "<insert your token here>" 

# COMMAND ----------

# MAGIC %md Log your model

# COMMAND ----------

from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_squared_error
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

with mlflow.start_run(run_name="Webhook RF Experiment") as run:
  # Data prep
  df = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")
  X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)
  signature = infer_signature(X_train, pd.DataFrame(y_train))
  example = X_train.head(3)
  
  # Train and log model
  rf = RandomForestRegressor(random_state=42)
  rf.fit(X_train, y_train)
  mlflow.sklearn.log_model(rf, "random-forest-model", signature=signature, input_example=example)
  mse = mean_squared_error(y_test, rf.predict(X_test))
  mlflow.log_metric("mse", mse)
  run_id = run.info.run_id
  experiment_id = run.info.experiment_id

# COMMAND ----------

# MAGIC %md Register the model

# COMMAND ----------

name = f"webhook_demo_{run_id}"
model_uri = "runs:/{run_id}/random-forest-model".format(run_id=run_id)

model_details = mlflow.register_model(model_uri=model_uri, name=name)

# COMMAND ----------

# MAGIC %md Create a job that executes the notebook `03b-Webhooks-Job-Demo` in the same folder as this notebook.<br><br>
# MAGIC 
# MAGIC - Hover over the sidebar in the Databricks UI on the left.
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/ml-deployment/Job_1.png" alt="step12" width="100"/>
# MAGIC <br></br>
# MAGIC 
# MAGIC - Click on Create Job
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/Job_2.png" alt="step12" width="750"/>
# MAGIC 
# MAGIC <br></br>
# MAGIC - Name your Job
# MAGIC - Select the notebook `03b-Webhooks-Job-Demo` 
# MAGIC - Select the current cluster
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/Job_3_6.png" alt="step12" width="750"/>
# MAGIC 
# MAGIC <br></br>
# MAGIC - Copy the Job ID
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/Job_7.png" alt="step12" width="750"/>

# COMMAND ----------

# Enter Job ID here

job_ID = "<insert your job id here>" 

# COMMAND ----------

# MAGIC %md ## Examine the Job
# MAGIC 
# MAGIC Take a look at [the notebook you just scheduled]($./03b-Webhooks-Job-Demo) to see what it accomplishes.

# COMMAND ----------

# MAGIC %md ## Create a Webhook
# MAGIC 
# MAGIC There are a few different events that can trigger a Webhook. In this notebook, we will be experimenting with triggering a job when our model transitions between stages.

# COMMAND ----------

#We need an enpoint at which to execute our request which we can get from the Notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
# With the token, we can create our authorization header for our subsequent REST calls
headers = {'Authorization': f'Bearer {token}'}

# Next we need an enpoint at which to execute our request which we can get from the Notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
# This ojbect comes from the Java CM - Convert the Java Map opject to a Python dictionary
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)
# Lastly, extract the databricks instance (domain name) from the dictionary
instance = tags['browserHostName']

# COMMAND ----------

import json
import urllib

url = f"https://{instance}"
endpoint = f"{url}/api/2.0/mlflow/registry-webhooks/create"

new_json = {"model_name": name,
  "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
  "description": "Job webhook trigger",
  "status": "Active",
  "job_spec": {
    "job_id": job_ID,
    "workspace_url": url,
    "access_token": token}}

json_body = json.dumps(new_json)
json_bytes = json_body.encode('utf-8')

req = urllib.request.Request(endpoint, data=json_bytes, headers=headers)
response = urllib.request.urlopen(req)

print(json.load(response))

# COMMAND ----------

# MAGIC %md Now that we have registered the webhook, we can **test it by transitioning our model from stage `None` to `Staging` in the Experiment UI.** We should see in the Jobs tab that our Job has run.

# COMMAND ----------

# MAGIC %md To get a list of active Webhooks, use a GET request with the LIST endpoint. Note that this command will return an error if no Webhooks have been created for the Model.

# COMMAND ----------

new_url = f"{url}/api/2.0/mlflow/registry-webhooks/list/?model_name={name.replace(' ', '%20')}"
req = urllib.request.Request(new_url, headers=headers)
response = json.load(urllib.request.urlopen(req))

for webhook in response.get("webhooks"):
  print(webhook)
  print()

# COMMAND ----------

# MAGIC %md Finally, delete the webhook by copying the webhook ID to the curl or python request. You can confirm that the Webhook was deleted by using the list request.

# COMMAND ----------

delete_hook = "<insert your webhook id here>" 

json_bytes = json.dumps({"id": delete_hook}).encode('utf-8')

req = urllib.request.Request(f"{url}/api/2.0/mlflow/registry-webhooks/delete", data=json_bytes, headers=headers, method='DELETE')
response = urllib.request.urlopen(req)

print(json.load(response))

# COMMAND ----------

# MAGIC %md ## Resources
# MAGIC 
# MAGIC - [See this blog for more details on CI/CD and webhooks](https://databricks.com/blog/2020/11/19/mlflow-model-registry-on-databricks-simplifies-mlops-with-ci-cd-features.html)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
