# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Real Time Deployment
# MAGIC 
# MAGIC While real time deployment represents a smaller share of the deployment landscape, many of these deployments represent high value tasks.  This lesson surveys real-time deployment options ranging from proofs of concept to both custom and managed solutions.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Survey the landscape of real-time deployment options
# MAGIC  - Prototype a RESTful service using MLflow
# MAGIC  - Deploy registered models using MLflow Model Serving
# MAGIC  - Query an MLflow Model Serving endpoint for inference using individual records and batch requests

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Why and How of Real Time Deployment
# MAGIC 
# MAGIC Real time inference is...<br><br>
# MAGIC 
# MAGIC * Generating predictions for a small number of records with fast results (e.g. results in milliseconds)
# MAGIC * The first question to ask when considering real time deployment is: do I need it?  
# MAGIC   - It represents a minority of machine learning inference use cases &mdash; it's necessary when features are only available at the time of serving
# MAGIC   - Is one of the more complicated ways of deploying models
# MAGIC   - That being said, domains where real time deployment is often needed are often of great business value.  
# MAGIC   
# MAGIC Domains needing real time deployment include...<br><br>
# MAGIC 
# MAGIC  - Financial services (especially with fraud detection)
# MAGIC  - Mobile
# MAGIC  - Ad tech

# COMMAND ----------

# MAGIC %md
# MAGIC There are a number of ways of deploying models...<br><br>
# MAGIC 
# MAGIC * Many use REST
# MAGIC * For basic prototypes, MLflow can act as a development deployment server
# MAGIC   - The MLflow implementation is backed by the Python library Flask
# MAGIC   - *This is not intended to for production environments*
# MAGIC 
# MAGIC In addition, Databricks offers a managed **MLflow Model Serving** solution. This solution allows you to host machine learning models from Model Registry as REST endpoints that are automatically updated based on the availability of model versions and their stages.
# MAGIC 
# MAGIC For production RESTful deployment, there are two main options...<br><br>
# MAGIC 
# MAGIC * A managed solution 
# MAGIC   - Azure ML
# MAGIC   - SageMaker
# MAGIC * A custom solution  
# MAGIC   - Involve deployments using a range of tools
# MAGIC   - Often using Docker or Kubernetes
# MAGIC * One of the crucial elements of deployment in containerization
# MAGIC   - Software is packaged and isolated with its own application, tools, and libraries
# MAGIC   - Containers are a more lightweight alternative to virtual machines
# MAGIC 
# MAGIC Finally, embedded solutions are another way of deploying machine learning models, such as storing a model on IoT devices for inference.

# COMMAND ----------

# MAGIC %md-sandbox ## Prototyping with MLflow
# MAGIC 
# MAGIC MLflow offers <a href="https://www.mlflow.org/docs/latest/models.html#pyfunc-deployment" target="_blank">a Flask-backed deployment server for development.</a>
# MAGIC 
# MAGIC Let's build a simple model below. This model will always predict 5.

# COMMAND ----------

import mlflow
import mlflow.pyfunc

class TestModel(mlflow.pyfunc.PythonModel):
  
  def predict(self, context, input_df):
    return 5
  
model_run_name="pyfunc-model"

with mlflow.start_run() as run:
  mlflow.pyfunc.log_model(artifact_path=model_run_name, python_model=TestModel())
  
  model_uri = f"runs:/{run.info.run_id}/{model_run_name}"

# COMMAND ----------

# MAGIC %md There are a few ways to send requests to the development server for testing purpose:
# MAGIC * using `click` library 
# MAGIC * using MLflow Model Serving API
# MAGIC * through CLI using `mlflow models serve`
# MAGIC 
# MAGIC In this lesson, we are going to demonstrate how to use both the `click` library and MLflow Model Serving API. 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Models can be served in this way in other languages as well.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 1: Using `click` Library

# COMMAND ----------

import time
import mlflow
from mlflow import pyfunc
from multiprocessing import Process

server_port_number = 6501
hostName = "127.0.0.1"

def run_server():
  try:
    import mlflow.models.cli
    from click.testing import CliRunner
    
    CliRunner().invoke(mlflow.models.cli.commands, 
                       ['serve', 
                        "--model-uri", model_uri, 
                        "-p", server_port_number, 
                        "-w", 4,
                        "--host", hostName, # "127.0.0.1", 
                        "--no-conda"])
  except Exception as e:
    print(e)

p = Process(target=run_server) # Create a background process
p.start()                      # Start the process
time.sleep(5)                  # Give it 5 seconds to startup
print(p)                       # Print it's status, make sure it's runnning

# COMMAND ----------

# MAGIC %md Create an input for our REST input.

# COMMAND ----------

import json
import pandas as pd

input_df = pd.DataFrame([0])
input_json = input_df.to_json(orient='split')

input_json

# COMMAND ----------

# MAGIC %md Perform a POST request against the endpoint.

# COMMAND ----------

import requests
from requests.exceptions import ConnectionError
from time import sleep

headers = {'Content-type': 'application/json'}
url = f"http://{hostName}:{server_port_number}/invocations"

try:
  response = requests.post(url=url, headers=headers, data=input_json)
except ConnectionError:
  print("Connection fails on a Run All.  Sleeping and will try again momentarily...")
  sleep(5)
  response = requests.post(url=url, headers=headers, data=input_json)

print(f"Status: {response.status_code}")
print(f"Value:  {response.text}")

# COMMAND ----------

# MAGIC %md Do the same in bash.

# COMMAND ----------

# MAGIC %sh (echo -n '{"columns":[0],"index":[0],"data":[[0]]}') | curl -H "Content-Type: application/json" -d @- http://127.0.0.1:6501/invocations

# COMMAND ----------

# MAGIC %md Clean up the background process.

# COMMAND ----------

p.terminate()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 2: MLflow Model Serving
# MAGIC Now, let's use MLflow Model Serving. 
# MAGIC 
# MAGIC Step 1: We first need to register the model in MLflow Model Registry and load the model. At this step, we don't specify the model stage, so that the stage version would be `None`. 
# MAGIC 
# MAGIC You can refer to the MLflow documentation [here](https://www.mlflow.org/docs/latest/model-registry.html#api-workflow).

# COMMAND ----------

import uuid

unique_id = uuid.uuid4().hex[:6]
model_name = f"demo-model_{unique_id}"

client = mlflow.tracking.MlflowClient()
mlflow.register_model(model_uri=run.info.artifact_uri+"/pyfunc-model", name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Step 2: Run Tests Against Registered Model in order to Promote To Staging

# COMMAND ----------

import time
time.sleep(10) # to wait for registration to complete

model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC Here, visit the MLflow Model Registry to enable Model Serving. 
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/mlflow/demo_model_register.png" width="600" height="20"/>

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

import os
import requests
import pandas as pd

def score_model(dataset: pd.DataFrame):
  
  url = f'https://{instance}/model/{model_name}/1/invocations'
  
  data_json = dataset.to_dict(orient='split')
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC After the model serving cluster is in the `ready` state, you can now send requests to the REST endpoint.

# COMMAND ----------

score_model(pd.DataFrame([0]))

# COMMAND ----------

# MAGIC %md
# MAGIC You can also optionally transition the model to `staging` or `production` stage, using [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html#transitioning-an-mlflow-models-stage). 
# MAGIC 
# MAGIC Sample code is below:
# MAGIC ```
# MAGIC client.transition_model_version_stage(
# MAGIC   name=model_name,
# MAGIC   version=model_version,
# MAGIC   stage="Staging",
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> **Remember to shut down the Model Serving Cluster to avoid incurring unexpected cost**. It does not terminate automatically! Click on `Stop` next to `Status` to stop the serving cluster.
# MAGIC <Br>
# MAGIC 
# MAGIC <div><img src="http://files.training.databricks.com/images/mlflow/demo_model_hex.png" style="height: 250px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md <img src="https://files.training.databricks.com/images/icon_warn_24.png"/> **Please be sure to delete any infrastructure you build after the course so you don't incur unexpected expenses.**

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Review
# MAGIC **Question:** What are the best tools for real time deployment?  
# MAGIC **Answer:** This depends largely on the desired features.  The main tools to consider are a way to containerize code and either a REST endpoint or an embedded model.  This covers the vast majority of real time deployment options.
# MAGIC 
# MAGIC **Question:** What are the best options for RESTful services?  
# MAGIC **Answer:** The major cloud providers all have their respective deployment options.  In the Azure environment, Azure ML manages deployments using Docker images. This provides a REST endpoint that can be queried by various elements of your infrastructure.
# MAGIC 
# MAGIC **Question:** What factors influence REST deployment latency?  
# MAGIC **Answer:** Response time is a function of a few factors.  Batch predictions should be used when needed since it improves throughput by lowering the overhead of the REST connection.  Geo-location is also an issue, as is server load.  This can be handled by geo-located deployments and load balancing with more resources.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Lab<br>
# MAGIC 
# MAGIC Start the labs for this lesson, [Real Time Lab]($./Labs/02-Real-Time-Lab)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find out more information on MLflow's `pyfunc`?  
# MAGIC **A:** Check out <a href="https://www.mlflow.org/docs/latest/models.html#pyfunc-deployment" target="_blank">the MLflow documentation</a>
# MAGIC 
# MAGIC **Q:** Where can I learn more about MLflow Model Serving on Databricks?   
# MAGIC **A:** Check out <a href="https://docs.databricks.com/applications/mlflow/model-serving.html#language-python" target="_blank">this MLflow Model Serving on Databricks documentation</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
