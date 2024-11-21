# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Real Time Deployment
# MAGIC 
# MAGIC While real time deployment represents a smaller share of the deployment landscape, many of these deployments represent high value tasks.  This lesson surveys real time deployment options ranging from proofs of concept to both custom and managed solutions with a focus on RESTful services.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Survey the landscape of real time deployment options
# MAGIC  - Deploy a REST endpoint using AzureML and ACI for development
# MAGIC  - Deploy a REST endpoint using AzureML and AKS for production
# MAGIC  - Query a REST endpoint for inference using individual records and batch requests
# MAGIC  
# MAGIC **Libraries Required:**
# MAGIC  * `azureml-sdk==1.18.0`

# COMMAND ----------

# %pip install azureml-sdk==1.18.0

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md ### The Why and How of Real Time Deployment
# MAGIC 
# MAGIC Real time inference is...<br><br>
# MAGIC 
# MAGIC * Generating predictions for a small number of records with fast results (e.g. results in milliseconds)
# MAGIC * The first question to ask when considering real time deployment is: do I need it?  
# MAGIC   - It represents a minority of machine learning inference use cases 
# MAGIC   - Is one of the more complicated ways of deploying models
# MAGIC   - That being said, domains where real time deployment is often needed are often of great business value.  
# MAGIC   
# MAGIC Domains needing real time deployment include...<br><br>
# MAGIC 
# MAGIC  - Financial services (especially with fraud detection)
# MAGIC  - Mobile
# MAGIC  - Ad tech

# COMMAND ----------

# MAGIC %md There are a number of ways of deploying models...<br><br>
# MAGIC 
# MAGIC * Many use REST
# MAGIC * For basic prototypes, MLflow can act as a development deployment server
# MAGIC   - The MLflow implementation is backed by the Python library Flask
# MAGIC   - *This is not intended to for production environments*
# MAGIC 
# MAGIC For production RESTful deployment, there are two main options...<br><br>
# MAGIC 
# MAGIC * A managed solution 
# MAGIC   - Azure ML
# MAGIC * A custom solution  
# MAGIC   - Involve deployments using a range of tools
# MAGIC   - Often using Docker or Kubernetes
# MAGIC * One of the crucial elements of deployment in containerization
# MAGIC   - Software is packaged and isolated with its own application, tools, and libraries
# MAGIC   - Containers are a more lightweight alternative to virtual machines
# MAGIC 
# MAGIC Finally, embedded solutions are another way of deploying machine learning models, such as storing a model on IoT devices for inference.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##Azure ML for dev deployment using Azure Container Instances (ACI)
# MAGIC 
# MAGIC Let's create a Sklearn random forest model for the airbnb data we have been working with.
# MAGIC 
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> **Please be sure to delete any infrastructure you build after the course so you don't incur unexpected expenses.**

# COMMAND ----------

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow.sklearn

# Define paths
modelPath = "random-forest-model"

# Train and log model
df = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

rf = RandomForestRegressor(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)

with mlflow.start_run(run_name="RF Model") as run: 
  mlflow.sklearn.log_model(rf, modelPath)
  runID = run.info.run_id
  
print("Run completed with ID {} and path {}".format(runID, modelPath))
model_uri = f"runs:/{run.info.run_id}/{modelPath}"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Before models can be deployed to Azure ML, you must create or obtain an Azure ML Workspace. The `azureml.core.Workspace.create()` function will load a workspace of a specified name or create one if it does not already exist. For more information about creating an Azure ML Workspace, see the [Azure ML Workspace management documentation](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace).
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Provide your own subscription ID and resource group in the cell below.<br>
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Follow the instructions that will be printed out when the cell is run to authenticate yourself.

# COMMAND ----------

workspace_name = ""
workspace_location = ""
resource_group = ""
subscription_id = ""

# COMMAND ----------

# MAGIC %md Now create a new workspace.  This uses a preexisting workspace if it already exists.

# COMMAND ----------

# import azureml
# from azureml.core import Workspace

# try:
#   workspace = Workspace.create(
#     name = workspace_name,
#     location = workspace_location,
#     resource_group = resource_group,
#     subscription_id = subscription_id,
#     exist_ok=True
#   )
# except Exception as e:
#   print("Error: Confirm that your workspace name, location, resource group, and subscription id were all correct \n")
#   print(e)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Use the `mlflow.azuereml.build_image` function to build an Azure Container Image for the trained MLflow model. This function also registers the MLflow model with a specified Azure ML workspace. The resulting image can be deployed to **Azure Container Instances (ACI)** or **Azure Kubernetes Service (AKS)** for real-time serving.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> ACI is a good solution for prototyping.  For production deployments, AKS is generally preferred.

# COMMAND ----------

# import mlflow.azureml

# # Build the image
# model_image, azure_model = mlflow.azureml.build_image(
#   model_uri=model_uri, 
#   workspace=workspace, 
#   model_name="rfmodel",
#   image_name="rfmodel",
#   description="Sklearn random forest image for predicting airbnb housing prices",
#   synchronous=False
# )

# # Wait for the image creation
# model_image.wait_for_creation(show_output=True)

# COMMAND ----------

# MAGIC %md The [Azure Container Instances (ACI) platform](https://docs.microsoft.com/en-us/azure/container-instances/) is the recommended environment for staging and developmental model deployments.
# MAGIC 
# MAGIC Using the Azure ML SDK, deploy the Container Image for the trained MLflow model to ACI.  This involves creating a webservice deployment.

# COMMAND ----------

# from azureml.core.webservice import AciWebservice, Webservice

# # Create the deployment 
# dev_webservice_name = "airbnb-model-dev"
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

# MAGIC %md Use the `Webservice` constructor to create an object associated with the Workspace.

# COMMAND ----------

# from azureml.core.webservice import AciWebservice, Webservice

# dev_webservice = Webservice(workspace, dev_webservice_name)

# COMMAND ----------

# MAGIC %md Create a sample query input.

# COMMAND ----------

sample = df.drop(["price"], axis=1).iloc[[0]]

query_input = sample.to_json(orient='split')
query_input = eval(query_input)
query_input.pop('index', None)

query_input

# COMMAND ----------

# MAGIC %md Create a wrapper function to make queries to a URI.

# COMMAND ----------

import requests
import json

def query_endpoint(scoring_uri, inputs, service_key=None):
  headers = {
    "Content-Type": "application/json",
  }
  if service_key is not None:
    headers["Authorization"] = "Bearer {service_key}".format(service_key=service_key)
    
  print("Sending batch prediction request with inputs: {}".format(inputs))
  response = requests.post(scoring_uri, data=json.dumps(inputs), headers=headers)
  preds = json.loads(response.text)
  print("Received response: {}".format(preds))
  return preds

# COMMAND ----------

# MAGIC %md Query the endpoint using the scoring URI.

# COMMAND ----------

# dev_prediction = query_endpoint(scoring_uri=dev_webservice.scoring_uri, inputs=query_input)

# COMMAND ----------

# MAGIC %md ### Production Deployment to Azure ML using Azure Kubernetes Services (AKS)
# MAGIC 
# MAGIC [Azure Kubernetes Service (AKS)](https://azure.microsoft.com/en-us/services/kubernetes-service/) is the preferred option for production model deployment. Choose one of the following:<br><br>
# MAGIC 
# MAGIC 1. **Option 1:** Creates a new AKS cluster
# MAGIC 2. **Option 2:** Launch against a pre-existing AKS cluster

# COMMAND ----------

# MAGIC %md #### Option 1: Create a New AKS Cluster
# MAGIC 
# MAGIC If you do not have an active AKS cluster for model deployment, create one using the Azure ML SDK.

# COMMAND ----------

# from azureml.core.compute import AksCompute, ComputeTarget

# # Use the default configuration (you can also provide parameters to customize this)
# prov_config = AksCompute.provisioning_configuration(location='westus')
# aks_cluster_name = "prod-ml" 

# # Create the cluster
# aks_target = ComputeTarget.create(
#   workspace = workspace, 
#   name = aks_cluster_name, 
#   provisioning_configuration = prov_config
# )

# # Wait for the create process to complete
# aks_target.wait_for_completion(show_output = True)
# print(aks_target.provisioning_state)
# print(aks_target.provisioning_errors)

# COMMAND ----------

# MAGIC %md #### Option 2: Use a Pre-existing Cluster
# MAGIC 
# MAGIC If you already have an active AKS cluster running, you can add it to your Workspace using the Azure ML SDK.

# COMMAND ----------

# from azureml.core.compute import AksCompute, ComputeTarget
# from azureml.core import Image

# aks_cluster_name = "prod-ml"

# aks_target = ComputeTarget(workspace, aks_cluster_name)
# model_image = Image(workspace, name="rfmodel")

# COMMAND ----------

# MAGIC %md Now that you have defined an AKS cluster that is up and running, confirm that it is in `Succeeded` status.

# COMMAND ----------

# aks_target.get_status()

# COMMAND ----------

# MAGIC %md Now deploy the model image to it.

# COMMAND ----------

# from azureml.core.webservice import Webservice, AksWebservice

# # Set configuration and service name
# prod_webservice_name = "airbnb-model"
# prod_webservice_deployment_config = AksWebservice.deploy_configuration()

# # Deploy from image
# prod_webservice = Webservice.deploy_from_image(
#   workspace=workspace, 
#   name=prod_webservice_name,
#   image=model_image,
#   deployment_config=prod_webservice_deployment_config,
#   deployment_target=aks_target
# )

# # Wait for the deployment to complete
# prod_webservice.wait_for_deployment(show_output = True)

# COMMAND ----------

# MAGIC %md You can also get the webservice this way when it is already up and running.

# COMMAND ----------

# from azureml.core.webservice import AciWebservice, Webservice

# prod_webservice = Webservice(workspace, prod_webservice_name)

# COMMAND ----------

# MAGIC %md Query the AKS webservice's scoring endpoint by sending an HTTP POST request that includes the input vector. The production AKS deployment may require an authorization token (service key) for queries. Include this key in the HTTP request header.  We'll use the `query_endpoint()` function we defined above.
# MAGIC 
# MAGIC Pull the endpoint URI and key.

# COMMAND ----------

# prod_scoring_uri = prod_webservice.scoring_uri
# prod_service_key = prod_webservice.get_keys()[0] if len(prod_webservice.get_keys()) > 0 else None

# COMMAND ----------

# MAGIC %md Create a query input.

# COMMAND ----------

def get_queries():
  
  sample = df.drop(["price"], axis=1).iloc[:2,:]

  query_input = sample.to_json(orient='split')
  query_input = eval(query_input)
  query_input.pop('index', None)
  return query_input

query_input=get_queries()
query_input

# COMMAND ----------

# MAGIC %md Query the endpoint.

# COMMAND ----------

# prod_prediction = query_endpoint(scoring_uri=prod_scoring_uri, service_key=prod_service_key, inputs=query_input)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> **Please be sure to delete any infrastructure you build after the course so you don't incur unexpected expenses.**

# COMMAND ----------

# MAGIC %md ## Review
# MAGIC **Question:** What are the best tools for real time deployment?  
# MAGIC **Answer:** This depends largely on the desired features.  The main tools to consider are a way to containerize code and either a REST endpoint or an embedded model.  This covers the vast majority of real time deployment options.
# MAGIC 
# MAGIC **Question:** What are the best options for RESTful services?  
# MAGIC **Answer:** The major cloud providers all have their respective deployment options.  In the Azure environment, Azure ML manages deployments using Docker images. This provides a REST endpoint that can be queried by various elements of your infrastructure.
# MAGIC 
# MAGIC **Question:** What factors influence REST deployment latency?  
# MAGIC **Answer:** Response time is a function of a few factors.  Batch predictions should be used when needed since it improves throughput by lowering the overhead of the REST connection.  Geo-location is also an issue, as is server load.  This can be handled by geo-located deployments and load balancing with more resources.

# COMMAND ----------

# MAGIC %md ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find out more information on MLflow's `pyfunc`?  
# MAGIC **A:** Check out <a href="https://www.mlflow.org/docs/latest/models.html#pyfunc-deployment" target="_blank">the MLflow documentation</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
