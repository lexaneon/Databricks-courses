# Databricks notebook source
# MAGIC %md-sandbox
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
# MAGIC  - Walk through the deployment of REST endpoint using SageMaker
# MAGIC  - Query a REST endpoint for inference using individual records and batch requests

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Why and How of Real Time Deployment
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
# MAGIC  - Adtech

# COMMAND ----------

# MAGIC %md
# MAGIC There are a number of ways of deploying models...<br><br>
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
# MAGIC   - SageMaker
# MAGIC * A custom solution  
# MAGIC   - Involve deployments using a range of tools
# MAGIC   - Often using Docker, Kubernetes, and Elastic Beanstalk
# MAGIC * One of the crucial elements of deployment in containerization
# MAGIC   - Software is packaged and isolated with its own application, tools, and libraries
# MAGIC   - Containers are a more lightweight alternative to virtual machines
# MAGIC 
# MAGIC Finally, embedded solutions are another way of deploying machine learning models, such as storing a model on IoT devices for inference.

# COMMAND ----------

# MAGIC %md ## Sagemaker Deployment
# MAGIC 
# MAGIC AWS offers the managed service SageMaker.  It allows data scientists a way of deploying machine learning models, offering a REST endpoint to make inference calls to.  MLflow integrates with SageMaker by way of containers using Amazon Container Services (ACS).  In order to use SageMaker you therefore need the following:<br><br>
# MAGIC 
# MAGIC 1. IAM credentials that give access to ACS and SageMaker
# MAGIC 2. An MLflow container image on ACS
# MAGIC 
# MAGIC We'll use the model from above.

# COMMAND ----------

# MAGIC %md ### Set up the `mlflow-pyfunc` Docker Image in ECR
# MAGIC 
# MAGIC During deployment, MLflow will use a specialized Docker container with resources required to load and serve the model. This container is named `mlflow-pyfunc`.
# MAGIC 
# MAGIC By default, MLflow will search for this container within your AWS Elastic Container Registry (ECR). You can build and upload this container to ECR using the
# MAGIC `mlflow.sagemaker.build_image()` function in MLflow, or `mlflow sagemaker build-and-push-container`, as it will require `docker`.
# MAGIC 
# MAGIC Alternatively, you can specify an alternative URL for this container by setting an environment variable as follows.
# MAGIC The ECR URL should look like: `{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}`
# MAGIC 
# MAGIC `os.environ["SAGEMAKER_DEPLOY_IMG_URL"] = "<ecr-url>"`
# MAGIC 
# MAGIC You can contact your Databricks representative for a prebuilt `mlflow-pyfunc` image URL in ECR.
# MAGIC 
# MAGIC Use MLflow's SageMaker API to deploy your trained model to SageMaker. The `mlflow.sagemaker.deploy()` function creates a SageMaker endpoint as well as all intermediate SageMaker objects required for the endpoint.
# MAGIC 
# MAGIC 
# MAGIC To do this at the command line, you can use something like the following:
# MAGIC 
# MAGIC ```
# MAGIC ACCOUNTID=""
# MAGIC REGION=us-west-2
# MAGIC 
# MAGIC mlflow sagemaker build-and-push-container
# MAGIC $(aws ecr get-login --no-include-email --region REGION)
# MAGIC docker tag mlflow-pyfunc:latest ${ACCOUNTID}.dkr.ecr.${REGION}.amazonaws.com/mlflow-pyfunc:latest
# MAGIC docker push ${ACCOUNTID}.dkr.ecr.${REGION}.amazonaws.com/mlflow-pyfunc:latest```
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_warn_24.png"/> This step is left up to the user since it depends on your IAM roles
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> You must create a new SageMaker endpoint for each new region

# COMMAND ----------

# MAGIC %md ### Deploying the Model
# MAGIC 
# MAGIC Once ECS is set up, you can use `mlflow.sagemaker.deploy` to deploy your model.  **This code depends on you filling out your AWS specification.** This includes:
# MAGIC <br><br>
# MAGIC 1. `app_name`: the name for how the app appears on SageMaker 
# MAGIC 2. `run_id`: the ID for the MLflow run 
# MAGIC 3. `model_path`: the path for the MLflow model 
# MAGIC 4. `region_name`: your preferred AWS region 
# MAGIC 5. `mode`: use `create` to make a new instance. You can also replace a pre-existing model
# MAGIC 6. `execution_role_arn`: the role ARN 
# MAGIC 7. `instance_type`: what size ec2 machine
# MAGIC 8. `image_url`: the URL for the ECR image
# MAGIC 
# MAGIC First set AWS keys as environment variables. This is not a best practice since this is not the most secure way of handling credentials. This works in our case sense the keys have a very limited policy associated with them.
# MAGIC 
# MAGIC The following is read-only credentials for a pre-made SageMaker endpoint.  They will not work for deploying a model.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> See the Secrets API and IAM roles for more secure ways of storing keys.

# COMMAND ----------

# Set AWS credentials as environment variables
# import os
# os.environ["AWS_ACCESS_KEY_ID"] = 'AKIAI4T2MLVBUB372FAA'
# os.environ["AWS_SECRET_ACCESS_KEY"] = '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' # READ ONLY ACCESS KEYS
# os.environ["AWS_DEFAULT_REGION"] = 'us-west-2'

# COMMAND ----------

# MAGIC %md Fill in the following command with your region, ARN, and image URL to deploy your model.

# COMMAND ----------

# import mlflow.sagemaker as mfs
# import random

# appName = "airbnb-latest-{}".format(random.randint(1,10000))

# mfs.deploy(app_name=appName, 
#            run_id=runID, 
#            model_path=modelPath, 
#            region_name=" < FILL IN> ", 
#            mode="create", 
#            execution_role_arn=" < FILL IN> ", 
#            instance_type="ml.t2.medium",
#            image_url=" < FILL IN> " )

# COMMAND ----------

# MAGIC %md ### Test the Connection
# MAGIC 
# MAGIC You can test the connection using `boto3`, the library for interacting with AWS in Python. The appName is for a SageMaker endpoint that has already been setup.

# COMMAND ----------

# appName = "airbnb-latest-0001"

# import boto3

# def check_status(appName):
#   sage_client = boto3.client('sagemaker', region_name="us-west-2")
#   endpoint_description = sage_client.describe_endpoint(EndpointName=appName)
#   endpoint_status = endpoint_description["EndpointStatus"]
#   return endpoint_status

# print("Application status is: {}".format(check_status(appName)))

# COMMAND ----------

# MAGIC %md ### Evaluate the Input Vector by Sending an HTTP Request
# MAGIC Define a helper function that connects to the `sagemaker-runtime` client and sends the record in the appropriate JSON format.

# COMMAND ----------

import json

def query_endpoint_example(inputs, appName="airbnb-latest-0001", verbose=True):
  if verbose:
    print("Sending batch prediction request with inputs: {}".format(inputs))
  client = boto3.session.Session().client("sagemaker-runtime", "us-west-2")
  
  response = client.invoke_endpoint(
      EndpointName=appName,
      Body=json.dumps(inputs),
      ContentType='application/json',
  )
  preds = response['Body'].read().decode("ascii")
  preds = json.loads(preds)
  
  if verbose:
    print("Received response: {}".format(preds))
  return preds

# COMMAND ----------

# query_input = X_train.iloc[[0]].values.flatten().tolist()

# print("Using input vector: {}".format(query_input))

# prediction = query_endpoint_example(appName=appName, inputs=[query_input])

# COMMAND ----------

# MAGIC %md Now try the same but by using more than just one record.  Create a helper function to query the endpoint with a number of random samples.

# COMMAND ----------

def random_n_samples(n, df=X_train, verbose=False):
  dfShape = X_train.shape[0]
  samples = []
  
  for i in range(n):
    sample = X_train.iloc[[random.randint(0, dfShape-1)]].values
    samples.append(sample.flatten().tolist())
  
  return query_endpoint_example(samples, appName="airbnb-latest-0001", verbose=verbose)

# COMMAND ----------

# MAGIC %md Test this using 10 samples.  The payload for SageMaker can be 1 or more samples.

# COMMAND ----------

# random_n_samples(10, verbose=True)

# COMMAND ----------

# MAGIC %md Compare the times between payload sizes.  **Notice how sending more records at a time reduces the time to prediction for each individual record.**

# COMMAND ----------

# %timeit -n5 random_n_samples(100)

# COMMAND ----------

# %timeit -n5 random_n_samples(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleaning up the deployment
# MAGIC 
# MAGIC When your model deployment is no longer needed, run the `mlflow.sagemaker.delete()` function to delete it.

# COMMAND ----------

# Specify the archive=False option to delete any SageMaker models and configurations
# associated with the specified application
# mfs.delete(app_name=appName, region_name="us-west-2", archive=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Review
# MAGIC 
# MAGIC **Question:** What are the best tools for real time deployment?  
# MAGIC **Answer:** This depends largely on the desired features.  The main tools to consider are a way to containerize code and either a REST endpoint or an embedded model.  This covers the vast majority of real time deployment options.
# MAGIC 
# MAGIC **Question:** What are the best options for RESTful services?  
# MAGIC **Answer:** The major cloud providers all have their respective deployment options.  In the Azure environment, Azure ML manages deployments using Docker images.  AWS SageMaker does the same.  This provides a REST endpoint that can be queried by various elements of your infrastructure.
# MAGIC 
# MAGIC **Question:** What factors influence REST deployment latency?  
# MAGIC **Answer:** Response time is a function of a few factors.  Batch predictions should be used when needed since it improves throughput by lowering the overhead of the REST connection.  Geo-location is also an issue, as is server load.  This can be handled by geo-located deployments and load balancing with more resources.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find out more information on MLflow's `pyfunc`?  
# MAGIC **A:** Check out <a href="https://www.mlflow.org/docs/latest/models.html#pyfunc-deployment" target="_blank">the MLflow documentation</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
