# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img src="https://files.training.databricks.com/images/Apache-Spark-Logo_TM_200px.png" style="float: left: margin: 20px"/>
# MAGIC # Course Overview
# MAGIC ## Machine Learning in Production
# MAGIC ### Managing the Complete Machine Learning Lifecycle with MLflow, Deployment and CI/CD
# MAGIC 
# MAGIC In this 1-day course, machine learning engineers, data engineers, and data scientists learn the best practices for managing the complete machine learning lifecycle from experimentation and model management through various deployment modalities and production issues. Students begin with end-to-end reproducibility of machine learning models using MLflow including data management, experiment tracking, and model management before deploying models with batch, streaming, and real time as well as addressing related monitoring, alerting, and CI/CD issues. Sample code accompanies all modules and theoretical concepts.
# MAGIC 
# MAGIC First, this course explores managing the experimentation process using MLflow with a focus on end-to-end reproducibility including data, model, and experiment tracking. Second, students operationalize their models by integrating with various downstream deployment tools including saving models to the MLflow model registry, managing artifacts and environments, and automating the testing of their models. Third, students implement batch, streaming, and real time deployment options. Finally, additional production issues including continuous integration, continuous deployment are covered as well as monitoring and alerting.
# MAGIC 
# MAGIC By the end of this course, you will have built an end-to-end pipeline to log, deploy, and monitor machine learning models. This course is taught entirely in Python.
# MAGIC 
# MAGIC ## Lessons
# MAGIC 
# MAGIC 00. Course Overview
# MAGIC 01. Experimentation
# MAGIC   01. Data Management
# MAGIC   02. Experiment Tracking
# MAGIC   03. Advanced Experiment Tracking
# MAGIC 02. Model Management
# MAGIC   01. Model Mangagement
# MAGIC   02. Model Registry
# MAGIC   03. Webhooks and Testing
# MAGIC 03. Deployment Paradigms
# MAGIC   01. Batch
# MAGIC   02. Real Time
# MAGIC   03. Streaming
# MAGIC 04. Production
# MAGIC   01. CI/CD
# MAGIC   02. Monitoring
# MAGIC   03. Alerting
# MAGIC 05. Pipeline Example
# MAGIC   01. Orchestrate
# MAGIC   02. Data Validate
# MAGIC   03. Featurize
# MAGIC   04. Model Validate
# MAGIC   05. Score
# MAGIC   06. Monitor
# MAGIC 06. Capstones
# MAGIC   01. Preproduction Capstone
# MAGIC   02. Production Capstone
# MAGIC 
# MAGIC ## Audience
# MAGIC * Primary Audience: Machine Learning Engineers and Data Engineers 
# MAGIC * Additional Audiences: Data Scientists and Data Analysts
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC - Web browser: Chrome
# MAGIC - A cluster configured with **DBR 8.3 ML**
# MAGIC - Python (`pandas`, `sklearn`, `numpy`)
# MAGIC - Background in machine learning and data science
# MAGIC 
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> **Certain features used in this course, such as the notebooks API and model registry, are only available to paid or trial subscription users of Databricks.**
# MAGIC If you are using the Databricks Community Edition, click the `Upgrade` button on the landing page <a href="https://accounts.cloud.databricks.com/registration.html#login" target="_blank">or navigate here</a> to start a free trial.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> Before You Start</h2>
# MAGIC 
# MAGIC Before starting this course, you will need to create a cluster and attach it to this notebook.
# MAGIC 
# MAGIC Please configure your cluster to use the Databricks **ML Runtime** version **8.3 ML**.
# MAGIC 
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> Do not use the stock or GPU accelerated runtimes
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> This courseware has been tested against the specific DBR listed above. Using an untested DBR may yield unexpected results and/or various errors. 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC In general, all courses are designed to run on one of the following Databricks platforms:
# MAGIC * Databricks Community Edition (CE)
# MAGIC * Databricks (an AWS hosted service)
# MAGIC * Databricks (an Google hosted service)
# MAGIC * Azure-Databricks (an Azure-hosted service)
# MAGIC 
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> Some features are not available on the Community Edition, which limits the ability of some courses to be executed in that environment. Please see the course's prerequisites for specific information on this topic.
# MAGIC 
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> Additionally, private installations of Databricks (e.g., accounts provided by your employer) may have other limitations imposed, such as aggressive permissions and or language restrictions such as prohibiting the use of Scala which will further inhibit some courses from being executed in those environments.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** All courses provided by Databricks Academy rely on custom variables, functions, and settings to provide you with the best experience possible.
# MAGIC 
# MAGIC For each lesson to execute correctly, please make sure to run the **`Classroom-Setup`** cell at the start of each lesson (see the next cell).

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Agile Data Science
# MAGIC 
# MAGIC Deploying machine learning models into production comes with a wide array of challenges, distinct from those data scientists face when they're initially training models.  Teams often solve these challenges with custom, in-house solutions that are often brittle, monolithic, time consuming, and difficult to maintain.
# MAGIC 
# MAGIC A systematic approach to the deployment of machine learning models results in an agile solution that minimizes developer time and maximizes the business value derived from data science.  To achieve this, data scientists and data engineers need to navigate various deployment solutions as well as have a system in place for monitoring and alerting once a model is out in production.
# MAGIC 
# MAGIC The main deployment paradigms are as follows:<br><br>
# MAGIC 
# MAGIC 1. **Batch:** predictions are created and stored for later use, such as a database that can be queried in real time in a web application
# MAGIC 2. **Streaming:** data streams are transformed where the prediction is needed soon after it arrives in a data pipeline but not immediately
# MAGIC 3. **Real time:** normally implemented with a REST endpoint, a prediction is needed on the fly with low latency
# MAGIC 4. **Mobile/Embedded:** entails embedding machine learning solutions in mobile or IoT devices and is outside the scope of this course
# MAGIC 
# MAGIC Once a model is deployed in one of these paradigms, it needs to be monitored for performance with regards to the quality of predictions, latency, throughput, and other production considerations.  When performance starts to slip, this is an indication that the model needs to be retrained, more resources need to be allocated to serving the model, or any number of improvements are needed.  An alerting infrastructure needs to be in place to capture these performance issues.
# MAGIC 
# MAGIC <br>
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-1/ml-stock.jpg" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
