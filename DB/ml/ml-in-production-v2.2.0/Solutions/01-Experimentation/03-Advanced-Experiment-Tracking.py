# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Advanced MLflow Tracking
# MAGIC 
# MAGIC In this lesson you explore more advanced MLflow tracking options available to extend logging to a wider variety of use cases.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Manage model inputs and outputs with MLflow signatures and input examples
# MAGIC  - Explore nested runs for hyperparameter tuning and iterative training
# MAGIC  - Integrate MLflow with HyperOpt
# MAGIC  - Log SHAP values and visualizations

# COMMAND ----------

# MAGIC %md ### Signatures and Input Examples
# MAGIC 
# MAGIC Previously, when logging a model in MLflow we only logged the model and name for the model artifact with `.log_model(model, model_name)`
# MAGIC 
# MAGIC However, it is a best practice to also log a model signature and input example. This allows for better schema checks and, therefore, integration with automated deployment tools.
# MAGIC 
# MAGIC **Signature**
# MAGIC * A model signature is just the schema of the input(s) and the output(s) of the model
# MAGIC * We usually get this with the `infer_schema` function
# MAGIC 
# MAGIC **Input Example**
# MAGIC * This is simply a few example inputs to the model 
# MAGIC * This will be converted to JSON and stored in our MLflow run
# MAGIC * It integrates well with MLflow model serving
# MAGIC 
# MAGIC In general, logging a model with these looks like `.log_model(model, model_name, signature=signature, input_example=input_example)`.
# MAGIC 
# MAGIC Let's look at an example, where we create a `sklearn` Random Forest Regressor model and log it with the signature and input example.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md Let's start by loading in the dataset

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

pdf = pd.read_parquet("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
X = pdf.drop("price", axis=1)
y = pdf["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md Now, let's train our model and log it with MLflow. This time, we will add a `signature` and `input_examples` when we log our model.

# COMMAND ----------

import mlflow 
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

with mlflow.start_run(run_name="Signature Example") as run:
  rf = RandomForestRegressor(random_state=42)
  rf_model = rf.fit(X_train, y_train)
  mse = mean_squared_error(rf_model.predict(X_test), y_test)
  mlflow.log_metric("mse", mse)
  
  # Log the model with signature and input example
  signature = infer_signature(X_train, pd.DataFrame(y_train))
  input_example = X_train.head(3)
  mlflow.sklearn.log_model(rf_model, "rf_model", signature=signature, input_example=input_example)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Look at the MLflow UI for this run to see our model signature and input example
# MAGIC 
# MAGIC <img style="width:50%" src="https://files.training.databricks.com/images/mlpupdates/signature_example.gif" >

# COMMAND ----------

# MAGIC %md ### Nested Runs
# MAGIC 
# MAGIC A useful organizational tool provided by MLflow is nested runs. Nested runs allow for parent runs and child runs in a tree structure. In the MLflow UI, you can click on a parent run to expand it and see the child runs. 
# MAGIC 
# MAGIC Example applications: 
# MAGIC * In **hyperparameter tuning**, you can nest all associated model runs under a parent run to better organize and compare hyperparameters. 
# MAGIC * In **parallel training** many models such as IoT devices, you can better aggregate the models. More information on this can be found [here]("https://databricks.com/blog/2020/05/19/manage-and-scale-machine-learning-models-for-iot-devices.html").
# MAGIC * In **iterative training** such as neural networks, you can checkpoint results after `n` epochs to save the model and related metrics.

# COMMAND ----------

with mlflow.start_run(run_name="Nested Example") as run:
  # Create nested run with nested=True argument
  with mlflow.start_run(run_name="Child 1", nested=True):
    mlflow.log_param("run_name", "child_1")
    
  with mlflow.start_run(run_name="Child 2", nested=True):
    mlflow.log_param("run_name", "child_2")

# COMMAND ----------

# MAGIC %md Take a look at the MLflow UI to see the nested runs.

# COMMAND ----------

# MAGIC %md ### Hyperparameter Tuning
# MAGIC 
# MAGIC One of the most common use cases for nested runs is hyperparameter tuning. For example, when running **HyperOpt** with SparkTrials on Databricks, it will automatically track the candidate models, parameters, etc as child runs in the MLflow UI.
# MAGIC 
# MAGIC Hyperopt allows for efficient hyperparameter tuning and now integrates with Apache Spark via:
# MAGIC 
# MAGIC * **Trials:** Sequential training of single-node or distributed ML models (e.g. MLlib)
# MAGIC * **SparkTrials:** Parallel training of single-node models (e.g. sklearn). The amount of parallelism is controlled via the `parallelism` parameter. 
# MAGIC 
# MAGIC Let's try using HyperOpt with SparkTrials to find the best sklearn random forest model. 
# MAGIC 
# MAGIC Check out this blog by Sean Owen on [How (Not) to Tune Your Model With Hyperopt](https://databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html).

# COMMAND ----------

# MAGIC %md Set up the Hyperopt run.  We need to define an objective function to minimize and a search space for the parameters for our Hyperopt run.
# MAGIC 
# MAGIC Hyperopt will work to minimize the objective function, so here we simply return the `loss` as the mse, since that is what we are trying to minimize. 
# MAGIC 
# MAGIC **Note**: If you're trying to maximize a metric, such as accuracy or r2, you would need to return `-accuracy` or `-r2` so Hyperopt can minimize it. 

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials

# Define objective function
def objective(params):
  model = RandomForestRegressor(n_estimators=int(params["n_estimators"]), 
                                max_depth=int(params["max_depth"]), 
                                min_samples_leaf=int(params["min_samples_leaf"]),
                                min_samples_split=int(params["min_samples_split"]))
  model.fit(X_train, y_train)
  pred = model.predict(X_train)
  score = mean_squared_error(pred, y_train)
  
  # Hyperopt minimizes score, here we minimize mse. 
  return score

# COMMAND ----------

# MAGIC %md Execute the MLflow Hyperopt Run
# MAGIC 
# MAGIC **Note:** This code using autologging.  You can also turn this on for other libraries such as `mlflow.tensorflow.autolog()`

# COMMAND ----------

from hyperopt import SparkTrials

# Define search space
search_space = {"n_estimators": hp.quniform("n_estimators", 100, 500, 5),
                "max_depth": hp.quniform("max_depth", 5, 20, 1),
                "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 5, 1),
                "min_samples_split": hp.quniform("min_samples_split", 2, 6, 1)}

# Set parallelism (should be order of magnitude smaller than max_evals)
spark_trials = SparkTrials(parallelism=2)

with mlflow.start_run(run_name="Hyperopt"):
  argmin = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=16,
    trials=spark_trials)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Look at the MLflow UI for the nested runs
# MAGIC 
# MAGIC <img style="width:50%" src="https://files.training.databricks.com/images/mlpupdates/HyperOpt.gif" >
# MAGIC 
# MAGIC If we select all the nested runs in this run and select `Compare` we can also create useful visualizations to better understand the hyperparameter tuning processes. 
# MAGIC 
# MAGIC Select `Compare` as shown above, and then `Parallel Coordinates Plot` in the next window to generate the following image. 
# MAGIC 
# MAGIC **Note**: You will have to add which parameters and metrics you want to generate the visualization.
# MAGIC 
# MAGIC <img style="width:50%" src="https://files.training.databricks.com/images/mlpupdates/Parallel Coordinates Plot.png" >

# COMMAND ----------

# MAGIC %md ### Advanced Artifact Tracking
# MAGIC 
# MAGIC In addition to the logging of artifacts you have already seen, there are some more advanced options. 
# MAGIC 
# MAGIC We will now look at: 
# MAGIC * [mlflow.shap](https://www.mlflow.org/docs/latest/python_api/mlflow.shap.html#mlflow.shap): Automatically calculates and logs Shapley feature importance plots
# MAGIC * [mlflow.log_figure](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_figure): Logs matplotlib and plotly plots

# COMMAND ----------

import matplotlib.pyplot as plt

with mlflow.start_run(run_name="Feature Importance Scores"):
  # Generate and log SHAP plot for first 5 records
  mlflow.shap.log_explanation(rf.predict, X_train[:5])
  
  # Generate feature importance plot
  feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
  fig, ax = plt.subplots()
  feature_importances.plot.bar(ax=ax)
  ax.set_title("Feature importances using MDI")
  ax.set_ylabel("Mean decrease in impurity")
  
  # Log figure
  mlflow.log_figure(fig, "feature_importance_rf.png")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Look at this in the MLflow UI
# MAGIC 
# MAGIC <img style="width:50%" src="https://files.training.databricks.com/images/mlpupdates/artifact_examples.gif" >

# COMMAND ----------

# MAGIC %md ## Additional Resources
# MAGIC 
# MAGIC * [Hyperopt Docs]("http://hyperopt.github.io/hyperopt/")
# MAGIC * [Hyperparamter Tuning Blog post]("https://databricks.com/blog/2019/06/07/hyperparameter-tuning-with-mlflow-apache-spark-mllib-and-hyperopt.html")
# MAGIC * [Spark Trials Hyperopt Documentation]("https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-spark-mlflow-integration.html#how-to-use-hyperopt-with-sparktrials")
# MAGIC * [Documentation here]("https://www.mlflow.org/docs/latest/python_api/mlflow.shap.html")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
