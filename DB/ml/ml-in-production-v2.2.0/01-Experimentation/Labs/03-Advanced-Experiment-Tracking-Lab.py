# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Lab: Advanced Experiment Tracking
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Manually log nested runs for hyperparameter tuning
# MAGIC  - Autolog nested runs using hyperopt

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md ## Manual Hyperparameter Tuning
# MAGIC 
# MAGIC Create an mlflow run structured in the following way:
# MAGIC 
# MAGIC * Create a parent run named "Parent"
# MAGIC * In this parent run:
# MAGIC   * Train an sklearn RandomForestRegressor on the `X_train` and `y_train` given
# MAGIC   * Get the signature and an input example. (Get the signature with `infer_signature`)
# MAGIC * Created a nested run named "Child 1"
# MAGIC * In "Child 1":
# MAGIC   * Train an sklearn RandomForestRegressor on the `X_train` and `y_train` given with a max_depth of 5
# MAGIC   * Log a parameter "max_depth" of 5
# MAGIC   * Log the mse
# MAGIC   * Log the model with input example and signature 
# MAGIC * Back in the parent run (no longer in "Child 1") create another nested run named "Child 2"
# MAGIC * In "Child 2":
# MAGIC   * Train an sklearn RandomForestRegressor on the `X_train` and `y_train` given with a max_depth of 10
# MAGIC   * Log a parameter "max_depth" of 10
# MAGIC   * Log the mse
# MAGIC   * Log the model with input example and signature 
# MAGIC   * Get the feature importance plot as a pyplot figure for the model 
# MAGIC     * Check the lesson for how to do this if you need a hint.
# MAGIC   * Log the plot as a figure as shown in the lesson

# COMMAND ----------

from sklearn.model_selection import train_test_split
import pandas as pd

pdf = pd.read_parquet("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
X = pdf.drop('price', axis=1)
y = pdf['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# TODO


# COMMAND ----------

# MAGIC %md ## Autologging with Hyperopt
# MAGIC 
# MAGIC In this exercise, you will hyperparamter tune an sklearn regression model with Hyperopt like we did in the lesson. 
# MAGIC 
# MAGIC This is the most common use of nested runs and it is good to make sure you can run Hyperopt, know where to find the nested runs, and compare them with visualizations. 
# MAGIC 
# MAGIC For this exercise:
# MAGIC 
# MAGIC 1. Run Hyperopt hyperparamter tuning on a sklearn Random Forest Regressor on the dataset given, just like we did in the lesson.
# MAGIC   * If you get stuck, refer back to the lesson.
# MAGIC 2. Find the nested runs in the MLflow UI
# MAGIC 3. Generate the Parallel Coordinates Plot as shown in the lesson on your nested runs. 
# MAGIC 
# MAGIC **Note:** You will need to select all nested runs and hit compare in the MLflow UI. If you select the bottom-most nested run and then shift-click the top-most nested run, you will select all of them.

# COMMAND ----------

# TODO


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
