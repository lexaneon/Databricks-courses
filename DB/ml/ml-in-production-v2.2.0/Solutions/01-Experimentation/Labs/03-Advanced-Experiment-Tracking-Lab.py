# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
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

# ANSWER
import mlflow 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature
import numpy as np 
import matplotlib.pyplot as plt

with mlflow.start_run(run_name="Parent") as run:
  
  mlflow.log_param("runType", "parent")
  
  signature = infer_signature(X_train, pd.DataFrame(y_train))
  input_example = X_train.head(3)
  
  
  with mlflow.start_run(run_name="Child1", nested=True) as run:
    
    rf = RandomForestRegressor(random_state=42, max_depth=5)
    rf_model = rf.fit(X_train, y_train)
    mse = mean_squared_error(rf_model.predict(X_test), y_test)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("mse", mse)
    
    mlflow.sklearn.log_model(rf_model, "model", signature=signature, input_example=input_example)
    
  with mlflow.start_run(run_name="Child2", nested=True):
    
    rf = RandomForestRegressor(random_state=42, max_depth=10)
    rf_model = rf.fit(X_train, y_train)
    mse = mean_squared_error(rf_model.predict(X_test), y_test)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("mse", mse)
    

    mlflow.sklearn.log_model(rf_model, "model", signature=signature, input_example=input_example)
    
    # Generate feature importance plot
    forest_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
  
    # Log figure
    mlflow.log_figure(fig, "Feature Importances RF.png")

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

# ANSWER
from hyperopt import fmin, tpe, hp, SparkTrials
from hyperopt import SparkTrials

# Define objective function
def objective(params):
  model = RandomForestRegressor(n_estimators=int(params['n_estimators']), max_depth=int(params['max_depth']), min_samples_leaf=int(params['min_samples_leaf']),
                                min_samples_split=int(params['min_samples_split']))
  model.fit(X_train, y_train)
  pred = model.predict(X_test)
  score = mean_squared_error(pred, y_test)
  
  # Hyperopt minimizes score, here we minimize mse. 
  return score

# Define search space
search_space = {'n_estimators':hp.uniform('n_estimators',100,500),
           'max_depth':hp.uniform('max_depth',5,20),
           'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
           'min_samples_split':hp.uniform('min_samples_split',2,6)}

# Set algorithm type
algo=tpe.suggest

# Set spark trials arguement
spark_trials = SparkTrials()


with mlflow.start_run(run_name="Hyperopt"):
  argmin = fmin(
    fn=objective,
    space=search_space,
    algo=algo,
    max_evals=16,
    trials=spark_trials)


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
