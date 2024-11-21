# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Lab: Grid Search with MLflow
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Import the housing data
# MAGIC  - Perform grid search using scikit-learn
# MAGIC  - Log the best model on MLflow
# MAGIC  - Load the saved model
# MAGIC  - Log the Delta version and hash

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md ## Data Import
# MAGIC 
# MAGIC Load in same Airbnb data and create train/test split.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)

# COMMAND ----------

# MAGIC %md ## Perform Grid Search using scikit-learn
# MAGIC 
# MAGIC We want to know which combination of hyperparameter values is the most effective. Fill in the code below to perform <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV" target="_blank"> grid search using `sklearn`</a>.
# MAGIC 
# MAGIC Set `n_estimators` to `[50, 100]` and `max_depth` to `[3, 5]`.

# COMMAND ----------

# ANSWER
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# dictionary containing hyperparameter names and list of values we want to try
parameters = {'n_estimators': [50, 100], 
              'max_depth': [3, 5]}

rf = RandomForestRegressor()
grid_rf_model = GridSearchCV(rf, parameters, cv=3)
grid_rf_model.fit(X_train, y_train)

best_rf = grid_rf_model.best_estimator_
for p in parameters:
  print("Best '{}': {}".format(p, best_rf.get_params()[p]))

# COMMAND ----------

# MAGIC %md ## Log Best Model with MLflow
# MAGIC 
# MAGIC Log the best model as `grid-random-forest-model`, its parameters, and its MSE metric under a run with name `RF-Grid-Search` in our new MLflow experiment.

# COMMAND ----------

# ANSWER
from sklearn.metrics import mean_squared_error

with mlflow.start_run(run_name="RF-Grid-Search") as run:
  # Create predictions of X_test using best model
  predictions = best_rf.predict(X_test)
  
  # Log model
  mlflow.sklearn.log_model(best_rf, "grid-random-forest-model")
  
  # Log params
  model_params = best_rf.get_params()
  [mlflow.log_param(p, model_params[p]) for p in parameters]
  
  # Create and log MSE metrics using predictions of X_test and its actual value y_test
  mse = mean_squared_error(y_test, predictions)
  mlflow.log_metric("mse", mse)
  
  runID = run.info.run_uuid
  artifactURI = mlflow.get_artifact_uri()
  print(f"Inside MLflow Run with id {runID} and artifact URI {artifactURI}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Load the Saved Model
# MAGIC 
# MAGIC Load the trained and tuned model we just saved. Check that the hyperparameters of this model matches that of the best model we found earlier.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Use the `artifactURI` variable declared above.

# COMMAND ----------

# ANSWER
model = mlflow.sklearn.load_model(artifactURI + "/grid-random-forest-model/")

# COMMAND ----------

# MAGIC %md Time permitting, use the `MlflowClient` to interact programatically with your run.

# COMMAND ----------

# MAGIC %md ### Delta Version and Hash
# MAGIC 
# MAGIC We want to make sure we have data reproducibility. Run the following cell to create the delta path for our delta table. 

# COMMAND ----------

delta_path = working_dir.replace("/dbfs", "dbfs:") + "/delta-example"
dbutils.fs.rm(delta_path, recurse=True)

# COMMAND ----------

# MAGIC %md Save our data to the delta path. We will have to create a spark DataFrame of our DataFrame first to write to delta.

# COMMAND ----------

spark.createDataFrame(df).write.format("delta").mode("overwrite").save(delta_path)

# COMMAND ----------

# MAGIC %md Load back the table and drop columns.

# COMMAND ----------

delta_df = (spark.read
  .format("delta")
  .load(delta_path)
  .drop("cancellation_policy", "instant_bookable")
)

# COMMAND ----------

# MAGIC %md Overwrite our delta path with the dropped columns and view the history of the delta table.

# COMMAND ----------

delta_df.write.format("delta").mode("overwrite").save(delta_path)

display(spark.sql(f"DESCRIBE HISTORY delta.`{delta_path}`"))

# COMMAND ----------

# MAGIC %md Log the best model we found with the following additional values:<br><br>
# MAGIC 
# MAGIC - A copy of the dataset
# MAGIC - The delta version that was used (before we dropped columns)
# MAGIC - A hash of the data

# COMMAND ----------

# ANSWER
import hashlib
from sklearn.metrics import mean_squared_error

with mlflow.start_run(run_name="RF-Grid-Search") as run:
  # Create predictions of X_test using best model
  predictions = best_rf.predict(X_test)
  
  # Log model
  mlflow.sklearn.log_model(best_rf, "grid-random-forest-model")
  
  # Log params
  model_params = best_rf.get_params()
  [mlflow.log_param(p, model_params[p]) for p in parameters]
  
  # Create and log MSE metrics using predictions of X_test and its actual value y_test
  mse = mean_squared_error(y_test, predictions)
  mlflow.log_metric("mse", mse)
  
  # Log a version of the data
  mlflow.log_artifact(delta_path.replace("dbfs:", "/dbfs"), "data")  
  
  # Log the delta version number
  version = 0 
  mlflow.log_param("delta_version", version)
  
  # Create and log the hash
  hash_value = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
  mlflow.log_param("hash", hash_value)


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> See the solutions folder for an example solution to this lab.

# COMMAND ----------

# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Steps
# MAGIC 
# MAGIC Start the next lesson, [Advanced Experiment Tracking]($../03-Advanced-Experiment-Tracking).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
