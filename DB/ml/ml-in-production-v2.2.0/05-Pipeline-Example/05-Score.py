# Databricks notebook source
# MAGIC 
# MAGIC %md ### Score
# MAGIC 
# MAGIC This notebook is called from Orchestrate to score a new dataset with the current production model's predictions, and then store it in the given path. 

# COMMAND ----------

# MAGIC %md Let's import what we need and load the data passed from Orchestrate.

# COMMAND ----------

import mlflow
from sklearn.metrics import mean_squared_error
from mlflow.tracking import MlflowClient

# COMMAND ----------

dbutils.widgets.text("storePath", "Default")
dbutils.widgets.text("readPath", "Default")
dbutils.widgets.text("registry_model_name", "Default")

data_to_score = dbutils.widgets.get("readPath")
store_path = dbutils.widgets.get("storePath")
registry_model_name = dbutils.widgets.get("registry_model_name")

# COMMAND ----------

# MAGIC %md Next, let's get the current production model and load the dataset.

# COMMAND ----------

model = mlflow.spark.load_model(model_uri=f"models:/{registry_model_name}/Production")

# COMMAND ----------

df = spark.read.format('delta').load(data_to_score)

# COMMAND ----------

# MAGIC %md Finally, let's save the predictions to a new column and store the new dataframe in the new store path.

# COMMAND ----------

scored_df = model.transform(df)
display(scored_df)

# COMMAND ----------

scored_df.write.format('delta').save(store_path)

