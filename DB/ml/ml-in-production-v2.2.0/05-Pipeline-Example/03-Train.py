# Databricks notebook source
# MAGIC 
# MAGIC %md ### Train
# MAGIC 
# MAGIC This notebook is called by Orchestrate to train a model on our first time period of featurized data. 
# MAGIC 
# MAGIC Let's first import the packages we need.

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# COMMAND ----------

# MAGIC %md Now, let's load the data from Orchestrate.

# COMMAND ----------

dbutils.widgets.text("filePath", "Default")
dbutils.widgets.text("experiment_path", "Default")
dbutils.widgets.text("registry_model_name", "Default")


filePath = dbutils.widgets.get("filePath")
experiment_path = dbutils.widgets.get("experiment_path")
registry_model_name = dbutils.widgets.get("registry_model_name")

# COMMAND ----------

# MAGIC %md Set the experiment from the data we loaded.

# COMMAND ----------

mlflow.set_experiment(experiment_path)

# COMMAND ----------

# MAGIC %md Load the data to train.

# COMMAND ----------

df = spark.read.format("delta").load(filePath)

# COMMAND ----------

# MAGIC %md Create our example model, making sure to log signature, input example, train mse, and test mse.

# COMMAND ----------

trainDF, testDF = df.randomSplit([0.9, 0.1], seed=42)

# COMMAND ----------

with mlflow.start_run() as run:
  params = {
      "numTrees": 100,
      "cacheNodeIds": True,
      "maxDepth": 30,
      "minInstancesPerNode": 100,
      "maxBins": 40
    }
  
  # Need get all columns that are not price, Index
  cols = [trainDF.dtypes[i][0] for i in range(len(trainDF.dtypes)) if trainDF.dtypes[i][1] != 'vector' and trainDF.dtypes[i][0] != 'price' and trainDF.dtypes[i][0][-5:] != 'Index']
  input_example = trainDF.select(cols).head(3)
  sig = infer_signature(trainDF.select(cols), trainDF.select("price"))
  
  regressor = RandomForestRegressor(featuresCol="features", labelCol="price", **params)
  
  rf_model = regressor.fit(trainDF)
  
  evaluator = RegressionEvaluator(metricName="mse", labelCol="price")
  
  train_mse = evaluator.evaluate(rf_model.transform(trainDF))
  test_mse = evaluator.evaluate(rf_model.transform(testDF))
  
  mlflow.log_metric("test_mse", test_mse)
  mlflow.log_metric("train_mse", train_mse)
  mlflow.spark.log_model(rf_model, "model", input_example=input_example, signature=sig)

# COMMAND ----------

# MAGIC %md Finally, we'll push our model to the Staging branch in the model registry.

# COMMAND ----------

client = MlflowClient()
model_version = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=registry_model_name)
model_version = client.transition_model_version_stage(
  name=model_version.name, 
  version=model_version.version, 
  stage="Staging", 
  archive_existing_versions=True
)

