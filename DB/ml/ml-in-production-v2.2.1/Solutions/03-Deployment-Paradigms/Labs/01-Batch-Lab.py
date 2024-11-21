# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab: Deploying a Model in Batch
# MAGIC Deploying a model via batch is the preferred solution for most machine learning applications. In this lab, you will scale the deployment of a single-node model via Spark UDFs and MLflow's `pyfunc`. 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Develop and register an MLflow model
# MAGIC  - Deploy the model as a Spark UDF
# MAGIC  - Optimize the predictions for reading using Delta

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC Before we begin, we'll need to load our data and split it into a modeling set and an inference set.

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

# Load data
df = (spark
      .read
      .parquet("/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
      .select("price", "bathrooms", "bedrooms", "number_of_reviews", "neighbourhood_cleansed")
      .withColumn("listing_id", monotonically_increasing_id()))

# Split into modeling and inference sets
modeling_df, inference_df = df.randomSplit([0.5, 0.5], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Develop and Register an MLflow Model
# MAGIC 
# MAGIC In this exercise, you will build, log, and register an XGBoost model using Scikit-learn and MLflow.
# MAGIC 
# MAGIC This model will predict the `price` variable using `bathrooms`, `bedrooms`, and `number_of_reviews` as features.

# COMMAND ----------

# ANSWER
import xgboost as xgb
import mlflow
import mlflow.xgboost
from numpy import mean
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error

# Start run
with mlflow.start_run(run_name="xgboost_model") as run:
    
    # Split data
    modeling_pdf = modeling_df.toPandas()
    X_train, X_test, y_train, y_test = train_test_split(
        modeling_pdf.drop(["price", "neighbourhood_cleansed", "listing_id"], axis=1), 
        modeling_pdf[["price"]].values.ravel(), 
        random_state=42
    )
    
    # Train model
    regressor = xgb.XGBRegressor(n_estimators=10, max_depth=10)
    regressor.fit(X_train, y_train)
    
    # Evaluate model
    train_predictions = regressor.predict(X_train)
    train_rmse = mean_squared_error(train_predictions, y_train, squared = False)
    
    # Log params and metric
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("n_estimators", 10)
    mlflow.log_metric("train_rmse", train_rmse)
    
    # Log model
    mlflow.xgboost.log_model(regressor, "xgboost-model")
    
# Register model
model_name = f"xgboost_model_{clean_username}"
model_uri = f"runs:/{run.info.run_id}/xgboost-model"
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy Model as a Spark UDF
# MAGIC 
# MAGIC Next, you will compute predictions for your model using a Spark UDF.

# COMMAND ----------

# ANSWER
# Create the prediction UDF
predict = mlflow.pyfunc.spark_udf(spark, run.info.artifact_uri + "/xgboost-model")

# Compute the predictions
prediction_df = inference_df.withColumn(
    "prediction", 
    predict(*modeling_pdf.drop(["price", "neighbourhood_cleansed", "listing_id"], axis=1).columns)
).select("listing_id", "neighbourhood_cleansed", "prediction")

# View the results
display(prediction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimize the predictions for reading using Delta
# MAGIC 
# MAGIC Now that the predictions are computed, you'll want to write them so they can be accessed efficiently.
# MAGIC 
# MAGIC There are a variety of features to take advantage of, but we'll just partition our written files by the `neighbourhood_cleansed` column here.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> If needed, you can overwrite the file.

# COMMAND ----------

# ANSWER
delta_partitioned_path = f"{working_dir}/batch-predictions-partitioned-lab.delta"

(prediction_df
    .write
    .partitionBy("neighbourhood_cleansed")
    .mode("overwrite")
    .format("delta")
    .save(delta_partitioned_path))

# COMMAND ----------

# MAGIC %md
# MAGIC Great! Now other data scientists, data analysts, data engineers, and machine learning engineers can quickly access the predictions for this model.
# MAGIC 
# MAGIC If you'd like to reduce the latency even more, check out <a href="$../../06-Cloud-Examples/03-Azure-Batch-Deployment" target="_blank">the Azure-specific example of writing predictions to CosmosDB</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Lesson<br>
# MAGIC 
# MAGIC Start the next lesson, [Real Time]($../02-Real-Time )

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
