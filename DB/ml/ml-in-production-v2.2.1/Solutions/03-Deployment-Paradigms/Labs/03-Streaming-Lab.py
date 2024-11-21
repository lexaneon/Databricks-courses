# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab: Post-Processing on a Data Stream
# MAGIC Machine learning predictions often require custom pre-processing or post-processing logic beyond what a serialized, trained model allows.  This lab uses the `mlflow` custom Python functionality to apply custom prediction logic across a stream of data.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Create a data stream and train a random forest model
# MAGIC  - Defining post-processing logic 
# MAGIC  - Apply post-processing logic to a data stream
# MAGIC  - Write a DataFrame to a scalable Delta format

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating Data Stream and Training Model
# MAGIC 
# MAGIC Create a data stream and train a random forest model.  First, run the following cell to define the scmea and set the shuffle partitions.

# COMMAND ----------

from pyspark.sql.types import DoubleType, IntegerType, StructType

schema = (StructType()
.add("host_total_listings_count", DoubleType())
.add("neighbourhood_cleansed", IntegerType())
.add("zipcode", IntegerType())
.add("latitude", DoubleType())
.add("longitude", DoubleType())
.add("property_type", IntegerType())
.add("room_type", IntegerType())
.add("accommodates", DoubleType())
.add("bathrooms", DoubleType())
.add("bedrooms", DoubleType())
.add("beds", DoubleType())
.add("bed_type", IntegerType())
.add("minimum_nights", DoubleType())
.add("number_of_reviews", DoubleType())
.add("review_scores_rating", DoubleType())
.add("review_scores_accuracy", DoubleType())
.add("review_scores_cleanliness", DoubleType())
.add("review_scores_checkin", DoubleType())
.add("review_scores_communication", DoubleType())
.add("review_scores_location", DoubleType())
.add("review_scores_value", DoubleType())
.add("price", DoubleType())
)

spark.conf.set("spark.sql.shuffle.partitions", "8")

# COMMAND ----------

# MAGIC %md Create the stream using the schema defined above.

# COMMAND ----------

# ANSWER
streamingData = (spark
                 .readStream
                 .schema(schema)
                 .option("maxFilesPerTrigger", 1)
                 .parquet("/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
                 .drop("price"))

# COMMAND ----------

# MAGIC %md And to help us manage our streams better, we will make use of **`untilStreamIsReady()`**, **`stopAllStreams()`** and define the following, **`myStreamName`**:

# COMMAND ----------

myStreamName = "lesson03"

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to import the static airbnb data and train a random forest model `rf` for making price predictions.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_parquet("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

# new random forest model
rf = RandomForestRegressor(n_estimators=100, max_depth=25)

# fit and evaluate new rf model
rf.fit(X_train, y_train)

# COMMAND ----------

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

# df = (spark
#   .read
#   .parquet("/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
#   .toPandas()
# )
# X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

# # new random forest model
# rf = RandomForestRegressor(n_estimators=100, max_depth=25)

# # fit and evaluate new rf model
# rf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Post-Processing Logic
# MAGIC 
# MAGIC When processing our data stream, we are interested in seeing whether each predicted price is "High", "Medium", or "Low". To accomplish this, we are going to define a model class that will apply the desired post-processing step to our random forest `rf`'s results with a `.predict()` call.
# MAGIC 
# MAGIC Complete the `postprocess_result()` method to change the predicted value from a number to one of 3 categorical labels, "High", "Medium", or "Low". Then finish the line in `predict()` to return the desired output.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> This can be done in pure Python where you apply the logic to `results` and return the transformed data in a list.  For a more performant solution, use broadcasting on a `pandas` series or DataFrame.
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Check out <a href="https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-create-custom" target="_blank">the `pyfunc` documentation for details</a><br>
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Check out <a href="https://github.com/mlflow/mlflow/blob/master/docs/source/models.rst#custom-python-models" target="_blank">the example code in the `mlflow` github repository</a><br>

# COMMAND ----------

# ANSWER
import mlflow
from mlflow.pyfunc import PythonModel

# Define the model class
class streaming_model(PythonModel):

    def __init__(self, trained_rf):
        self.rf = trained_rf

    def postprocess_result(self, results):
        '''return post-processed results
        High: predicted price >= 120
        Medium: predicted price < 120 and >= 70
        Low: predicted price < 70'''
        output = []
        for result in results:
          if result >= 120:
            output.append("High")
          elif result >= 70:
            output.append("Medium")
          else:
            output.append("Low")
        return output
    
    def predict(self, context, model_input):
        results = self.rf.predict(model_input)
        return self.postprocess_result(results)

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to perform the following:<br><br>
# MAGIC 
# MAGIC 1. Create a path where you will save the model
# MAGIC 2. Instantiate the class using the trained random forest model
# MAGIC 3. Save the model
# MAGIC 4. Load the model as a Python function
# MAGIC 5. Apply the model to the data using the `.predict()` function.
# MAGIC 
# MAGIC You should see a list of price labels output underneath the cell.

# COMMAND ----------

import shutil

# Construct and save the model
try:
  model_path = (f"{working_dir}/ml-production/streaming_model/").replace("dbfs:", "/dbfs")
  shutil.rmtree(model_path)
except:
  print("Cannot delete directory.")

# Instantiate the class using the trained model
model = streaming_model(trained_rf=rf)

# Save the model
mlflow.pyfunc.save_model(path=model_path, python_model=model)

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(model_path)

# Apply the model
loaded_model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Post-Processing Step to Data Stream
# MAGIC 
# MAGIC Finally, after confirming that your model works properly, apply it in parallel on all rows of `streamingData`.

# COMMAND ----------

# ANSWER
import mlflow.pyfunc

# Load the model in as a spark UDF
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_path, result_type="string")

# Apply UDF to data stream
predictionsDF = streamingData.withColumn("prediction", pyfunc_udf(*streamingData.columns))

display(predictionsDF, streamName=myStreamName)

# COMMAND ----------

# Wait until the stream is ready
from time import sleep

sleep(10)

# COMMAND ----------

# When you are done previewing the results, stop the stream.
for stream in spark.streams.active:
  print(f"Stopping {stream.name}")
  stream.stop() # Stop the stream

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Write DataFrame to Delta

# COMMAND ----------

# MAGIC %md
# MAGIC Now continuously write `predictionsDF` to a parquet file as they get created by the model.

# COMMAND ----------

# ANSWER
checkpointLocation = f"{working_dir}/stream-lab4.checkpoint"
writePath = f"{working_dir}/lab4-predictions"

(streamingData
  .writeStream                                           # Write the stream
  .queryName(myStreamName)                               # Name the stream
  .format("delta")                                       # Use the delta format
  .option("checkpointLocation", checkpointLocation)      # Specify where to log metadata
  .option("path", writePath)                             # Specify the output path
  .outputMode("append")                                  # Append new records to the output path
  .start()                                               # Start the operation
)

# COMMAND ----------

# Wait until the stream is ready
from time import sleep

sleep(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Check that your predictions are indeed being written out to `writePath`.

# COMMAND ----------

from pyspark.sql.utils import AnalysisException

try:
  print(spark.read.format("delta").load(writePath).count())
  
except AnalysisException:
  print("Files not found.  This could be because the stream hasn't initialized.  Try again in a moment.")

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to terminate all active streams.

# COMMAND ----------

# Stop all of our streams
for stream in spark.streams.active:
  print(f"Stopping {stream.name}")
  stream.stop() # Stop the stream

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Lesson<br>
# MAGIC 
# MAGIC Start the next lesson, [CI/CD]($../../04-Production/01-CICD)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
