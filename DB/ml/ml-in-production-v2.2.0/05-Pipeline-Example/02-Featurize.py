# Databricks notebook source
# MAGIC 
# MAGIC %md ### Featurize
# MAGIC 
# MAGIC In this notebook we will featurize our validated data by String Encoding the categorical columns and generating feature vectors. 
# MAGIC 
# MAGIC First, let's load in our data passed from Orchestrate. 

# COMMAND ----------

dbutils.widgets.text("filePath", "Default")
dbutils.widgets.text("savePath", "Default")

filePath = dbutils.widgets.get("filePath")
savePath = dbutils.widgets.get("savePath")

# COMMAND ----------

# MAGIC %md Now, we can load in the validated data.

# COMMAND ----------

df = spark.read.format('delta').load(filePath)

# COMMAND ----------

# MAGIC %md Next, we can String Encode the columns and assemble our feature vectors.

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

categoricalCols = [field for (field, dataType) in df.dtypes if dataType == "string"]
indexOutputCols = [x + "Index" for x in categoricalCols]

stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=indexOutputCols, handleInvalid="skip")

numericCols = [field for (field, dataType) in df.dtypes if ((dataType == "double") & (field != "price"))]
assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

stages = [stringIndexer, vecAssembler]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(df)
featurized_df = pipelineModel.transform(df)

# COMMAND ----------

# MAGIC %md Finally, we can save our featurized data to the paths we reserved in Orchestrate.

# COMMAND ----------

featurized_df.write.format("delta").save(savePath)

