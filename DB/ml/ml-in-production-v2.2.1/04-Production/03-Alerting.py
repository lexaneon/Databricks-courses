# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Alerting
# MAGIC 
# MAGIC Alerting allows you to announce the progress of different applications, which becomes increasingly important in automated production systems.  In this lesson, you explore basic alerting strategies using email and REST integration with tools like Slack and Microsoft Teams.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Explore the alerting landscape
# MAGIC  - Create a basic REST alert integrated with Slack or Microsoft Teams
# MAGIC  - Create a more complex REST alert for Spark jobs using `SparkListener`

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md 
# MAGIC ### The Alerting Landscape
# MAGIC 
# MAGIC There are a number of different alerting tools with various levels of sophistication.<br><br>
# MAGIC * PagerDuty 
# MAGIC  - has become one of the most popular tools for monitoring production outages
# MAGIC  - allows for the escalation of issues across a team with alerts including text messages and phone calls
# MAGIC * Slack or Microsoft Teams
# MAGIC * Twilio   
# MAGIC * Email alerts
# MAGIC 
# MAGIC Most alerting frameworks allow for custom alerting done through REST integration.
# MAGIC 
# MAGIC One additional helpful tool for Spark workloads:
# MAGIC * the `SparkListener`
# MAGIC * can perform custom logic on various Cluster actions

# COMMAND ----------

# MAGIC %md ### Setting Basic Alerts
# MAGIC 
# MAGIC Create a basic alert using a Slack or Microsoft Teams endpoint.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC To set up a Microsoft Teams endpoint, do the following:<br><br>
# MAGIC 
# MAGIC 1. After setting up teams, click on the **Teams** tab.
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams1.png" style="height: 200px; margin: 20px"/></div>
# MAGIC 2. Click the dropdown next to the team you want to associate the endpoint to (create a new team if you don't have one already).  Then click **Connectors**. <br></br>
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams2.png" style="height: 350px; margin: 20px"/></div>
# MAGIC 3. Choose **Configure** next to **Incoming Webhook**. <br></br>
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams3.png" style="height: 250px; margin: 20px"/></div>
# MAGIC 4. Give the webhook a name and click **Create**. <br></br>
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams4.png" style="height: 250px; margin: 20px"/></div>
# MAGIC 5. Copy the URL and paste it below.
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams5.png" style="height: 250px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md 
# MAGIC Define a Slack webhook.  This has **not** been done for you.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Define your own Slack webhook <a href="https://api.slack.com/incoming-webhooks#getting-started" target="_blank">Using these 4 steps.</a><br>
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> This same approach applies to PagerDuty as well.

# COMMAND ----------

# MAGIC %md
# MAGIC // INSTRUCTOR_NOTE
# MAGIC Follow the directions here to enable this on your own: https://api.slack.com/incoming-webhooks#
# MAGIC 
# MAGIC Note that this lesson could be rewritten using AWS API Gateway and Lambda/Kinesis to have an API that would display the latest N requests to the endpoint.

# COMMAND ----------

webhookMLProductionAPIDemo = "" # FILL_IN

# COMMAND ----------

# MAGIC %md Send a test message and check Slack.

# COMMAND ----------

def postToAPIEndpoint(content, webhook=""):
  '''
  Post message to Teams to log progress
  '''
  import requests
  from requests.exceptions import MissingSchema
  from string import Template
  
  t = Template('{"text": "${content}"}')
  
  try:
    response = requests.post(webhook, data=t.substitute(content=content), headers={'Content-Type': 'application/json'})
    return response
  except MissingSchema:
    print("Please define an appropriate API endpoint use by defining the `webhook` argument")

postToAPIEndpoint("This is my post from Python", webhookMLProductionAPIDemo)

# COMMAND ----------

# MAGIC %md Do the same thing using Scala.  This involves a bit more boilerplate and a different library.

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC val webhookMLProductionAPIDemo = "" // FILL_IN

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC def postToAPIEndpoint(content:String, webhook:String):Unit = {
# MAGIC   import org.apache.http.entity._
# MAGIC   import org.apache.http.impl.client.{HttpClients}
# MAGIC   import org.apache.http.client.methods.HttpPost
# MAGIC 
# MAGIC   val client = HttpClients.createDefault()
# MAGIC   val httpPost = new HttpPost(webhook)
# MAGIC   
# MAGIC   val payload = s"""{"text": "${content}"}"""
# MAGIC 
# MAGIC   val entity = new StringEntity(payload)
# MAGIC   httpPost.setEntity(entity)
# MAGIC   httpPost.setHeader("Accept", "application/json")
# MAGIC   httpPost.setHeader("Content-type", "application/json")
# MAGIC   
# MAGIC   try {
# MAGIC     val response = client.execute(httpPost)
# MAGIC     client.close()
# MAGIC   } catch {
# MAGIC     case cPE: org.apache.http.client.ClientProtocolException => println("Please define an appropriate API endpoint to hit by defining the variable webhookMLProductionAPIDemo")
# MAGIC   } 
# MAGIC }
# MAGIC 
# MAGIC postToAPIEndpoint("This is my post from Scala", webhookMLProductionAPIDemo)

# COMMAND ----------

# MAGIC %md Now you can easily integrate custom logic back to Slack.

# COMMAND ----------

mse = .45

postToAPIEndpoint(webhookMLProductionAPIDemo, "The newly trained model MSE is now {}".format(mse))

# COMMAND ----------

# MAGIC %md ### Using a `SparkListener`
# MAGIC 
# MAGIC A custom `SparkListener` allows for custom actions taken on cluster activity.  **This API is only available in Scala.**  Take a look at the following code.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> See [SparkListener](https://spark.apache.org/docs/latest/api/java/index.html?org/apache/spark/scheduler/SparkListener.html) docs</a>. You can refer to this [doc](https://kb.databricks.com/metrics/explore-spark-metrics.html).
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_warn_24.png"/> Be sure to update the webhook variable in the following cell.

# COMMAND ----------

# MAGIC %scala
# MAGIC // Package in a notebook helps to ensure a proper singleton
# MAGIC package com.databricks.academy
# MAGIC 
# MAGIC object SlackNotifyingListener extends org.apache.spark.scheduler.SparkListener {
# MAGIC   import org.apache.spark.scheduler._
# MAGIC 
# MAGIC   val webhook = "" // FILL_IN
# MAGIC   
# MAGIC   def postToAPIEndpoint(content:String):Unit = {
# MAGIC     import org.apache.http.entity._
# MAGIC     import org.apache.http.impl.client.{HttpClients}
# MAGIC     import org.apache.http.client.methods.HttpPost
# MAGIC 
# MAGIC     val client = HttpClients.createDefault()
# MAGIC     val httpPost = new HttpPost(webhook)
# MAGIC 
# MAGIC     val payload = s"""{"text": "${content}"}"""
# MAGIC 
# MAGIC     val entity = new StringEntity(payload)
# MAGIC     httpPost.setEntity(entity)
# MAGIC     httpPost.setHeader("Accept", "application/json")
# MAGIC     httpPost.setHeader("Content-type", "application/json")
# MAGIC 
# MAGIC     try {
# MAGIC       val response = client.execute(httpPost)
# MAGIC       client.close()
# MAGIC     } catch {
# MAGIC       case cPE: org.apache.http.client.ClientProtocolException => println("Please define an appropriate API endpoint to hit by defining the variable webhookMLProductionAPIDemo")
# MAGIC     } 
# MAGIC   }
# MAGIC   
# MAGIC   override def onApplicationEnd(applicationEnd: SparkListenerApplicationEnd): Unit = {
# MAGIC     postToAPIEndpoint("Called when the application ends")
# MAGIC   }
# MAGIC 
# MAGIC   override def onApplicationStart(applicationStart: SparkListenerApplicationStart): Unit = {
# MAGIC     postToAPIEndpoint("Called when the application starts")
# MAGIC   }
# MAGIC 
# MAGIC   override def onBlockManagerAdded(blockManagerAdded: SparkListenerBlockManagerAdded): Unit = {
# MAGIC     postToAPIEndpoint("Called when a new block manager has joined")
# MAGIC   }
# MAGIC 
# MAGIC   override def onBlockManagerRemoved(blockManagerRemoved: SparkListenerBlockManagerRemoved): Unit = {
# MAGIC     postToAPIEndpoint("Called when an existing block manager has been removed")
# MAGIC   }
# MAGIC 
# MAGIC   override def onBlockUpdated(blockUpdated: SparkListenerBlockUpdated): Unit = {
# MAGIC     postToAPIEndpoint("Called when the driver receives a block update info.")
# MAGIC   }
# MAGIC 
# MAGIC   override def onEnvironmentUpdate(environmentUpdate: SparkListenerEnvironmentUpdate): Unit = {
# MAGIC     postToAPIEndpoint("Called when environment properties have been updated")
# MAGIC   }
# MAGIC 
# MAGIC   override def onExecutorAdded(executorAdded: SparkListenerExecutorAdded): Unit = {
# MAGIC     postToAPIEndpoint("Called when the driver registers a new executor.")
# MAGIC   }
# MAGIC 
# MAGIC   override def onExecutorBlacklisted(executorBlacklisted: SparkListenerExecutorBlacklisted): Unit = {
# MAGIC     postToAPIEndpoint("Called when the driver blacklists an executor for a Spark application.")
# MAGIC   }
# MAGIC 
# MAGIC   override def onExecutorBlacklistedForStage(executorBlacklistedForStage: SparkListenerExecutorBlacklistedForStage): Unit = {
# MAGIC     postToAPIEndpoint("Called when the driver blacklists an executor for a stage.")
# MAGIC   }
# MAGIC 
# MAGIC   override def onExecutorMetricsUpdate(executorMetricsUpdate: SparkListenerExecutorMetricsUpdate): Unit = {
# MAGIC     // This one is a bit on the noisy side so I'm pre-emptively killing it
# MAGIC     // postToSlack("Called when the driver receives task metrics from an executor in a heartbeat.")
# MAGIC   }
# MAGIC 
# MAGIC   override def onExecutorRemoved(executorRemoved: SparkListenerExecutorRemoved): Unit = {
# MAGIC     postToAPIEndpoint("Called when the driver removes an executor.")
# MAGIC   }
# MAGIC 
# MAGIC   override def onExecutorUnblacklisted(executorUnblacklisted: SparkListenerExecutorUnblacklisted): Unit = {
# MAGIC     postToAPIEndpoint("Called when the driver re-enables a previously blacklisted executor.")
# MAGIC   }
# MAGIC 
# MAGIC   override def onJobEnd(jobEnd: SparkListenerJobEnd): Unit = {
# MAGIC     postToAPIEndpoint("Called when a job ends")
# MAGIC   }
# MAGIC 
# MAGIC   override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
# MAGIC     postToAPIEndpoint("Called when a job starts")
# MAGIC   }
# MAGIC 
# MAGIC   override def onNodeBlacklisted(nodeBlacklisted: SparkListenerNodeBlacklisted): Unit = {
# MAGIC     postToAPIEndpoint("Called when the driver blacklists a node for a Spark application.")
# MAGIC   }
# MAGIC 
# MAGIC   override def onNodeBlacklistedForStage(nodeBlacklistedForStage: SparkListenerNodeBlacklistedForStage): Unit = {
# MAGIC     postToAPIEndpoint("Called when the driver blacklists a node for a stage.")
# MAGIC   }
# MAGIC 
# MAGIC   override def onNodeUnblacklisted(nodeUnblacklisted: SparkListenerNodeUnblacklisted): Unit = {
# MAGIC     postToAPIEndpoint("Called when the driver re-enables a previously blacklisted node.")
# MAGIC   }
# MAGIC 
# MAGIC   override def onOtherEvent(event: SparkListenerEvent): Unit = {
# MAGIC     postToAPIEndpoint("Called when other events like SQL-specific events are posted.")
# MAGIC   }
# MAGIC 
# MAGIC   override def onSpeculativeTaskSubmitted(speculativeTask: SparkListenerSpeculativeTaskSubmitted): Unit = {
# MAGIC     postToAPIEndpoint("Called when a speculative task is submitted")
# MAGIC   }
# MAGIC 
# MAGIC   override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = {
# MAGIC     postToAPIEndpoint("Called when a stage completes successfully or fails, with information on the completed stage.")
# MAGIC   }
# MAGIC 
# MAGIC   override def onStageSubmitted(stageSubmitted: SparkListenerStageSubmitted): Unit = {
# MAGIC     postToAPIEndpoint("Called when a stage is submitted")
# MAGIC   }
# MAGIC 
# MAGIC   override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
# MAGIC     postToAPIEndpoint("Called when a task ends")
# MAGIC   }
# MAGIC 
# MAGIC   override def onTaskGettingResult(taskGettingResult: SparkListenerTaskGettingResult): Unit = {
# MAGIC     postToAPIEndpoint("Called when a task begins remotely fetching its result (will not be called for tasks that do not need to fetch the result remotely).")
# MAGIC   }
# MAGIC 
# MAGIC   override def onTaskStart(taskStart: SparkListenerTaskStart): Unit = {
# MAGIC     postToAPIEndpoint("Called when a task starts")
# MAGIC   }
# MAGIC 
# MAGIC   override def onUnpersistRDD(unpersistRDD: SparkListenerUnpersistRDD): Unit = {
# MAGIC     postToAPIEndpoint("Called when an RDD is manually unpersisted by the application")
# MAGIC   }
# MAGIC }

# COMMAND ----------

# MAGIC %md Register this Singleton as a `SparkListener`

# COMMAND ----------

# MAGIC %scala
# MAGIC sc.addSparkListener(com.databricks.academy.SlackNotifyingListener)

# COMMAND ----------

# MAGIC %md Now run a basic DataFrame operation and observe the results in Slack.

# COMMAND ----------

# MAGIC %scala
# MAGIC spark.read
# MAGIC   .option("header", true)
# MAGIC   .option("inferSchema", true)
# MAGIC   .parquet("/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
# MAGIC   .count

# COMMAND ----------

# MAGIC %md This will also work in Python.

# COMMAND ----------

(spark.read
  .option("header", True)
  .option("inferSchema", True)
  .parquet("/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
  .count()
)

# COMMAND ----------

# MAGIC %md When this is done, remove the listener.

# COMMAND ----------

# MAGIC %scala
# MAGIC sc.removeSparkListener(com.databricks.academy.SlackNotifyingListener)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Review
# MAGIC **Question:** What are the most common alerting tools?  
# MAGIC **Answer:** PagerDuty tends to be the tool most used in production environments.  SMTP servers emailing alerts are also popular, as is Twilio for text message alerts.  Slack webhooks and bots can easily be written as well.
# MAGIC 
# MAGIC **Question:** How can I write custom logic to monitor Spark?  
# MAGIC **Answer:** The `SparkListener` API is only exposed in Scala.  This allows you to write custom logic based on your cluster activity.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Steps
# MAGIC 
# MAGIC Start the next lesson, [Pipeline Example]($../05-Pipeline-Example/00-Orchestrate)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find the alerting tools mentioned in this lesson?  
# MAGIC **A:** Check out <a href="https://www.twilio.com" target="_blank">Twilio</a> and <a href="https://www.pagerduty.com" target="_blank">PagerDuty</a>.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
