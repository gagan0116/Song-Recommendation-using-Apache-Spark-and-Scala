// Databricks notebook source
// MAGIC %scala
// MAGIC import org.apache.spark.sql.SparkSession
// MAGIC

// COMMAND ----------

// MAGIC %scala
// MAGIC val spark = SparkSession.builder()
// MAGIC   .appName("Spotify Song Recomm")
// MAGIC   .getOrCreate()

// COMMAND ----------

// MAGIC %scala
// MAGIC val s3Bucket = "spotify-songs-csv"
// MAGIC val s3ObjectPath = "songs.csv"

// COMMAND ----------

// MAGIC %scala
// MAGIC val s3Url = s3Bucket + "/" + s3ObjectPath

// COMMAND ----------

// MAGIC %scala
// MAGIC val df = spark.read
// MAGIC   .format("csv")
// MAGIC   .option("header", "true")
// MAGIC   .load("s3://" + s3Url)

// COMMAND ----------

// MAGIC %scala
// MAGIC df

// COMMAND ----------

// MAGIC %scala
// MAGIC df.printSchema()

// COMMAND ----------

import org.apache.spark.sql.functions._
val df_2002 = df.filter(col("year") > 2002)
df.show()

// COMMAND ----------

// MAGIC %scala
// MAGIC val nameCounts = df_2002.groupBy("name").count().orderBy(desc("count"))
// MAGIC
// MAGIC nameCounts.show()
// MAGIC

// COMMAND ----------

// MAGIC %scala
// MAGIC val df_home = df_2002.filter(col("name") === "Home")

// COMMAND ----------

// MAGIC %scala
// MAGIC df_home.show()

// COMMAND ----------

// MAGIC %scala
// MAGIC val df_1 = df_2002.dropDuplicates("name")
// MAGIC df_1.show()

// COMMAND ----------

// MAGIC %scala
// MAGIC val featureCols = Array("danceability", "energy", "valence", "loudness")
// MAGIC val new_df = featureCols.foldLeft(df_1) { (tempDF, colName) =>
// MAGIC   tempDF.withColumn(colName, col(colName).cast("double"))
// MAGIC }
// MAGIC new_df.show()

// COMMAND ----------

// MAGIC %scala
// MAGIC import org.apache.spark.ml.clustering.KMeans
// MAGIC import org.apache.spark.ml.feature.{VectorAssembler, MinMaxScaler}
// MAGIC
// MAGIC val assembler = new VectorAssembler()
// MAGIC   .setInputCols(featureCols)
// MAGIC   .setOutputCol("features")
// MAGIC
// MAGIC val assembledDF = assembler.transform(new_df)

// COMMAND ----------

// MAGIC %scala
// MAGIC val scaler = new MinMaxScaler()
// MAGIC   .setInputCol("features")
// MAGIC   .setOutputCol("scaledFeatures")
// MAGIC
// MAGIC val scaledDF = scaler.fit(assembledDF).transform(assembledDF)

// COMMAND ----------

val kmeans = new KMeans()
  .setK(4) // Number of clusters
  .setSeed(15) // Random seed
  .setFeaturesCol("scaledFeatures")
  .setPredictionCol("cluster")

// COMMAND ----------

val model = kmeans.fit(scaledDF)

// COMMAND ----------

val clusterDF = model.transform(scaledDF)

// COMMAND ----------

clusterDF.select("cluster").show()

// COMMAND ----------

val counts = clusterDF.groupBy("cluster").count()
counts

// COMMAND ----------

counts.show()

// COMMAND ----------

val y = clusterDF.select("cluster")
y

// COMMAND ----------

y.show()

// COMMAND ----------

val columnsToDrop = Seq("name", "artists", "id", "release_date","features","scaledFeatures")
val X = clusterDF.drop(columnsToDrop: _*)

// COMMAND ----------

X.show()

// COMMAND ----------

val testSize = 0.15

val Array(x_train, x_test) = X.randomSplit(Array(1 - testSize, testSize), seed = 12345L)

// COMMAND ----------

val featureCols = Array("acousticness", "danceability", "duration_ms", "energy", "explicit", "instrumentalness", "key", "liveness", "loudness", "mode", "popularity", "speechiness", "tempo", "valence", "year")

val x_train_1 = featureCols.foldLeft(x_train) { (tempDF, colName) =>
  tempDF.withColumn(colName, col(colName).cast("double"))
}

val assembler = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features")

val x_train_assembled = assembler.transform(x_train_1)


// COMMAND ----------

x_train_assembled.show()

// COMMAND ----------

import org.apache.spark.ml.classification.RandomForestClassifier

val rf = new RandomForestClassifier()
  .setLabelCol("cluster")
  .setFeaturesCol("features")
  .setNumTrees(100)
  .setSeed(15)

// COMMAND ----------

val rfModel = rf.fit(x_train_assembled)

// COMMAND ----------

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("cluster") // Specify the label column name
  .setPredictionCol("prediction") 
  .setMetricName("accuracy")

// COMMAND ----------

x_test.show(
  
)

// COMMAND ----------

val featureCols = Array("acousticness", "danceability", "duration_ms", "energy", "explicit", "instrumentalness", "key", "liveness", "loudness", "mode", "popularity", "speechiness", "tempo", "valence", "year")

val x_test_1 = featureCols.foldLeft(x_test) { (testDF, colNam) =>
  testDF.withColumn(colNam, col(colNam).cast("double"))
}

val assembler_test = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features")

val x_test_assembled = assembler_test.transform(x_test_1)

// COMMAND ----------

x_test_assembled.show()

// COMMAND ----------

val predictions = rfModel.transform(x_test_assembled)

// COMMAND ----------

predictions.show()

// COMMAND ----------

val predictions_cluster = predictions.select("prediction","cluster")

// COMMAND ----------

val accuracy = evaluator.evaluate(predictions_cluster)

// COMMAND ----------

val featureImportances = rfModel.featureImportances.toArray
val featureImportanceDF = featureCols.zip(featureImportances).toSeq.toDF("Feature", "Importance")

// COMMAND ----------

featureImportanceDF.show()

// COMMAND ----------

val sortedDF = featureImportanceDF.withColumn("Importance", col("Importance").cast("double")) // Correct import
  .sort(desc("Importance"))

// COMMAND ----------

sortedDF.show()

// COMMAND ----------

clusterDF.show()

// COMMAND ----------

import org.apache.spark.sql.functions._

val groupedData = clusterDF.groupBy("cluster")
groupedData

// COMMAND ----------

val sortedDF = clusterDF.orderBy(desc("popularity"))
sortedDF

// COMMAND ----------

sortedDF.show()

// COMMAND ----------

val selectedColumns = sortedDF.select("name", "popularity", "cluster", "artists")
selectedColumns.show()

// COMMAND ----------

import org.apache.spark.sql.DataFrame
def getResults(emotionWord: String, data: DataFrame): DataFrame = {
  val NUM_RECOMMEND = 10
  
  val filteredData = emotionWord match {
    case "happy" =>
      data.filter(col("cluster") === 2 && col("popularity") > 75 && col("name") =!= "NaN")
    case "sad" =>
      data.filter(col("cluster") === 1 && col("popularity") > 66 && col("name") =!= "NaN")
    case "disgust" =>
      data.filter(col("cluster") === 0 && col("popularity") > 75 && col("name") =!= "NaN")
    case "neutral" =>
      data.filter(col("cluster") === 3 && col("popularity") > 69 && col("name") =!= "NaN")
    case _ =>
      data // Handle other cases or return the original DataFrame
  }
  
  val recommendedSongs = filteredData.select("name", "artists","popularity")
  recommendedSongs.limit(NUM_RECOMMEND)
}

// COMMAND ----------

val emotion = "happy"
val recommendedSongs = getResults(emotion, selectedColumns)
recommendedSongs.show()

// COMMAND ----------

val emotion = "sad"
val recommendedSongs = getResults(emotion, selectedColumns)
recommendedSongs.show()

// COMMAND ----------

val emotion = "disgust"
val recommendedSongs = getResults(emotion, selectedColumns)
recommendedSongs.show()

// COMMAND ----------

val emotion = "neutral"
val recommendedSongs = getResults(emotion, selectedColumns)
recommendedSongs.show()

// COMMAND ----------


