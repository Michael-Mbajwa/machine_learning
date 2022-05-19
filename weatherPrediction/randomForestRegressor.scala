import org.apache.spark.rdd._
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.mllib.evaluation._
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, CrossValidatorModel}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.ml.feature.QuantileDiscretizer
import org.apache.spark.ml.Pipeline
import java.io.{BufferedWriter, FileWriter, File}
import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Row, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._

import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import spark.implicits._


//Load the previously saved dataset from Question 2A
val data = spark.read.option("inferSchema", true).option("header", true).csv("/home/users/mmbajwa/problem2A_data.csv")

// Check the dataTypes of each column and ensure it matches with the previous types used in Problem_2A
data.printSchema

val conditionTraining = col("year") < 2022
val conditionTest = col("year") === 2022

val training = data.filter(conditionTraining)
val testing = data.filter(conditionTest)

training.cache()
testing.cache()

val featureCols = training.columns.filter(_ != "airTemperature")

val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

// Train a RandomForest model.
val rF = new RandomForestRegressor().setNumTrees(10).
  setFeatureSubsetStrategy("auto").
  setLabelCol("airTemperature").
  setFeaturesCol("features")

val pipeline = new Pipeline().setStages(Array(assembler, rF))

val paramGrid = new ParamGridBuilder().addGrid(rF.impurity, Array("variance")).addGrid(rF.maxDepth, Array(5,10,15)).addGrid(rF.maxBins, Array(20,50, 100)).build()

// Select (prediction, true label) and compute mean square test error.
val evaluator = new RegressionEvaluator().
  setLabelCol("airTemperature").
  setPredictionCol("prediction").
  setMetricName("mse")

val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(5).setParallelism(8)

// Train model.
val startTime = System.nanoTime()
val rFmodel = cv.fit(training)
val elapsedTime = (System.nanoTime() - startTime) / 1e9

rFmodel.write.overwrite().save("/home/users/mmbajwa/problem2_rfModel")

training.unpersist()

val bestrfModel = rFmodel.bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[RandomForestRegressionModel].extractParamMap()

// Predict the temperature for Luxembourg on a few recent dates in 2022
// Load the Luxembourg data we obtained for May temperatures
// This LuxData was obtained by using the code in Problem 2A to extract the data from https://www1.ncdc.noaa.gov/pub/data/noaa/2022/065900-99999-2022.gz
val LuxData = spark.read.option("inferSchema", true).option("header", true).csv("/home/users/mmbajwa/luxWeather_data.csv")
// select only data for May 1st to May 4th
val condLux = col("month") === 5
val LuxTestData = LuxData.filter(condLux)

val predictionsLux = rFmodel.transform(LuxTestData)
predictionsLux.show(10)
val mseLux = evaluator.evaluate(predictionsLux)


// 2C: predict the temperatures for the Luxembourg  along the whole year of 2022 available from NOAA. Calculate the Mean Squared Error (MSE)
LuxData.select("month").distinct().show() // Shows we have data from January to May
val predictions = rFmodel.transform(LuxData)
val MSE = evaluator.evaluate(predictions)

// 2D
testing.unpersist()

def Correlation( a:RDD[Double], b:RDD[Double] ) = {
  val correlation: Double = Statistics.corr(a, b, "spearman")
  println(s"Correlation is: $correlation")
}

// Correlation with no discretization
val X = data.select("airTemperature").as[Double].rdd
println("DewPoint - AirTemperature")
Correlation(X, data.select("dewPointTemperature").as[Double].rdd)

println("distanceDimension - AirTemperature")
Correlation(X, data.select("distanceDimension").as[Double].rdd)

println("ceilingHeightDimension - AirTemperature")
Correlation(X, data.select("ceilingHeightDimension").as[Double].rdd)

println("speedRate - AirTemperature")
Correlation(X, data.select("speedRate").as[Double].rdd)

println("directionAngle - AirTemperature")
Correlation(X, data.select("directionAngle").as[Double].rdd)

println("elevationDimension - Air Temperature")
Correlation(X, data.select("elevationDimension").as[Double].rdd)

println("Longitude - Air Temperature")
Correlation(X, data.select("longitude").as[Double].rdd)

println("Latitude - Air Temperature")
Correlation(X, data.select("latitude").as[Double].rdd)

println("Hour - Air Temperature")
Correlation(X, data.select("hour").as[Double].rdd)

println("Day - Air Temperature")
Correlation(X, data.select("day").as[Double].rdd)

println("Month - Air Temperature")
Correlation(X, data.select("month").as[Double].rdd)

println("Year - Air Temperature")
Correlation(X, data.select("year").as[Double].rdd)


// Correlation with discretization
val pipelineDis = new Pipeline().setStages(for {
  c <- data.columns
  if c != "airTemperature"
} yield new QuantileDiscretizer().setInputCol(c).setOutputCol(s"${c}_discretized").setNumBuckets(15))

var result = pipelineDis.fit(data).transform(data)
result = result.drop(data.columns.diff(Seq("airTemperature")): _*)
val airTemp = result.select("airTemperature").as[Double]

println("Descritized: DewPoint - AirTemperature")
Correlation(airTemp.rdd, result.select("dewPointTemperature_discretized").as[Double].rdd)

println("Descritized: distanceDimension - AirTemperature")
Correlation(airTemp.rdd, result.select("distanceDimension_discretized").as[Double].rdd)

println("Descritized: ceilingHeightDimension - AirTemperature")
Correlation(airTemp.rdd, result.select("ceilingHeightDimension_discretized").as[Double].rdd)

println("Descritized: speedRate - AirTemperature")
Correlation(airTemp.rdd, result.select("speedRate_discretized").as[Double].rdd)

println("Descritized: directionAngle - AirTemperature")
Correlation(airTemp.rdd, result.select("directionAngle_discretized").as[Double].rdd)

println("Descritized: elevationDimension - Air Temperature")
Correlation(airTemp.rdd, result.select("elevationDimension_discretized").as[Double].rdd)

println("Descritized: Longitude - Air Temperature")
Correlation(airTemp.rdd, result.select("longitude_discretized").as[Double].rdd)

println("Latitude - Air Temperature")
Correlation(airTemp.rdd, result.select("latitude_discretized").as[Double].rdd)

println("Hour - Air Temperature")
Correlation(airTemp.rdd, result.select("hour_discretized").as[Double].rdd)

println("Day - Air Temperature")
Correlation(airTemp.rdd, result.select("day_discretized").as[Double].rdd)

println("Month - Air Temperature")
Correlation(airTemp.rdd, result.select("month_discretized").as[Double].rdd)

println("Year - Air Temperature")
Correlation(airTemp.rdd, result.select("year").as[Double].rdd)