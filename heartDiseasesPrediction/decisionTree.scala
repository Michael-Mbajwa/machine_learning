import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.sql.types._
import org.apache.spark.rdd._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, FeatureHasher, IndexToString, VectorIndexer}

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics


// Load data as Dataframe
val data = spark.read.option("header", "true").option("inferschema", "true").csv("/home/users/mmbajwa/heart_2020_cleaned.csv")

// Extract features
val featureHasher = new FeatureHasher().setInputCols("BMI", "Smoking", "AlcoholDrinking", "Stroke", "PhysicalHealth", "MentalHealth", "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity", "GenHealth", "SleepTime", "Asthma", "KidneyDisease", "SkinCancer").setNumFeatures(1234).setOutputCol("features")

val labelindexer = new StringIndexer().setInputCol("HeartDisease").setOutputCol("label").fit(data)

// Random split of the DataFrame
val Array(training, testing) = data.randomSplit(Array(0.90, 0.10), 12345)
training.cache()
testing.cache()


// Define a basic ML Pipeline

val decisionTree = new DecisionTreeClassifier().setSeed(100).setLabelCol("label").setFeaturesCol("features")

val pipeline = new Pipeline().setStages(Array(labelindexer, featureHasher, decisionTree))

// implement a fully automatic hyper-parameter tuning
val paramGrid = new ParamGridBuilder().addGrid(decisionTree.maxDepth, Array(5,10,15)).addGrid(decisionTree.maxBins, Array(20,50, 100)).addGrid(decisionTree.impurity, Array("entropy", "gini")).build()

val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(5).setParallelism(8)

// Run cross-validation, and choose the best set of parameters.
val startTime = System.nanoTime()

val dtCvModel = cv.fit(training)

val elapsedTime = (System.nanoTime() - startTime) / 1e9

// save model into a file
dtCvModel.write.save("/home/users/mmbajwa/dtCvModel")

// Best model
val bestModelParams = dtCvModel.bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap

// measure precision, recall and accuracy over the two possible labels of heart disease for the best ML model and
// hyper-parameter setting over the 10% test split you created in (i) by using again the BinaryClassificationMetrics package of Spark
val dtPredictions = dtCvModel.transform(testing)

val predictionsAndLabels = dtPredictions.select("label", "prediction").as[(Double, Double)].rdd
val bCMetrics = new BinaryClassificationMetrics(predictionsAndLabels)

// Precision
val precision = bCMetrics.precisionByThreshold
precision.collect()

// Recall
val recall = bCMetrics.recallByThreshold
recall.collect()

// Accuracy
val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction")

val accuracy = evaluator.evaluate(dtPredictions)
evaluator.getMetricName