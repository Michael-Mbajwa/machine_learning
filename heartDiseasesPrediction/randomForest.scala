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
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit, CrossValidatorModel}
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


val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setFeatureSubsetStrategy("auto")

val pipelineRF = new Pipeline().setStages(Array(labelindexer, featureHasher,rf))

val paramGridRF = new ParamGridBuilder().addGrid(rf.maxDepth, Array(5,10,15)).addGrid(rf.maxBins, Array(20,50, 100)).addGrid(rf.impurity, Array("entropy", "gini")).addGrid(rf.numTrees, Array(5,10,20)).build()

val cvRF = new CrossValidator().setEstimator(pipelineRF).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGridRF).setNumFolds(5).setParallelism(8)

// Run cross-validation, and choose the best set of parameters.
val startTime = System.nanoTime()
val rfCvModel = cvRF.fit(training)
val elapsedTime = (System.nanoTime() - startTime) / 1e9

// best parameters
val rfbestModelParams = rfCvModel.bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap

// save model into a file
rfCvModel.write.overwrite().save("/home/users/mmbajwa/rfCvModel")

val rfPrediction = rfCvModel.transform(testing)

val rfpredictionsAndLabels = rfPrediction.select("label", "prediction").as[(Double, Double)].rdd

val rfbCMetrics = new BinaryClassificationMetrics(rfpredictionsAndLabels)

// Accuracy
val evaluatorRF = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction")

val accuracyRF = evaluatorRF.evaluate(rfPrediction)

// 80-10-10 split for comparison with previous accuracy
val trainValidationSplit = new TrainValidationSplit().setEstimator(pipelineRF).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGridRF).setTrainRatio(0.8).setParallelism(8)

val modelRF = trainValidationSplit.fit(training)
val modelRFpredict = modelRF.transform(testing)
val accuracyModelRF = evaluatorRF.evaluate(modelRFpredict)
