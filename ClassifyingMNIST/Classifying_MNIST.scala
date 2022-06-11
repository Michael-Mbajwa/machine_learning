import scala.io.Source._
import java.util.Properties
import java.io.{ObjectOutputStream, FileOutputStream, ObjectInputStream, FileInputStream, File}
import java.io.{BufferedWriter, FileWriter}
import scala.reflect.io.Directory
import org.apache.spark.ml.classification.{LogisticRegression, LinearSVC}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.rdd._
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}


// Load training data
val data = spark.read.format("libsvm").load("/home/users/mmbajwa/mnist.bz2")
data.show(10) // we can observe we have 780 features

// Brief data exploration
// We have 60000 rows
data.count()

// How many unique labels do we have:
data.select("label").distinct().show()
data.select("label").distinct().count()
// we have 10 unique labels i.e. numbers 0 to 9

// we can check the occurences of each label
data.groupBy("label").count().show()

// Prepare training and test sets with randomSplit
val Array(training, testing) = data.randomSplit(Array(0.90, 0.10), 12345)
training.cache()
testing.cache()


// Implement pipeline for Logistic Regression and for Hyperparameter tuning
val lr = new LogisticRegression()
val pipeline = new Pipeline().setStages(Array(lr))

val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.3, 0.01)).addGrid(lr.maxIter, Array(10, 20, 100)).addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)).addGrid(lr.fitIntercept).build()

// We will now wrap the the pipeline in a cross validator instance
val tcv = new TrainValidationSplit().setEstimator(pipeline).setEvaluator(new MulticlassClassificationEvaluator).setEstimatorParamMaps(paramGrid).setParallelism(2).setTrainRatio(0.8)

// Run cross-validation, and choose the best set of parameters.
val startTime = System.nanoTime()
val cvModel = tcv.fit(training)
val elapsedTime = (System.nanoTime() - startTime) / 1e9
println("Logistic Regression Training..\n")
println("Total training time for the logistic Regression is " + elapsedTime + "seconds.")


// Make prediction on test data and inspect output of model
val predictionsLR = cvModel.transform(testing).select("label", "prediction")
predictionsLR.show()

// Evaluate our model
println("\nLogistic Regression model Evaluation.\n")
val predLabels = predictionsLR.as[(Double, Double)].rdd
val Multi_Class_Metrics = new MulticlassMetrics(predLabels)

// Let's deep dive into certain statistics
// confusion matrix
println("Confusion matrix:")
println(Multi_Class_Metrics.confusionMatrix)


// Overall statistics
val accuracy = Multi_Class_Metrics.accuracy
printtln("\nThe accuracy of the logistic Regression model is: " + accuracy)

// Precision by label
prinln("\nThe precision for each label is:")
val labels = Multi_Class_Metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + Multi_Class_Metrics.precision(l))
}

// False positive rate by label
println("\nThe false positive rate per class is:")
labels.foreach { l =>
  println(s"FPR($l) = " + Multi_Class_Metrics.falsePositiveRate(l))
}


println("\n\n\nWe will now try to Improve the accuracy of our classification with RANDOM FOREST")
println("Attempting RANDOM FOREST")
// Train a RandomForest model.
val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features")

// Chain forest in a Pipeline.
val pipeline = new Pipeline().setStages(Array(rf))

// Hyperparameter tuning
val paramGridRF = new ParamGridBuilder().addGrid(rf.maxDepth, Array(5, 10, 15)).addGrid(rf.maxBins, Array(30, 50, 100)).addGrid(rf.impurity, Array("entropy", "gini")).addGrid(rf.numTrees, Array(5, 10, 20)).build()
val cvRF = new CrossValidator().setEstimator(pipeline).setEvaluator(new MulticlassClassificationEvaluator).setEstimatorParamMaps(paramGridRF).setNumFolds(5).setParallelism(16)

// Run cross-validation (no validation split), and choose the best set of parameters.
val startTime = System.nanoTime()
val rfCvModel = cvRF.fit(training)
val elapsedTime = (System.nanoTime() - startTime) / 1e9
println("Random Forest Training..\n")
println("Total training time for the Random Forest is " + elapsedTime + "seconds.")

// The best parameters
val rfbestModelParams = rfCvModel.bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap

val predictionRF = rfCvModel.transform(testing).select("label", "prediction")

// Evaluate our model
val predLabelsRF = predictionRF.as[(Double, Double)].rdd
val MultiClassMetricsRF = new MulticlassMetrics(predLabelsRF)

// Let's deep dive into certain statistics
// confusion matrix
println("Confusion matrix for Random Forest:")
println(MultiClassMetricsRF.confusionMatrix.toString())

// Overall statistics
println("\nAccuracy of Random Forest:")
val accuracyRF = MultiClassMetricsRF.accuracy
println("Accuracy of random forest is " + accuracyRF)
println("The Random Forest Model outperforms the logistic regression model.")

// Precision by label
println("\nPrecision by label for Random Forest:\n")
val labelsRF = MultiClassMetricsRF.labels

labelsRF.foreach { l =>
  println(s"Precision($l) = " + MultiClassMetricsRF.precision(l))
}

// False positive rate by label
println("\nFalse positive rate by label for Random Forest:\n")
labelsRF.foreach { l =>
  println(s"FPR($l) = " + MultiClassMetricsRF.falsePositiveRate(l))
}
