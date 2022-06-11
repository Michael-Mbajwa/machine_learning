Problem 3
Big Data Use Case

The title of our case is Classifying MNIST dataset via Logistic Regression, Decision Trees & Random Forests. Our dataset is the popular MNIST dataset downloaded from:
 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2. 
The dataset has a training set of 60,000 examples and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. Consequently, this is a very good example dataset for those who are trying to learn techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.


// Load data
Our dataset has 60000 data points with 780 features each.
We have 10 distinct labels I.e. 0 - 9

+-----+-----+
|label|count|
+-----+-----+
|  8.0| 5851|
|  0.0| 5923|
|  7.0| 6265|
|  1.0| 6742|
|  4.0| 5842|
|  3.0| 6131|
|  2.0| 5958|
|  6.0| 5918|
|  5.0| 5421|
|  9.0| 5949|
+-----+-----+

We can also see that our labels have an even occurrence so it's good to work with.


// Training and testing split
We split the dataset into 90% for training and 10% for testing.


//Training Step 1
We first attempt classification using logistic Regression.

We build a pipeline and cross validated with the following hyper parameters;
	- regParam, Array(0.1, 0.3, 0.01)
	- maxIter, Array(10, 20, 100)
	- elasticNetParam, Array(0.0, 0.5, 1.0)

Total training time for logistic regression amounted to 150 seconds.

// Evaluated our results

1. CONFUSION MATRIX
0	1	2    3      4.      5.     6.     7.    8.      9
-+-----+-------+----+------+-------+------+------+-----+-------+--
608.0  0.0    4.0    1.0    0.0    3.0    3.0    1.0    4.0    1.0
0.0    666.0  7.0    10.0   5.0    5.0    0.0    6.0    11.0   4.0
4.0    2.0    556.0  15.0   6.0    1.0    7.0    5.0    8.0    2.0
1.0    3.0    7.0    541.0  0.0    18.0   0.0    3.0    16.0   13.0
0.0    1.0    8.0    2.0    544.0  5.0    4.0    4.0    3.0    21.0
3.0    5.0    1.0    24.0   1.0    431.0  6.0    1.0    19.0   3.0
3.0    0.0    8.0    2.0    5.0    11.0   559.0  0.0    5.0    0.0
0.0    3.0    7.0    5.0    1.0    4.0    0.0    559.0  4.0    19.0
3.0    8.0    10.0   12.0   3.0    12.0   4.0    2.0    491.0  3.0
1.0    1.0    3.0    4.0    20.0   1.0    0.0    24.0   8.0    517.0


The accuracy of the logistic Regression model is: 0.9188916876574307

The precision for each class/label is:
Precision(0.0) = 0.9759229534510433
Precision(1.0) = 0.9666182873730044
Precision(2.0) = 0.9099836333878887
Precision(3.0) = 0.8782467532467533
Precision(4.0) = 0.9299145299145299
Precision(5.0) = 0.8778004073319755
Precision(6.0) = 0.9588336192109777
Precision(7.0) = 0.9239669421487603
Precision(8.0) = 0.8629173989455184
Precision(9.0) = 0.8867924528301887

And finally, the false positive rate for each label is:
FPR(0.0) = 0.0028142589118198874
FPR(1.0) = 0.004388475481778287
FPR(2.0) = 0.010282295756216114
FPR(3.0) = 0.014010835045768728
FPR(4.0) = 0.007644974827521909
FPR(5.0) = 0.010986998718183483
FPR(6.0) = 0.004475941812756434
FPR(7.0) = 0.00859331216140482
FPR(8.0) = 0.014425744405400407
FPR(9.0) = 0.012276785714285714


To further Improve the results, we attempted to use Random Forest to classify the dataset.

// We use the same dataset and build a pipeline for cross validation

To avoid java heap space error, we adopted the following hyper parameters;
	- maxDepth, Array(5, 10, 15)
	- maxBins, Array(30, 50, 100)
	- impurity, Array("entropy", "gini")
	- numTrees, Array(5, 10, 20)

Total training time for the Random Forest is 968.546610535 seconds.

The confusion Matrix for Random Forest:
0       1      2      3      4      5      6      7      8      9
+------+------+------+-------+------+-----+------+------+------+----
617.0  0.0    3.0    0.0    1.0    1.0    3.0    1.0    1.0    3.0
0.0    677.0  2.0    1.0    1.0    0.0    0.0    2.0    3.0    2.0
0.0    3.0    587.0  13.0   1.0    1.0    2.0    3.0    3.0    0.0
0.0    0.0    3.0    576.0  0.0    9.0    1.0    1.0    7.0    11.0
1.0    2.0    2.0    0.0    562.0  1.0    0.0    4.0    3.0    8.0
0.0    1.0    1.0    8.0    0.0    469.0  6.0    0.0    6.0    4.0
1.0    1.0    3.0    0.0    2.0    1.0    567.0  0.0    3.0    1.0
0.0    5.0    5.0    4.0    2.0    1.0    0.0    583.0  2.0    7.0
3.0    0.0    2.0    9.0    1.0    7.0    3.0    2.0    536.0  5.0
1.0    0.0    3.0    5.0    15.0   1.0    1.0    9.0    5.0    542.0

// We can already begin to observe the random forest classifier did a much better job than the logistic regression.

Accuracy of random forest is 0.9598656591099916
The Random Forest Model outperforms the logistic regression model.

The precision for each label is:
Precision(0.0) = 0.9903691813804173
Precision(1.0) = 0.9825834542815675
Precision(2.0) = 0.9607201309328969
Precision(3.0) = 0.935064935064935
Precision(4.0) = 0.9606837606837607
Precision(5.0) = 0.955193482688391
Precision(6.0) = 0.9725557461406518
Precision(7.0) = 0.9636363636363636
Precision(8.0) = 0.9420035149384886
Precision(9.0) = 0.9296740994854202

The false positive rate for each label is:
FPR(0.0) = 0.0011267605633802818
FPR(1.0) = 0.002278336814125688
FPR(2.0) = 0.004492699363534257
FPR(3.0) = 0.007480830372171311
FPR(4.0) = 0.004281459419210722
FPR(5.0) = 0.00402930402930403
FPR(6.0) = 0.002976190476190476
FPR(7.0) = 0.00411522633744856
FPR(8.0) = 0.006125858548357156
FPR(9.0) = 0.007630746324213661

