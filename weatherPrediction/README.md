Weather Prediction via Regression Trees & Regression Forests
14 - 28 April 2022


Technologies Used: Scala/Spark

Mined dataset from https://www1.ncdc.noaa.gov/pub/data/noaa/ - Integrated Surface Data (ISD) of the National Oceanic and Atmospheric Administration (NOAA) provides a rich collection of weather measurements from around the world.

Parsed the input files into an appropriate RDD. Transformed it into a data frame and selected specific features for the ML pipeline.

Computed correlations of all features with the target variable temperature.

Trained a random forest regressor with all records from 1949 until 2021. Tested the model on records for 2022. The target variable is Temperature.

With the selected hyperparameters, the best parameters achieved a Mean Squared Error of  4.835512589969317.
