Latent Semantic Analysis of Wikipedia Movie Plots
5-19 May 2022

Technologies Used: Scala/Spark

Mined dataset from https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots. 

Read the articles' title, genre, and plot fields into an initial DataFrame.

Created a features column, which contains a list of lemmatized text tokens extracted from each of the plots.

Computed an SVD decomposition of the ~30000 movie plots with specified parameters.

Used consine similary metric for extracting keyword queries.

You would require the following jar files to run this program: stanford-corenlp-3.9.2-models.jar, stanford-corenlp-3.9.2.jar, breeze_2.12-2.0.jar. Visit here: https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.9.2/ for the stanford jar files.