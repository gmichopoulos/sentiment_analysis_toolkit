sentiment_analysis_toolkit
==========================

Simple module for determining sentence sentiment using a Naive Bayes classifier

Composed of three scripts:

1. train.py

   Used to train a naive Bayes classifier on two input files of positive and negative data. Outputs classifier as .pickle file.

2. classify.py

   Uses pickled classifier to predict whether input test data is positive or negative.

3. nbayes_sentiment.py

   Contains feature extraction methods used by train.py and also serves a testing demo to evaulate accuracy of classifiers with different options.

The toolkit also includes a copy of PyRoc (https://github.com/marcelcaraciolo/PyROC) which is used to graph the ROC curves of different classifiers when testing in nbayes_sentiment.py