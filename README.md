sentiment_analysis_toolkit
==========================

Simple module for determining sentence sentiment using a Naive Bayes classifier

Composed of three scripts:

- train.py

   Used to train a naive Bayes classifier on two input files of positive and negative data. Outputs classifier as .pickle file.

- classify.py

   Uses pickled classifier to predict whether input test data is positive or negative.

- nbayes_sentiment.py

   Contains feature extraction methods used by train.py and also serves as a testing demo to evaluate accuracy of classifiers with different options.


Installing Dependencies
--------------------------

Required for basic functionality:

   - [nltk](http://nltk.org/)
      - install with 'sudo easy_install pip nltk'

Needed for stopword removal:

   - [nltk corpus](http://nltk.org/data.html) stopword files
      - install with 'sudo python -m nltk.downloader -d /usr/share/nltk_data stopwords'

Needed for nbayes_sentiment.py ROC graphing:

   - [matplotlib](http://matplotlib.sourceforge.net/users/installing.html)
      - Install dmg from site
   - [pyroc](https://github.com/marcelcaraciolo/PyROC) 
      - Clone file and include in folder


Usage
--------------------------

- train.py
   
   python train.py -p [positive_file_name] -n [negative_file_name] -o [classifier_file_name]
   
   Optional options:
   - -l [number_of_best_features_to_limit] sets a specific limit to the number of best features used for training (default is 1000)
   - -b uses bigrams as features
   - -s removes stopwords before processing
   - -v prints status messages
   
- classify.py

   python classify.py -c [classifier_file_name] -t [test_data_file_name]

   Optional options:
   - -o [output_file_name] outputs results to this file instead of STDOUT
   - -d [delimiter] sets column delimiter for files (default is "\t")
   - -b denotes that input classifier using bigram features
      - -l [number_of_features] specifies the number of best features used when training bigram classifier
   - -v prints status messages
   
- nbayes_sentiment.py

   python nbayes_sentiment.py -p [positive_file_name] -n [negative_file_name]
   
   Optional options:
   - -l [number_of_features] sets the number of best features to use (default is 1000)
   - -d [number_of_divisions] selects the number of divisions created in input data: 1 out of d will be used for testing.)
   - -a trains and tests over each possible set of divisions of data and averages the results for more smoothing
   - -b uses bigrams as features
   - -s filters out stopwords from features
   - -r randomizes training data to reduce clumping
   - -g graphs the resulting ROC curves for each round of testing

Example
--------------------------
Suppose that we have sets of positive and negative movie reviews, in data/pos.txt and data/neg.txt respectively, and we want to create the best classifier possible for determining whether new reviews are good or bad:

1. First, we should try different classifier evaluation methods to find the options whose resulting classifier has the sensitivity and specificity closest to the values we want:
   1. Try using different numbers of the best single-word features:
      - python nbayes_sentiment.py -p data/pos.txt -n data/neg.txt -r -l 100
      - python nbayes_sentiment.py -p data/pos.txt -n data/neg.txt -r -l 1000
      - python nbayes_sentiment.py -p data/pos.txt -n data/neg.txt -r -l 10000

   2. Try using bigrams as features:
      - python nbayes_sentiment.py -p data/pos.txt -n data/neg.txt -r -l 1000 -b

   3. Try removing stop words from whichever of the previous methods worked best
      - python nbayes_sentiment.py -p data/pos.txt -n data/neg.txt -r -l 1000 -s

   4. If our dataset is small, try using the averaging option for smoother results, as well setting a the d option to something greater than 4:
      - python nbayes_sentiment.py -p data/pos.txt -n data/neg.txt -r -l 1000 -s -a -d 5

2. At this point we know which options work best for our data, so the next step is to use train.py to train a classifier on all of our data with those options.
   - python train.py -p data/pos.txt -n data/neg.txt -o movieNBClassifierNoStop -l 1000 -s

3. Now we can use "movieNBClassifierNoStop.pickle" to classify any new sets of movie review data, in this case reviews.txt. We can use the -o and -d options to specify the name of a two-column output file that uses a specific delimiter:
   - python classify.py -c movieNBClassifierNoStop.pickle -t reviews.txt
   - python classify.py -c movieNBClassifierNoStop.pickle -t reviews.txt -o classified_reviews.txt -d | 
      - *I recommend using ` or | as your delimiter to avoid quotation and comma issues during post-processing and importing into Excel


