sentiment_analysis_toolkit
==========================

Simple module for determining sentence sentiment using a Naive Bayes classifier

Composed of three scripts:

- train.py

   Used to train a naive Bayes classifier on two input files of positive and negative data. Outputs classifier as .pickle file.

- classify.py

   Uses pickled classifier to predict whether input test data is positive or negative.

- nbayes_sentiment.py

   Contains feature extraction methods used by train.py and also serves a testing demo to evaulate accuracy of classifiers with different options.


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
      - Already bundled with toolkit


Usage
--------------------------

- train.py
   
   python train.py -p [positive_file_name] -n [negative_file_name] -o [classifier_file_name]
   
   Optional options:
   - -l [number_of_best_features_to_limit] sets a specific limit to the number of best features uesed for training (default is 1000)
   - -b uses bigrams as features
   - -s removes stop words before processing
   - -t tags negated words
   - -v prints status messages
   
- classify.py

   python train.py -c [classifier_file_name] -t [test_data_file_name]

   Optional options:
   - -o [output_file_name] outputs results to this file instead of STDOUT
   - -d [delimiter] sets column delimiter for files (default is "\t")
   - -t tags negated words
   - -v prints status messages
   
- nbayes_sentiment.py

   python nbayes_sentiment.py -p [positive_file_name] -n [negative_file_name]
   
   Optional options:
   - -l [number_of_features] sets the number of best features to use (default is 1000)
   - -d [number_of_divisions] selects the number of divisions created in input data: 1 out of d will be used for testing.)
   - -a trains and tests over each possible set of divisions of data and average the results for more smoothing
   - -b uses bigrams as features
   - -s filters out stopwords from features
   - -r randomizes training data to reduce clumping
   - -g graphs the resulting ROC curves of each round of testing that occurs
   - -t tags negated words with word_not



