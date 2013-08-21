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

==========================

Installing Dependencies
--------------------------

Required for basic functionality:

   - nltk: install with 'sudo easy_install pip nltk'

Needed for stopword removal:

   - stopwords from nltk corpus: instlal with 'sudo python -m nltk.downloader -d /usr/share/nltk_data stopwords'

Needed for nbayes_sentiment.py ROC graphing:

   - matplotlib: see http://matplotlib.sourceforge.net/users/installing.html
   - pyroc: included in repo, see: https://github.com/marcelcaraciolo/PyROC

==========================

Usage
--------------------------

- train.py
   
   python train.py -p [positive_file_name] -n [negative_file_name] -o [classifier_file_name]
   
   Optional options:
   - -b (use bigrams as features)
   - -s (remove stop words before processing)
   - -t (tag negated words) 
   - -v (print status messages)
   
- classify.py

   python train.py -c [classifier_file_name] -t [test_data_file_name]

   Optional options:
   - -o [output_file_name]  (use to output to file instead of STDOUT)
   - -d [delimiter] (choose column delimiter for files)
   - -v (print status messages)
   - -t (tag negated words) 
   
- nbayes_sentiment.py

   python nbayes_sentiment.py -p [positive_file_name] -n [negative_file_name]
   
   Optional options:
   - -n [negative_file_name] (path to newLine separated list of negative input)
   - -l [number_of_features] (number of best features to use)
   - -d [number_of_divisions] (select the number of divisions created in input data: 1 out of d will be used for testing.)
   - -a (train and test over each possible set of divisions of data and average the results for more smoothing)
   - -b (use bigrams as features)
   - -s (filter out stopwords from features)
   - -r (randomize training data to reduce clumping)
   - -g (graph the resulting ROC curves of each round of testing that occurs)
   - -t (tag negated words with word_not)   



