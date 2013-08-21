# train.py
#
# This program is designed to train a Naive Bayes classifier.
# It must be passed in the paths to positive and negative files containing 
# a newline separated list of strings to train the model with, as well as 
# a file name for the output file. 
#
# Options:
# ------------------------
# -p [positive_file_name] 
# -n [negative_file_name]
# -o [output_file_name]
# -l [number_of_best_features_to_train_on]
# -b (use bigrams as features)
# -s (remove stop words before processing)
# -t (tag negated words) 
# -v (print status messages)
#
# written by George Michopoulos, 7/20/13 
#
# TODO:
# ------------------------
# -c [csv_file_name] (use two column csv as input instead)
# -d [delimiter] (use to specify non-comma delimiter for input file)
#

import re, math, random, collections, itertools, pickle, os, sys, argparse
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords
from nbayes_sentiment import *

def train(feature_method, pos, neg, limit, rand, stop, stopset, word_scores):
  """ This function returns a NaiveBayesClassifier trained on the top [limit] features using
      the passed in feature_method. """

  pos_features = []
  neg_features = []

  with open(pos, 'r') as pos_sentences:
    for s in pos_sentences:
      pos_words = re.findall(r"[\w']+|[.,!?;]", s.rstrip())
      pos_words = [feature_method(pos_words, limit, stop, stopset, word_scores), 1]
      pos_features.append(pos_words)

  with open(neg, 'r') as neg_sentences:
    for s in neg_sentences:
      neg_words = re.findall(r"[\w']+|[.,!?;]", s.rstrip())
      neg_words = [feature_method(neg_words, limit, stop, stopset, word_scores), 0]
      neg_features.append(neg_words)

  # Trains a Naive Bayes Classifier
  return NaiveBayesClassifier.train(pos_features + neg_features)  


###### MAIN ######

def main(argv):
  parser = argparse.ArgumentParser(description="Run sentiment analysis using\
                                   a positive and a negative input file")

  parser.add_argument("-p", "--positive", help="input relative path of a \
                      positive data file", required=True)

  parser.add_argument("-n", "--negative", help="input relative path of a \
                      negative data file", required=True)

  parser.add_argument("-o", "--output", help="file name for output classifier, \
                      not including .pickle file ending", default="nBayesClassifier")

  parser.add_argument("-l", "--limit_features", type=int, help="number of best \
                      features to use", default="1000")

  parser.add_argument("-b", "--bigram", help="train using bigram features.",
                      action="store_true")

  parser.add_argument("-s", "--stopwords", help="filter out stop words before \
                      training.", action="store_true")

  parser.add_argument("-t", "--tag_negative_words", help="tag negated words with \
                      word_not to capture more meaning.", action="store_true")

  parser.add_argument("-v", "--verbose", help="print status messages",
                      action="store_true")

  args = parser.parse_args()


  f = open(args.output + '.pickle', 'wb')


  # Set up stopword set
  stopset = 0
  if args.stopwords:
    if args.verbose: print "Stop words are being filtered out."
    stopset = set(stopwords.words('english'))

  if args.bigram:
    if args.verbose: print '\nEvaluating the best %d bigram word features\n' % (args.limit_features)
    classifier = train(bigram_word_features, args.positive, args.negative, args.limit_features, 0, args.stopwords, stopset, 0)

  else:
    # Finds word scores
    if args.verbose: print '\nEvaluating the best %d word features\n' % (args.limit_features)
    word_scores = create_word_scores(args.positive, args.negative)
    classifier = train(best_word_features, args.positive, args.negative, args.limit_features, 0, args.stopwords, stopset, word_scores)

  pickle.dump(classifier, f)
  f.close()
  if args.verbose: print 'Successfully wrote classifier to file ' + args.output + ".pickle!"


if __name__ == '__main__':
    main(sys.argv[1:])
