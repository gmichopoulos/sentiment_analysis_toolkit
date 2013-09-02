# classify.py
#
# This script is designed to classify data using a Naive Bayes classifier. 
# It must be passed in a file containing a trained nltk.NaiveBayesClassifier
# (the output of train.py) and a file containing a list of newline (\r) separated
# test input. It will output the list of positive and negative questions as a
# two column, tab-separated list. (positive \t negative)
# 
# Options:
# ------------------------
# -c [classifier_file_name] (required classifier from train.py)
# -t [test_data_file_name] (required; input data must be newline separated)
# -o [output_file_name]  (use to output to file instead of STDOUT)
# -d [delimiter] (chooses column delimiter for files)
# -b (specifies that input classifier using bigram features)
# -l [number_of_features] (specifies the number of best features used when training bigram classifier)
# -v (print status messages)
#
# written by George Michopoulos, 7/20/13 
#

import re, math, random, collections, itertools, pickle, csv, os, sys, argparse
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nbayes_sentiment import *


def main(argv):

  parser = argparse.ArgumentParser(description="Classify input data into a \
                                    positive and negative list using the passed\
                                    in classifier (output of train.py)")

  parser.add_argument("-c", "--classifier", help="input file name containing \
                      classifier created by train.py", required=True)

  parser.add_argument("-t", "--test_data", help="file name containing the test \
                      data to be classified, must be newline separated list of \
                      sentences.", required=True)

  parser.add_argument("-o", "--output", help="output file name, including file \
                      ending", default="STDOUT")

  parser.add_argument("-d", "--delimiter", help="choose column delimiter for \
                      files.", default="\t")

  parser.add_argument("-l", "--limit_features", type=int, help="number of best \
                      features to use", default="1000")

  parser.add_argument("-b", "--bigram", help="classify using bigram features",
                      action="store_true")

  parser.add_argument("-v", "--verbose", help="print status messages",
                      action="store_true")

  args = parser.parse_args()


  # Validate and load classifier
  if args.classifier.split(".")[-1] != "pickle":
    sys.exit("classifier must be a .pickle file.")
  f = open(args.classifier)
  classifier = pickle.load(f)
  if args.verbose: print "Classifier loaded.\n"
  f.close()

  test_features = []
  original_sentences = []
  predicted_positives = []
  predicted_negatives = []

  # Extract features from test data
  with open(args.test_data, 'r') as f:
    for s in f:
      words = re.findall(r"[\w']+|[.,!?;]", s.rstrip())
      if args.bigram:
        words = bigram_word_features(words, args.limit_features, 0, 0, 0)
      else:
        words = dict([(word, True) for word in words])
      test_features.append(words)
      original_sentences.append(s.rstrip())

  if args.verbose: print "Features loaded for " + str(len(test_features)) + " sentences.\n"


  # Put original sentence into appropriate list based on classification
  for i, features in enumerate(test_features):
    predicted = classifier.classify(features)
    if predicted:
      predicted_positives.append(original_sentences[i])
    else:
      predicted_negatives.append(original_sentences[i])

  if args.verbose: 
    print "Classification Complete.\n " + str(len(predicted_positives)) + \
          " sentences classified as positive,\n " + str(len(predicted_negatives)) + \
          " sentences classified as negative.\n "

  # Write output
  if args.output == "STDOUT":
    for pos, neg in itertools.izip_longest(predicted_positives, predicted_negatives, fillvalue=''):
      print pos + args.delimiter + neg + "\n"

  else:
     with open(args.output, 'wb') as f:
      result_writer = csv.writer(f, dialect='excel', delimiter=args.delimiter)
      result_writer.writerows(itertools.izip_longest(predicted_positives,predicted_negatives,fillvalue=''))
      if args.verbose: print "Results Successfully Written to " + args.output + "!\n"



if __name__ == '__main__':
    main(sys.argv[1:])