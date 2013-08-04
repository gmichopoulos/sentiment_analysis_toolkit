# classify.py
#
# This program is designed to classify data using a Naive Bayes classifier. 
# It must be passed in a file containing a trained nltk.NaiveBayesClassifier
# (the output of train.py) and a file containing a list of newline separated
# test input. It will output the list of positive and negative questions as a
# two column, tab-separated list. (positive \t negative)
# 
# Options:
# ------------------------
# -c [classifier_file_name]
# -o [output_file_name]  (use to output to file instead of STDOUT)
#
# written by George Michopoulos, 7/20/13 
#
##############################
# TODO:
##############################
# Write script

import re, math, random, collections, itertools, pickle, csv, os, sys, argparse
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nbayes_sentiment import *


###### MAIN ######

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

  parser.add_argument("-b", "--bigram", help="classify using bigram features.",
                      action="store_true")

  args = parser.parse_args()


  # Validate and load classifier
  if args.classifier.split(".")[-1] != "pickle":
    sys.exit("classifier must be a .pickle file.")

  f = open(args.classifier)
  classifier = pickle.load(f)
  print "Classifier loaded.\n"
  f.close()

  test_features = []
  original_sentences = []
  predicted_positives = []
  predicted_negatives = []

  with open(args.test_data, 'r') as sentences:
    for s in sentences:
      words = re.findall(r"[\w']+|[.,!?;]", s.rstrip())
      if args.bigram:
        words = bigram_word_features(words, args.limit_features, 0, 0, 0)
      else:
        words = dict([(word, True) for word in words])
      test_features.append(words)
      original_sentences.append(s)
  print "Features Loaded.\n"

  # Puts original sentence into appropriate list based on classification
  for i, features in enumerate(test_features):
    predicted = classifier.classify(features)
    if predicted:
      predicted_positives.append(original_sentences[i])
    else:
      predicted_negatives.append(original_sentences[i])

  print "Classification Complete.\n"

  if args.output == "STDOUT":
    for pos, neg in itertools.izip_longest(predicted_positives,predicted_negatives,fillvalue=''):
      print pos + args.delimiter + neg + "\n"

  else:
     with open(args.output, 'wb') as f:
      result_writer = csv.writer(f, delimiter=args.delimiter, quotechar='"',
                                 quoting=csv.QUOTE_MINIMAL)
      result_writer.writerows(itertools.izip_longest(predicted_positives,predicted_negatives,fillvalue=''))
      print "Results Successfully Written to " + args.output + "!\n"



if __name__ == '__main__':
    main(sys.argv[1:])