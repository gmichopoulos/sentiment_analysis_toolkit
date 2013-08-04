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
# -b (use bigrams as features)
# -s (remove stop words before processing)
# -t (tag negated words) 
# -r (randomize training data to reduce clumping)
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

# def best_word_features(words, limit, stop, stopset, word_scores):
#   best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, \
#                                                 reverse=True)[:limit]
#   best_words = set([w for w, s in best_vals])

#   if stop:
#     return dict([(word, True) for word in words if word in best_words 
#                                                 and word not in stopset])

#   return dict([(word, True) for word in words if word in best_words])


# def bigram_word_features(words, limit, stop, stopset, word_scores, score_fn=BigramAssocMeasures.chi_sq):

#   if stop:
#     words = [w for w in words if w not in stopset]
#   bigram_finder = BigramCollocationFinder.from_words(words)
#   bigrams = bigram_finder.nbest(score_fn, limit)
#   return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


# def tag_negated_words(sentence):

#   neglist = ['not', 'didn\'t', 'didnt', 'cant', 'can\'t', 'don\'t', 'dont', 'wouldn\'t', 'wouldnt', 'shouldn\'t', 'shouldnt', 'isn\'t', 'isnt']
#   for i, word in enumerate(sentence):
#     if word.lower() in neglist and i < len(sentence) - 1:
#       sentence[i+1] = sentence[i+1] + "_not"
#       del sentence[i]

#   return sentence


# def create_word_scores(pos, neg):
#   # Create lists of all positive and negative words
#   pos_words = []
#   neg_words = []
#   with open(pos, 'r') as pos_sentences:
#     for i in pos_sentences:
#       pos_word = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
#       #if args.tag_negative_words:
#       #  pos_word = tag_negated_words(pos_word)
#       pos_words.append(pos_word)
#   with open(neg, 'r') as neg_sentences:
#     for i in neg_sentences:
#       neg_word = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
#       #if args.tag_negative_words:
#       #  neg_word = tag_negated_words(neg_word)
#       neg_words.append(neg_word)
#   pos_words = list(itertools.chain(*pos_words))
#   neg_words = list(itertools.chain(*neg_words))

#   # Build frequency distibution of all words, positive, and negative labels
#   word_fd = FreqDist()
#   cond_word_fd = ConditionalFreqDist()
#   for word in pos_words:
#     word_fd.inc(word.lower())
#     cond_word_fd[1].inc(word.lower())
#   for word in neg_words:
#     word_fd.inc(word.lower())
#     cond_word_fd[0].inc(word.lower())

#   # Finds word counts for all sets
#   pos_word_count = cond_word_fd[1].N()
#   neg_word_count = cond_word_fd[0].N()
#   total_word_count = pos_word_count + neg_word_count

#   # Builds dictionary of word scores based on chi-squared test
#   word_scores = {}
#   for word, freq in word_fd.iteritems():
#     pos_score = BigramAssocMeasures.chi_sq(cond_word_fd[1][word], \
#                           (freq, pos_word_count), total_word_count)
#     neg_score = BigramAssocMeasures.chi_sq(cond_word_fd[0][word], \
#                           (freq, neg_word_count), total_word_count)
#     word_scores[word] = pos_score + neg_score

#   return word_scores

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
                      positive data file")

  parser.add_argument("-n", "--negative", help="input relative path of a \
                      negative data file")

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

  args = parser.parse_args()


  f = open(args.output + '.pickle', 'wb')


  # Set up stopword set
  if args.stopwords:
    print "Stop words are being filtered out."
    stopset = set(stopwords.words('english'))

  if args.bigram:
    print '\nEvaluating the best %d bigram word features\n' % (args.limit_features)
    classifier = train(bigram_word_features, args.positive, args.negative, args.limit_features, 0, args.stopwords, stopset, 0)

  else:
    # Finds word scores
    print '\nEvaluating the best %d word features\n' % (args.limit_features)
    word_scores = create_word_scores(args.positive, args.negative)
    classifier = train(best_word_features, args.positive, args.negative, args.limit_features, 0, args.stopwords, stopset, word_scores)

  pickle.dump(classifier, f)
  f.close()
  print 'Successfully wrote classifier to file ' + args.output + ".pickle!"


if __name__ == '__main__':
    main(sys.argv[1:])
