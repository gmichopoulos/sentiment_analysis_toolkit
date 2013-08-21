# nbayes_sentiment.py
#
# usage: python nbayes_sentiment.py -p [pos_filename] -n [neg_filename]
# ----------------------------
# This is a module designed to be used with the train.py and classify.py scripts,
# but can also be run as a stand-alone script for testing the success of a
# Naive Bayes classifier using different feature extraction methods on input 
# positive and negative data files.
#
# Dependencies:
# ----------------------------
# nltk (sudo easy_install pip)
# nltk corpus (sudo python -m nltk.downloader -d /usr/share/nltk_data all) (needed for stop words)
# matplotlib (http://matplotlib.sourceforge.net/users/installing.html)
# pyroc (https://github.com/marcelcaraciolo/PyROC)
#
# Options:
# ----------------------------
# -p [positive_file_name] 
#    (path to newline separated list of positive input)
#
# -n [negative_file_name] 
#    (path to newLine separated list of negative input)
#
# -l [number_of_features]
#    (number of best features to use)
#
# -d [number_of divisions] 
#    (select the number of divisions created in input data: 1 out of d will 
#     be used for testing.)
#
# -a (train and test over each possible set of divisions and average the 
#     results for more smoothing)
#
# -b (use bigrams as features)
#
# -s (filter out stopwords from features)
#
# -r (randomize training data to reduce clumping )
#
# -g (graph the resulting ROC curves of each round of testing that occurs)
#
# -t (tag negated words with word_not)
#
# written by George Michopoulos, 7/19/13 
#

import re, math, random, collections, itertools, pickle, os, sys, argparse
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

###### CONSTANTS #########

NUM_FEATURES_TO_TEST = [100, 1000, 10000, 100000]

###### MODULE FUNCTIONS ######

def best_word_features(words, limit, stop, stopset, word_scores):

  best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, \
                                                reverse=True)[:limit]
  best_words = set([w for w, s in best_vals])

  if stop:
    return dict([(word, True) for word in words if word in best_words 
                                                and word not in stopset])

  return dict([(word, True) for word in words if word in best_words])


def bigram_word_features(words, limit, stop, stopset, word_score_placeholder, \
                                          score_fn=BigramAssocMeasures.chi_sq):
  if stop:
    words = [w for w in words if w not in stopset]
  bigram_finder = BigramCollocationFinder.from_words(words)
  bigrams = bigram_finder.nbest(score_fn, limit)
  return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


# Didn't seem to accomplish anything with my test data; let me know if it increases
# the accuracy of your classifiers, or if you make any changes that make it do so
def tag_negated_words(sentence):

  neglist = ['not', 'didn\'t', 'didnt', 'cant', 'can\'t', 'don\'t', 'dont', \
              'wouldn\'t', 'wouldnt', 'shouldn\'t', 'shouldnt', 'isn\'t', 'isnt']
  for i, word in enumerate(sentence):
    if word.lower() in neglist and i < len(sentence) - 1:
      sentence[i+1] = sentence[i+1] + "_not"
      del sentence[i]

  return sentence


def create_word_scores(pos, neg):
  # Create lists of all positive and negative words
  pos_words = []
  neg_words = []
  with open(pos, 'r') as pos_sentences:
    for i in pos_sentences:
      pos_word = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
      if args.tag_negative_words:
       pos_word = tag_negated_words(pos_word)
      pos_words.append(pos_word)
  with open(neg, 'r') as neg_sentences:
    for i in neg_sentences:
      neg_word = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
      if args.tag_negative_words:
       neg_word = tag_negated_words(neg_word)
      neg_words.append(neg_word)
  pos_words = list(itertools.chain(*pos_words))
  neg_words = list(itertools.chain(*neg_words))

  # Build frequency distibution of all words, positive, and negative labels
  word_fd = FreqDist()
  cond_word_fd = ConditionalFreqDist()
  for word in pos_words:
    word_fd.inc(word.lower())
    cond_word_fd[1].inc(word.lower())
  for word in neg_words:
    word_fd.inc(word.lower())
    cond_word_fd[0].inc(word.lower())

  # Finds word counts for all sets
  pos_word_count = cond_word_fd[1].N()
  neg_word_count = cond_word_fd[0].N()
  total_word_count = pos_word_count + neg_word_count

  # Builds dictionary of word scores based on chi-squared test
  word_scores = {}
  for word, freq in word_fd.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(cond_word_fd[1][word], \
                          (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(cond_word_fd[0][word], \
                          (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score

  return word_scores

def evaluate_features(feature_select, pos, neg, num_training_sets, avg, limit, \
                                    rand, stop, stopset, word_scores, ROC_data=None):
  pos_features = []
  neg_features = []

  with open(pos, 'r') as pos_sentences:
    for s in pos_sentences:
      pos_words = re.findall(r"[\w']+|[.,!?;]", s.rstrip())
      pos_words = [feature_select(pos_words, limit, stop, stopset, word_scores), 1]
      pos_features.append(pos_words)

  with open(neg, 'r') as neg_sentences:
    for s in neg_sentences:
      neg_words = re.findall(r"[\w']+|[.,!?;]", s.rstrip())
      neg_words = [feature_select(neg_words, limit, stop, stopset, word_scores), 0]
      neg_features.append(neg_words)

  if rand:
    random.shuffle(pos_features)
    random.shuffle(neg_features)

  # Call divide_and_test for each possible combo and average results
  if avg:
    avg_stats = [0,0,0,0,0,0]

    # Iterates through different training sets
    for t in xrange(num_training_sets):
      curr_stats = divide_and_test(pos_features, neg_features, t, \
                                   num_training_sets, limit, ROC_data)
      for i in xrange(len(avg_stats)):
        avg_stats[i] += curr_stats[i]

    for i in xrange(len(avg_stats)):
      avg_stats[i] /= num_training_sets

    print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
    print 'Completed training/testing on all %d sets' % num_training_sets
    print 'average accuracy:', avg_stats[0]
    print 'average pos precision:', avg_stats[1]
    print 'average pos recall:', avg_stats[2]
    print 'average neg precision:', avg_stats[3]
    print 'average neg recall:', avg_stats[4]
    if ROC_data: print 'average AUC:', avg_stats[5] 
    print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
  
  else:
    divide_and_test(pos_features, neg_features, 0, num_training_sets, \
                                                limit, ROC_data)


def divide_and_test(pos_features, neg_features, t, num_training_sets, num_features, ROC_data=None):

  # Selects features to be used for training and testing
  pos_start = int(math.floor(len(pos_features) * t/num_training_sets))
  neg_start = int(math.floor(len(neg_features) * t/num_training_sets))
  pos_cutoff = int(math.floor(len(pos_features) * (t + 1)/num_training_sets))
  neg_cutoff = int(math.floor(len(neg_features) * (t + 1)/num_training_sets))

  train_features = pos_features[:pos_start] + pos_features[pos_cutoff:] + \
                   neg_features[:neg_start] + neg_features[neg_cutoff:]
  test_features = pos_features[pos_start:pos_cutoff] + \
                  neg_features[neg_start:neg_cutoff]

  # Trains a Naive Bayes Classifier
  classifier = NaiveBayesClassifier.train(train_features)  

  # Initiates referenceSets and testSets
  reference_sets = collections.defaultdict(set)
  test_sets = collections.defaultdict(set) 

  # Puts correct sentences in referenceSets and the predicted ones in testsets
  for i, (features, label) in enumerate(test_features):
    reference_sets[label].add(i)
    predicted = classifier.classify(features)
    #print "predicted feature set:"+ str(features) + "as being " + str(label) 
    test_sets[predicted].add(i)  

  curr_accuracy = nltk.classify.util.accuracy(classifier, test_features)
  curr_pos_precision = nltk.metrics.precision(reference_sets[0], test_sets[1])
  curr_pos_recall = nltk.metrics.recall(reference_sets[1], test_sets[1])
  curr_neg_precision = nltk.metrics.precision(reference_sets[0], test_sets[0])
  curr_neg_recall = nltk.metrics.recall(reference_sets[0], test_sets[0])

  # Print ROC curve and AUC
  auc = 0
  if ROC_data:
    roc_data = ROCData((label, classifier.prob_classify(feature_set).prob(1)) \
                                    for feature_set, label in test_features)
    auc = roc_data.auc()
    ROC_data[0].append(roc_data)
    ROC_data[1].append(str(num_features) + " Features: set " + str(t + 1) + \
                      " of " + str(num_training_sets) + ", AUC = " + str(auc))

  #prints metrics to show how well the feature selection did
  print 'testing on %d of %d sets, from [positive] index %d to index %d:' \
                      % ((t + 1), num_training_sets, pos_start, pos_cutoff)
  print 'train on %d instances, test on %d instances' \
                               % (len(train_features), len(test_features))
  print 'accuracy:', curr_accuracy
  print 'pos precision:', curr_pos_precision
  print 'pos recall:', curr_pos_recall
  print 'neg precision:', curr_neg_precision
  print 'neg recall:', curr_neg_recall
  if ROC_data: print 'AUC:', auc
  classifier.show_most_informative_features(10)

  return [curr_accuracy, curr_pos_precision, curr_pos_recall, 
          curr_neg_precision, curr_neg_recall, auc]



###### MAIN ######

def main(argv):

  parser = argparse.ArgumentParser(description="Run sentiment analysis using\
                                      a positive and a negative input file")

  parser.add_argument("-p", "--positive", help="input relative path of a \
                      positive data file", default=POS_FILE)

  parser.add_argument("-n", "--negative", help="input relative path of a \
                      negative data file", default=NEG_FILE)

  parser.add_argument("-d", "--divisions", type=int, help="select the number \
                      of divisions created in input data: 1 out of d will \
                      be used for testing.", default=4)

  parser.add_argument("-l", "--limit_features", type=int, help="number of best \
                      features to use", default="0")

  parser.add_argument("-b", "--bigram", help="classify using bigram features.",
                      action="store_true")

  parser.add_argument("-s", "--stopwords", help="filter out stop words before \
                      training.", action="store_true")

  parser.add_argument("-t", "--tag_negative_words", help="tag negated words with \
                      word_not to capture more meaning.", action="store_true")

  parser.add_argument("-r", "--randomize", help="randomize training data to \
                      reduce clumping while training.", action="store_true")

  parser.add_argument("-a", "--average", help="train and test over each \
                      possible set of divisions and average the results for \
                      more smoothing.", action="store_true")

  parser.add_argument("-g", "--graph", help="graphs the resulting ROC curves \
                      against eachother", action="store_true")

  args = parser.parse_args()

  # Set up ROC graphing data and import pyroc as needed
  if args.graph:
    from pyroc import *
    ROC_data = [[],[]]

  # Set up stopword set
  stopset = []
  if args.stopwords:
    from nltk.corpus import stopwords
    print "Stop words are being filtered out."
    stopset = set(stopwords.words('english'))

  # Finds word scores if not using bigrams
  word_scores = []
  if not args.bigram:
    word_scores = create_word_scores(args.positive, args.negative)

  # Check to see what mode of testing is being used; input feature limit:
  if args.limit_features:
    limit = args.limit_features

    if (args.bigram):
      print '\nEvaluating the best %d bigram word features\n' % (limit)
      evaluate_features(bigram_word_features, args.positive, args.negative, \
                        args.divisions, args.average, limit, args.randomize, \
                        args.stopwords, stopset, word_scores, ROC_data)

    else:
      print '\nEvaluating the best %d word features\n' % (limit)
      evaluate_features(best_word_features, args.positive, args.negative, \
                        args.divisions, args.average, limit, args.randomize, \
                        args.stopwords, stopset, word_scores, ROC_data)

  # Or iteration through default array of feature numbers
  else:
    for limit in NUM_FEATURES_TO_TEST:
      if (args.bigram):
        print '\nEvaluating the best %d bigram word features\n' % (limit)
        evaluate_features(bigram_word_features, args.positive, args.negative, \
                        args.divisions, args.average, limit, args.randomize, \
                        args.stopwords, stopset, word_scores, ROC_data)
      else:
        print '\nEvaluating the best %d word features\n' % (limit)
        evaluate_features(best_word_features, args.positive, args.negative, \
                        args.divisions, args.average, limit, args.randomize, \
                        args.stopwords, stopset, word_scores, ROC_data)

  if args.graph:
      plot_multiple_roc(ROC_data[0],'ROC Curves', labels = ROC_data[1])

if __name__ == '__main__':
    main(sys.argv[1:])
