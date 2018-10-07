import logging
import initLogs

from wordvectors import WordVectors
from vectorizers import VectorizerWrapper, Transform2WordVectors
from tokens import Tokens

import numpy as np

from tabulate import tabulate

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

import sys
import time

def getNeuralNet (nHidden, neurons):
    neuralNet = (neurons,)
    for i in range(2, nHidden):
        neuralNet = neuralNet + (neurons,)
    return neuralNet

def processArgs():
    args = sys.argv
    suppliedArgs = len(args) - 1
    requiredArgs = 5
    if (suppliedArgs != requiredArgs):
        logger.critical ('Need 5 args: wordCorpus, min_df, model, tokenType, wordVecSource... Exiting')
        sys.exit(0)
    else:
        wordCorpus = str(args[1])       # twenty-news
        min_df = int(args[2])           # 2, 3 etc...
        model = str(args[3])            # nb,linearSvc,mlp, etc...
        tokenType = str(args[4])        # stemmed, stopped
        wordVecSource = str(args[5])    # none, twenty-vectors-fasttext, twenty-vectors-word2vec, google, fasttext, glove
    
        if ( (model == 'nb') and (wordVecSource != 'none') ):
            logger.error('Run Naive Bayes without word vectors... exiting')
            sys.exit(0)

        if (wordVecSource == 'none'):
            wordVecSource = None

        if ( ( (wordVecSource == 'google') or (wordVecSource == 'fasttext') or (wordVecSource == 'glove') ) and (tokenType != 'stopped') ):
            logger.error('For generic embedding use stopped words only... exiting')
            sys.exit(0)

    return wordCorpus, min_df, model, tokenType, wordVecSource

def defineModels(min_df, model, wvObject = None):
    vectorizers = [ ('counts', ("vectorizer", VectorizerWrapper(model=CountVectorizer(analyzer=lambda x: x, min_df=min_df)))), ('tfidf', ("vectorizer", VectorizerWrapper(model=TfidfVectorizer(analyzer=lambda x: x, min_df=min_df)))) ]
    transformer = ('transformer', Transform2WordVectors(wvObject = wvObject))
    classifiers = { 'nb' : ("nb", MultinomialNB()), 'linearsvc' : ("linearsvc", LinearSVC()) }
    if (model == 'mlp'):
        mlpClassifiers = []
        for nHidden in [1, 2, 3]:
            for neurons in [50, 100, 200]:
                name = str(nHidden) + '-' + str(neurons)
                mlpClf = (name, MLPClassifier(hidden_layer_sizes=getNeuralNet(nHidden, neurons),verbose=False))
                mlpClassifiers.append(mlpClf)
        classifiers['mlp'] = mlpClassifiers

    modelRuns = []
    for vectorizer in vectorizers:
        if (wvObject):
            name = 'vectorizer-' + vectorizer[0] + '-embed_' + wvObject.wordVecSource + '-' + model
        else:
            name = 'vectorizer-' + vectorizer[0] + '-no-embed-' + model
        if (model == 'mlp'):
            for mlpClf in classifiers['mlp']:
                modelRun = (name + '-' + mlpClf[0], Pipeline([vectorizer[1], transformer, mlpClf]))
                modelRuns.append (modelRun)
        else:
            modelRun = (name, Pipeline([vectorizer[1], transformer, classifiers[model]]))
            modelRuns.append (modelRun)
    return modelRuns

def main():
    start0 = time.time()

    wordCorpus, min_df, model, tokenType, wordVecSource = processArgs()
    logger.info('Running: WordCorpus: {}, Models: {}, TokenType: {}, min_df: {}, wordVecSource: {}'.format(wordCorpus, model, tokenType, min_df, wordVecSource))

    X, y, classNames = Tokens(wordCorpus).getTokens(tokenType)
    vocabularyGenerator = CountVectorizer(analyzer=lambda x: x, min_df=min_df).fit(X) # This is only to generate a vocabulary with min_df
    corpusVocab = sorted(vocabularyGenerator.vocabulary_, key=vocabularyGenerator.vocabulary_.get)
    logger.info('Total Corpus Size: len(corpusVocab) with frequency > min_df : {}, X.shape: {}, y.shape: {}, # classes: {}'.format(len(corpusVocab), X.shape, y.shape, len(classNames)))
    logger.info('Class Names:{}'.format(classNames))
    if (wordVecSource):
        wvObject = WordVectors(wordCorpus=wordCorpus, wordVecSource=wordVecSource,corpusVocab=corpusVocab,tokenType=tokenType) # nWords_in_this_set X wvLength
    else:
        wvObject = None
    results = {}
    results['timeForDataFetch'] = time.time() - start0
    logger.info('Time Taken For Data Fetch: {}'.format(results['timeForDataFetch']))

    modelRuns = defineModels(min_df, model, wvObject)
    logger.info ('Model Runs:\n{}'.format(modelRuns))

    if (wordCorpus == 'twenty-news'):
        testDataFraction = 0.2
        sss = StratifiedShuffleSplit(n_splits=1, test_size=testDataFraction, random_state=0)
        sss.get_n_splits(X, y)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    elif (wordCorpus == 'acl-imdb'):
        X_train, y_train, classNames = Tokens(wordCorpus).getTokens(tokenType,'train')
        X_test, y_test, classNames = Tokens(wordCorpus).getTokens(tokenType,'test')

    marker = 'X y vocab: Train => Test:' + str(X_train.shape) + ',' + str(y_train.shape) + '=>' + str(X_test.shape) + ',' + str(y_test.shape)
    for name, model in modelRuns:
        results[name] = {}
        results[name][marker] = {}
        logger.info('\n\nCurrent Run: {} => {}'.format(name, marker))
        start = time.time()
        logger.info("Training Begin")
        model.fit(X_train, y_train)
        logger.info("Training End")
        logger.info("Prediction Begin")
        predicted = model.predict(X_test)
        logger.info("Prediction End")
#`
#   nclasses x nclasses matrix.`
#       
#           M_ij    =>  Truly 'i' but predicted as 'j'. C_ii => TP.
#               class_i =>  class[groupIndex]           =>  so classNames go top->bottom & left->right
#               
#               sum row i - C_ii       =>   Predicted as NOT 'i' when it should be 'i'  =>  FN  =>  C_ii/(sum_row_i) = recall
#               sum column i - C_ii    =>  Predicted as 'i' when it should NOT be 'i' =>    FP     => C_ii/(sum_column_i) = precision
#`
        results[name][marker]['model_vocabulary_size'] = len(model.named_steps['vectorizer'].model.vocabulary_)
        results[name][marker]['confusion_matrix'] = confusion_matrix(y_test, predicted)
        results[name][marker]['timeForThisModel_fit_predict'] = time.time()-start

        logger.info ('Run:{}, {}, Confusion Matrix:\n{}'.format(name,marker, results[name][marker]['confusion_matrix']))
        logger.info ('Run:{}, {}, Classification Report:\n{}'.format(name,marker,classification_report(y_test, predicted, target_names=classNames)))
        logger.info ('Model Vocab Size:{}'.format(results[name][marker]['model_vocabulary_size']))
        logger.info ('Time Taken For This Model Run:{}'.format(results[name][marker]['timeForThisModel_fit_predict']))

    results['overAllTimeTaken'] = time.time() - start0
    logger.info('Overall Time Taken:{}'.format(results['overAllTimeTaken']))
    logger.info("Prediction End")

if __name__ == '__main__':
    initLogs.setup()
    logger = logging.getLogger(__name__)
    np.set_printoptions(linewidth=100)
    main()

