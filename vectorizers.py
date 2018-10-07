
# import modules & set up logging
import logging
logger = logging.getLogger(__name__)

import time

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix

np.set_printoptions(threshold=np.nan)

__all__ = ['VectorizerWrapper', 'Transform2WordVectors']

def logTrace(message, args):
    logger.info('In {}. #args:{}'.format(message, len(args)))
    for arg in args:
        logger.info('\t{}'.format(type(arg)))

class VectorizerWrapper (TransformerMixin, BaseEstimator):
    def __init__(self, model):
        self.model = model

    def fit (self, *args):
        logTrace ('VectorizerWrapper:fit', args)
        self.model.fit (args[0], args[1])
        return self

    def transform (self, *args):
        logTrace ('VectorizerWrapper:transform.', args)
        return {'sparseX': self.model.transform(args[0]), 'vocab': self.model.vocabulary_}

class Transform2WordVectors (BaseEstimator, TransformerMixin):

    wvObject = None

    def __init__(self, wvObject = None):
        Transform2WordVectors.wvObject = wvObject

    def fit (self, *args):
        logTrace ('Transform2WordVectors:fit.', args)
        return self

    def transform(self, *args):
        logTrace ('Transform2WordVectors:transform.', args)
        sparseX = args[0]['sparseX']
        if (not Transform2WordVectors.wvObject):         # No transformation
            return sparseX
        else:
            vocab = args[0]['vocab']            # key: vocab val: {'thu': 62368, 'apr': 2787, 'date': 13236, 
            sortedWords = sorted(vocab, key=vocab.get)    # sorting so they would be in the same order as the Doc-Term matrix
            logger.info('sortedWords. type:{}, size:{}'.format(type(sortedWords), len(sortedWords)))
            wordVectors = Transform2WordVectors.wvObject.getWordVectors (sortedWords) # nDocs_in_this_set x nWords_in_this_set
            logger.info('wordVectors. type:{}, shape:{}'.format(type(wordVectors),wordVectors.shape))

            reducedMatrix = self.sparseMultiply (sparseX, wordVectors)
            logger.info('reducedMatrix. type:{}, shape:{}'.format(type(reducedMatrix),reducedMatrix.shape))
        return reducedMatrix

    def sparseMultiply (self,sparseX, wordVectors):
        wvLength = len(wordVectors[0])
        reducedMatrix = []
        for row in sparseX:
            newRow = np.zeros(wvLength)
            for nonzeroLocation, value in list(zip(row.indices, row.data)):
                newRow = newRow + value * wordVectors[nonzeroLocation]
            reducedMatrix.append(newRow)
        reducedMatrix=np.array([np.array(xi) for xi in reducedMatrix])
        return reducedMatrix

