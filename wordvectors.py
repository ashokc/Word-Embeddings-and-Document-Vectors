
import logging
logger = logging.getLogger(__name__)
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

__all__ = ['WordVectors']

class WordVectors():

    es = Elasticsearch([{'host':'localhost','port':9200}])
    corpusWordVectors = {}
    corpusVocab = []
    tokenType = 'stopped'
    wordCorpus = ''
    wordVecSource = ''

    es_logger = logging.getLogger('elasticsearch')
    es_logger.setLevel(logging.WARNING)

    def __init__(self, wordCorpus = 'twenty-news', wordVecSource = 'fasttext', corpusVocab = None, tokenType='stopped'):
        WordVectors.wordCorpus = wordCorpus
        WordVectors.wordVecSource = wordVecSource
        WordVectors.corpusVocab = corpusVocab
        WordVectors.tokenType = tokenType
        self.getEmbedSource (wordCorpus, wordVecSource)
        self.initializeWordVectorsFromSource(corpusVocab)

    def getEmbedSource (self, wordCorpus, wordVecSource):
        self.wvLength = 300
        if (wordCorpus == 'twenty-news'):
            self.esIndex = 'twenty-vectors'
            self.queryClause = ',{ "term" : {"tokenType" : "' + WordVectors.tokenType + '"} }'
            if (wordVecSource == 'custom-vectors-word2vec'):
                self.embedSource = 'twenty_news_word2vec_sgns_model'
            elif (wordVecSource == 'custom-vectors-fasttext'):
                self.embedSource = 'twenty_news_fasttext_model'
        elif (wordCorpus == 'acl-imdb'):
            self.esIndex = 'imdb-vectors'
            self.queryClause = ',{ "term" : {"tokenType" : "' + WordVectors.tokenType + '"} }'
            if (wordVecSource == 'custom-vectors-word2vec'):
                self.embedSource = 'imdb_word2vec_sgns_model'
            elif (wordVecSource == 'custom-vectors-fasttext'):
                self.embedSource = 'imdb_fasttext_model'

        if (wordVecSource == 'google'):
            self.queryClause = ''
            self.esIndex = 'word-embeddings'
            self.embedSource = 'GoogleNews-vectors-negative300.bin'
            self.wvLength = 300
        elif (wordVecSource == 'glove'):
            self.queryClause = ''
            self.esIndex = 'word-embeddings'
            self.embedSource = 'glove.6B.300d.txt.word2vec'
            self.wvLength = 300
        elif (wordVecSource == 'fasttext'):
            self.queryClause = ''
            self.esIndex = 'word-embeddings'
            self.embedSource = 'crawl-300d-2M-subword.vec'
            self.wvLength = 300

    def getWordVectors (self,vocab):
        zeroVector = np.zeros(self.wvLength)     # a zero vector of length wvLength
        wordVectors = []
        nZero = 0
        for word in vocab:      # these vocab words list is already sorted in the same order as the Doc-Term matrix X
            if (word in WordVectors.corpusVocab):
                wordVectors.append(WordVectors.corpusWordVectors[word])
            else:
                wordVectors.append(zeroVector)
                nZero = nZero + 1
        logger.info('# of words without embeddings:{}'.format(nZero))
        wordVectors=np.array([np.array(xi) for xi in wordVectors])
        return wordVectors
            
    def initializeWordVectorsFromSource (self, corpusVocab):
        zeroVector = np.zeros(self.wvLength)     # a zero vector of length wvLength
        nZero = 0 
        for word in corpusVocab:
            query = '{ "query" : { "bool" : { "must" : [{ "term" : {"word" : "' + word + '"} }' + self.queryClause +  ',{ "term" : {"file.keyword" : "' + self.embedSource + '"} } ] } } }'
#            logger.info('query:{}'.format(query))
            response = WordVectors.es.search(index=[self.esIndex],doc_type=['words'],body=query)
            total = response['hits']['total']
            if (total == 1):
                wv = response['hits']['hits'][0]['_source']['vector']
                WordVectors.corpusWordVectors[word] = wv
            elif (total == 0):
                WordVectors.corpusWordVectors[word] = zeroVector
                nZero = nZero + 1
            else:
                logger.error ('More than one word vector for word: {}'.format(word))
        logger.info ('TOTAL # words in corpus without embeddings:{}'.format(nZero))


