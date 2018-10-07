
# import modules & set up logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy as np

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

__all__ = ['Tokens']

class Tokens():

    es = Elasticsearch([{'host':'localhost','port':9200}])
    es_logger = logging.getLogger('elasticsearch')
    es_logger.setLevel(logging.WARNING)

    def __init__(self, dataSource):
        if (dataSource == 'twenty-news'):
            self.esIndex = 'twenty-news'
        elif (dataSource == 'acl-imdb'):
            self.esIndex = 'acl-imdb'

    def getTokens(self,tokenType, split=None):
        X, y, classNames = [], [], set()
        docType = 'article'
        if (split):
            query = { "query": { "term" : {"split" : split} }, "_source" : [tokenType, 'groupIndex', 'groupName'] }
        else:
            query = { "query": { "match_all" : {} }, "_source" : [tokenType, 'groupIndex', 'groupName'] }
        hits = scan (self.es, query=query, index=self.esIndex, doc_type=docType, request_timeout=120)
        for hit in hits:
            X.append(hit['_source'][tokenType])
            y.append(hit['_source']['groupIndex'])
            classNames.add(hit['_source']['groupName'])
        X=np.array([np.array(xi) for xi in X])          #   rows: Docs. columns: words
        return X, np.array(y), sorted(list(classNames))

