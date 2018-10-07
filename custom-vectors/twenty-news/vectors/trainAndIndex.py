# import modules & set up logging
import logging
from gensim.models import Word2Vec
from gensim.models import FastText
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy as np

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk
from elasticsearch.helpers import scan
es = Elasticsearch([{'host':'localhost','port':9200}])

def getWord():
    for word in embedModel.wv.index2word:
        vector = embedModel.wv.get_vector(word)
        yield {
            "_index": "twenty-vectors",
            "_type": "words",
            "_id" : filename + '_' + word + '_' + tokenType,
            "tokenType": tokenType,
            "word": word,
            "vector": vector.tolist(),
            "file": filename
            }

def getTokens():
    listOfArticles = []
    query = { "query": { "match_all" : {} }, "_source" : [tokenType]}
    hits = scan (es, query=query, index="twenty-news", doc_type="article")
    for hit in hits:
        listOfArticles.append(hit['_source'][tokenType])

    return listOfArticles

min_count = 2

filename = 'twenty_news_word2vec_sgns_model'
for tokenType in ['stemmed', 'stopped']:
    embedModel = Word2Vec(getTokens(), size=300, sg=1, min_count=min_count, window=5, negative=5)
    bulk(client=es, actions=getWord(),chunk_size=50,request_timeout=120)

#
#	fasttText
#

filename = 'twenty_news_fasttext_model'
for tokenType in ['stemmed', 'stopped']:
    embedModel = FastText(getTokens(), size=300, sg=1, min_count=min_count, window=5, negative=5)
    bulk(client=es, actions=getWord(),chunk_size=50,request_timeout=120)


