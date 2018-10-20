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
            "_index": "imdb-vectors",
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
    hits = scan (es, query=query, index="acl-imdb", doc_type="article")
    for hit in hits:
        listOfArticles.append(hit['_source'][tokenType])

    return listOfArticles

#
# To explore an object in python use __dict__       print (OBJECT.__dict__)
#

#
# train word2vec on the two sentences
#
# min_count is # of times a word has to occur in the WHOLE CORPUS for it to be considered... for embedding. we take it as 2 here... all else default. SGNS with window=5 & negative sampling=5
#
#
#   wv, vocabulary useful attrribs
#       wv  =>  vectors     type (wv.vectors)   =>  numpy.ndarray           wv.vectors.shape    =>  (13801, 300)
#       wv  =>  vocab       type (wv.vocab) =>  dict                        len(wv.vocab)   =>  13801
#               'vector_size': 300,
#                'index2word': ['use', 'one', 'would', 'max', 'like', ....
#
#                len(wv.index2word)      =>  13801 
#
# save the model

#
#	word2vec / SGNS
#

min_count = 2

filename = 'imdb_word2vec_sgns_model'
for tokenType in ['stemmed', 'stopped']:
    embedModel = Word2Vec(getTokens(), size=300, sg=1, min_count=min_count, window=5, negative=5)
    bulk(client=es, actions=getWord(),chunk_size=50,request_timeout=120)

#
#	fasttText
#

filename = 'imdb_fasttext_model'
for tokenType in ['stemmed', 'stopped']:
    embedModel = FastText(getTokens(), size=300, sg=1, min_count=min_count, window=5, negative=5)
    bulk(client=es, actions=getWord(),chunk_size=50,request_timeout=120)


