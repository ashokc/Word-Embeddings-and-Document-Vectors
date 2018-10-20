from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk

from sklearn.datasets import fetch_20newsgroups

def untokenize (tokens):
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

def tokenize (text):        #   no punctuation & starts with a letter & between 2-15 characters in length
    tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return tokens

def removeStopWords (tokens):
    filteredTokens = [f.lower() for f in tokens if f and f.lower() not in nltk_stopw]
    return filteredTokens

def stem (filteredTokens):      # stemmed & > 2 letters
    return [stemmer.stem(token) for token in filteredTokens if len(token) > 1]

def getArticle():
    for dataset in ['train', 'test']:
        for classIndex, directory in enumerate(['neg', 'pos']):
            dirName = "./" + dataset + "/" + directory
            for reviewFile in os.listdir(dirName):
                with open (dirName + '/' + reviewFile, 'r') as f:
                    article = f.read()
                stopped = removeStopWords (tokenize (article))
                stemmed = stem (stopped)
                fileName = dirName + '/' + reviewFile
                groupIndex = classIndex
                groupName = directory
                yield {
                    "_index": "acl-imdb",
                    "_type": "article",
                    "original": article, "stopped" : stopped, "stemmed" : stemmed, "split" : dataset, "groupIndex": groupIndex, "groupName" : directory, "fileName" : fileName
                }

es = Elasticsearch([{'host':'localhost','port':9200}])
nltk_stopw = stopwords.words('english')
stemmer = SnowballStemmer("english")

bulk(es, getArticle())

