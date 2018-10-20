from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string

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
    for i, article in enumerate(twenty_news['data']):
        stopped = removeStopWords (tokenize (article))
        stemmed = stem (stopped)
        fileName = twenty_news['filenames'][i] 
        groupIndex = twenty_news['target'][i] 
        groupName = twenty_news['target_names'][groupIndex]
        yield {
                "_index": "twenty-news",
                "_type": "article",
                "original": article, "stopped" : stopped, "stemmed" : stemmed, "groupIndex": str(twenty_news['target'][i]), "groupName" : twenty_news['target_names'][groupIndex], "fileName" : twenty_news['filenames'][i]
                }
es = Elasticsearch([{'host':'localhost','port':9200}])
nltk_stopw = stopwords.words('english')
stemmer = SnowballStemmer("english")

#
# twenty_news is a 'bunch' object... The returned dataset is a scikit-learn “bunch”: a simple holder object with fields that can be both accessed as python dict keys or object attributes for convenience, for instance the target_names holds the list of the requested category names:
#
#   twenty_news.keys()    =>  dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR', 'description'])
#

#   print (type(twenty_news))                          <class 'sklearn.utils.Bunch'>
#   print (type(twenty_news['data']))                  <class 'list'>
#   print (type(twenty_news['target_names']))          <class 'list'>
#   print (type(twenty_news['target']))                <class 'numpy.ndarray'>
#   print (type(twenty_news['filenames']))             <class 'numpy.ndarray'>

#
#           print (twenty_news['filenames'])       11314 files
#
#       ['/home/ashokc/scikit_learn_data/20news_home/20news-bydate-train/rec.autos/102994' '/home/ashokc/scikit_learn_data/20news_home/20news-bydate-train/comp.sys.mac.hardware/51861' '/home/ashokc/scikit_learn_data/20news_home/20news-bydate-train/comp.sys.mac.hardware/51879' ...  '/home/ashokc/scikit_learn_data/20news_home/20news-bydate-train/comp.sys.ibm.pc.hardware/60695' '/home/ashokc/scikit_learn_data/20news_home/20news-bydate-train/comp.graphics/38319' '/home/ashokc/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104440']
#
#           print (twenty_news['target_names'])        20 of them
#
#       ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
#
#           print (twenty_news['target']) #          integer index for classification purposes...
#
#            [7 4 4 ... 3 1 8]      for example '/home/ashokc/scikit_learn_data/20news_home/20news-bydate-train/rec.autos/102994' => 'rec.autos' => 7
#
#       print (len(twenty_news['data']))               => 11314
#       print (len(twenty_news['target']))             => 11314
#       print (len(twenty_news['target_names']))       => 20
#       print (len(twenty_news['filenames']))         => 11314
#

#twenty_news = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
#bulk(es, getArticle('train'))
#twenty_news = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
#bulk(es, getArticle('test'))

twenty_news = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
bulk(es, getArticle())

