from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk
from gensim.models import KeyedVectors
import sys
import time
import os

es = Elasticsearch([{'host':'localhost','port':9200}])

args = sys.argv
if (len(args) < 2):
    print ("Need 2 args ... filepath, binary/text ")
    sys.exit(0)
else:
    filepath = str(args[1])
    filename = os.path.basename(filepath)

    if (str(args[2]) == 'binary'):
        isBinary = True
    else:
        isBinary = False

def getWord():
    for word in embedModel.index2word:
        wv = embedModel.get_vector(word) 
        yield {
            "_index": "word-embeddings",
            "_type": "words",
            "_id" : filename + '_' + word,
            "word": word,
            "vector": wv.tolist(),
            "file": filename
            }
#
#   for k,v in status.__dict__.items():  #same thing as `vars(status)`
#       print k,v      =>    vectors vocab vector_size index2word vectors_norm
#
#   print (list(embedModel.vocab.keys())[0:10]) => ['</s>', 'in', 'for', 'that', 'is', 'on', '##', 'The', 'with', 'said']
#   print (list(embedModel.vocab.keys())[299990:299999]) => ['Sidman', 'Moroccan_Islamic_Combatant', 'cow_pastures', 'GFW', '#Q.####', 'Telluride_Film_Festival', 'Glavic', 'HemCon', 'Leggat']

#   print (embedModel.vocab['in'])          =>  Vocab(count:2999999, index:1)
#   print (embedModel.vocab['said'])        =>  Vocab(count:2999991, index:9)           NOT SURE WHAT THIS 'COUNT' & 'INDEX' EXACTLY ARE
#   print (embedModel.vocab['HemCon'])      =>  Vocab(count:2700003, index:299997)
#
#===============
#
#   print (embedModel.vectors.shape)        =>  (3000000, 300)
#   print (embedModel.vectors[0])
#       [ 1.1291504e-03 -8.9645386e-04  3.1852722e-04  1.5335083e-03 1.1062622e-03 -1.4038086e-03 ... ]
#
#===============
#
##      type (embedModel.vector_size)       =>  int
#       print (embedModel.vector_size)      =>  300
#
#===============
#
#type (embedModel.index2word)    =>  list
#print (embedModel.index2word[0:10])     =>  ['</s>', 'in', 'for', 'that', 'is', 'on', '##', 'The', 'with', 'said']
#
#===============
#type(embedModel.vectors_norm)   =>  NoneType
#

start = time.time()
embedModel = KeyedVectors.load_word2vec_format(filepath, binary=isBinary)
print('Finished loading original model %.2f min' % ((time.time()-start)/60))
print('word2vec: %d' % len(embedModel.vocab))
print('non-phrases: %d' % len([w for w in embedModel.vocab.keys() if '_' not in w]))

bulk(client=es, actions=getWord(),chunk_size=100,request_timeout=120)

