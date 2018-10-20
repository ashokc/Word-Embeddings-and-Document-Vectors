# Word Embeddings and Document Vectors

This is the source code to along with the series of blog articles

* [Word Embeddings and Document Vectors: Part 1. Similarity](http://xplordat.com/2018/09/27/word-embeddings-and-document-vectors-part-1-similarity/)
* [Word Embeddings and Document Vectors: Part 2. Classification](http://xplordat.com/2018/10/09/word-embeddings-and-document-vectors-part-2-classification/)

The code employs,

* Elasticsearch (localhost:9200) as the repository
   1. to save tokens to, and get them as needed. 
   2. to save word-vectors (pre-trained or custom) to, and get them as needed. 

* See the Pipfle for Python dependencies

## Usage

1. Generate tokens for the 20-news corpus & the movie review data set and save them to Elasticsearch.
	* The dataset for 20-news is downloaded as part of the script. But you need to download the [movie review dataset](http://ai.stanford.edu/~amaas/data/sentiment/) separately.
	* The shell script & python code in the folders *text-data/twenty-news* & *text-data/acl-imdb*

2. Generate custom word vectors for the two text corpus in 1 above and save them to Elasticsearch. *text-data/twenty-news/vectors* & *text-data/acl-imdb/vectors* directories have the scripts

3. Process pre-trained vectors and save them to Elasticsearch. Look into *pre-trained-vectors/* for the code. You need to download the actual published vectors from their sources. We have used [Word2Vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing), [Glove](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip) and [FastText](https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip) in these articles.

4. The script *run.sh* can be configured to run whichever combination of the pipeline steps,. The logs contain the F-scores and timing results.




