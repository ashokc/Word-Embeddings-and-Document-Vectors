#!/bin/bash

#wordCorpus="twenty-news"
wordCorpus="acl-imdb"
min_df=2
for model in mlp linearsvc nb; do
	for tokenType in stopped stemmed; do
		for wordVecSource in glove fasttext google custom-vectors-fasttext custom-vectors-word2vec; do
			outfile=$model"-"$tokenType"-"$wordVecSource".out"
			echo "pipenv run python classify.py $wordCorpus $min_df $model $tokenType $wordVecSource"
			exit 0
			runStart=`date +%s`
#			pipenv run python classify.py $wordCorpus $min_df $model $tokenType $wordVecSource
			runEnd=`date +%s`
			runtime=$((runEnd-runStart))
			echo "Time taken for results/$outfile: $runtime"
			mv logs/classify.log results/$outfile
		done
	done
done

