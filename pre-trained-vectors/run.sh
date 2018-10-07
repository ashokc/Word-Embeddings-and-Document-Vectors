#!/bin/bash

echo "pipenv run python indexEmbeds.py fasttext/crawl-300d-2M-subword.vec text > fasttext.out"
pipenv run python indexEmbeds.py fasttext/crawl-300d-2M-subword.vec text > fasttext.out

echo "pipenv run python indexEmbeds.py google/GoogleNews-vectors-negative300.bin binary > google.out"
pipenv run python indexEmbeds.py google/GoogleNews-vectors-negative300.bin binary > google.out

echo "pipenv run python indexEmbeds.py glove/glove.6B.300d.txt.word2vec text > glove.out"
pipenv run python indexEmbeds.py glove/glove.6B.300d.txt.word2vec text > glove.out

