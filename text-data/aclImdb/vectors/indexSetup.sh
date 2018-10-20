#!/bin/bash

CURL=/usr/bin/curl
port=9200
host=localhost
index=$1

$CURL -XPUT -H "Content-Type: application/json; charset=UTF-8" http://$host:$port/$index -d @$index.json

