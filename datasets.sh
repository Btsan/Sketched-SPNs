#!/bin/bash

# download Stats-CEB dataset
curl -L -o End-to-End-CardEst-Benchmark.zip https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/archive/refs/heads/master.zip
unzip End-to-End-CardEst-Benchmark.zip
rm End-to-End-CardEst-Benchmark.zip

# download IMDb for JOB-light
curl -L -o imdb.tgz http://homepages.cwi.nl/~boncz/job/imdb.tgz
mkdir imdb
tar zxvf imdb.tgz -C imdb
rm imdb.tgz