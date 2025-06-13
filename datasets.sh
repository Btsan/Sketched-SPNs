#!/bin/bash

# download Stats-CEB dataset
curl -L -o End-to-End-CardEst-Benchmark.zip https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/archive/refs/heads/master.zip
unzip End-to-End-CardEst-Benchmark.zip
rm End-to-End-CardEst-Benchmark.zip

# download IMDb for JOB-light (old link broke http://homepages.cwi.nl/~boncz/job/imdb.tgz)
curl -L -o imdb.tgz http://event.cwi.nl/da/job/imdb.tgz 
mkdir End-to-End-CardEst-Benchmark-master/datasets/imdb/
tar zxvf imdb.tgz -C End-to-End-CardEst-Benchmark-master/datasets/imdb/
rm imdb.tgz