#!/usr/bin/env bash
set -x

wget https://ftp.postgresql.org/pub/source/v18.3/postgresql-18.3.tar.bz2
tar xvf postgresql-18.3.tar.bz2 && cd postgresql-18.3
patch -s -p1 < ../benchmark_pg18.patch && cd ..

echo "To build the image, now run:"
echo "    tar cvf postgresql-18.3.tar.gz postgresql-18.3 && mv postgresql-18.3.tar.gz dockerfile/"
echo "    sudo docker build -t ceb-18 dockerfile/"