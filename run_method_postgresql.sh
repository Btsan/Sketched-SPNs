#!/bin/bash

# clusters=(1 5e-1 2e-1 1e-1 5e-2 1e-2) # 6
# widths=(4096 1e4 1e5 1e6) # 4
# methods=(count bound) # 2 x 2

dataset=$1 # stats or imdb
workload=$2 # stats_CEB.sql or job_light_queries.sql
method=$3 # stats-ceb_5x4096_min1e-1_count_exact, etc.

psql \
-d $dataset \
-U postgres \
-c "SET ml_joinest_fname='estimates/"$method".txt';" \
-f prefix.sql \
-f $workload \
&> "runtimes/"$method".log"

# docker exec -it -w /var/lib/pgsql/13.1/data/ ce-benchmark bash