#!/bin/bash

clusters=(1 5e-1 1e-1 5e-2 1e-2)
widths=(4096 1e4 1e5)

for c in "${clusters[@]}"
do
    for w in "${widths[@]}"
    do
        echo "running JOB-light: cluster = "$c" sketch width = "$w
        echo .
        echo .
        echo .

        python3 sketched_spn.py \
        --depth 5 \
        --width $w \
        --data ./imdb/ \
        --workload job_light_sub_query_with_star_join.sql.txt \
        --experiment job-light \
        --independence 128 \
        --min_cluster $c \
        --writefile "./job-light_"$w"_min"$c"_times.csv" \
        &> "./job-light_"$w"_min"$c"_times.log"
    done
done