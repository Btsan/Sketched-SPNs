#!/bin/bash

clusters=(1 5e-1 1e-1 5e-2 1e-2)
widths=(4096 1e4 1e5)

for c in "${clusters[@]}"
do
    for w in "${widths[@]}"
    do
        echo "running Stats-CEB: min. cluster = "$c" sketch width = "$w
        echo .
        echo .
        echo .

        python3 sketched_spn.py \
        --depth 5 \
        --width $w \
        --data ./End-to-End-CardEst-Benchmark-master/datasets/stats_simplified/ \
        --workload ./stats_CEB_sub_queries_corrected.sql \
        --experiment stats-ceb \
        --independence 128 \
        --min_cluster $c \
        --writefile "./stats-ceb_"$w"_min"$c"_times.csv" \
        > "./stats-ceb_"$w"_min"$c"_times.log"
    done
done