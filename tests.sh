#!/bin/bash

test_dir="./test_percentile"
mkdir -p $test_dir

clusters=(2e-1 1e-1 5e-2 1e-2) # must be in scientific notation
widths=(1e4 1e5 1e6)
decompose=(5e-2 1e-2) # must be in scientific notation
percentiles=(1 9e-1 75e-2 5e-1) # must be in scientific notation

for w in "${widths[@]}"
do
    for p in "${percentiles[@]}"
    do
        filename=$test_dir"/stats-ceb_5x"$w"_p"$p
        
        # if [ ! -e $filename"_count_exact.csv" ]; then
        python3 sketched_spn.py \
        --depth 5 \
        --width $w \
        --pickle ./features/ \
        --independence 32 \
        --percentile $p \
        --method count-sketch \
        --exact \
        --writefile $filename"_count_exact.csv" \
        --cuda \
        > $filename"_count_exact.log"
        # else
        #     echo "skipped count-sketch exact"
        # fi

        # if [ ! -e $filename"_bound_exact.csv" ]; then
        #     python3 sketched_spn.py \
        #     --depth 5 \
        #     --width $w \
        #     --pickle ./features/ \
        #     --independence 32 \
        #     --percentile $p \
        #     --method bound-sketch \
        #     --exact \
        #     --writefile $filename"_bound_exact.csv" \
        #     --cuda \
        #     > $filename"_bound_exact.log"
        # else
        #     echo "skipped bound-sketch exact"
        # fi

        for c in "${clusters[@]}"
        do
            for d in "${decompose[@]}"
            do
                echo "running Stats-CEB: min. cluster = "$c" sketch width = "$w
                echo .
                echo .
                echo .

                filename=$test_dir"/stats-ceb_5x"$w"_decompose"$d"_min"$c"_p"$p

                # if [ ! -e $filename"_count.csv" ]; then
                python3 sketched_spn.py \
                --depth 5 \
                --width $w \
                --pickle ./features/ \
                --independence 32 \
                --min_cluster $c \
                --decompose $d \
                --percentile $p \
                --method count-sketch \
                --writefile $filename"_count.csv" \
                --cuda \
                > $filename"_count.log"
                # else
                #     echo "skipped count-sketch"
                # fi

                # if [ ! -e $filename"_count_pessimistic.csv" ]; then
                python3 sketched_spn.py \
                --depth 5 \
                --width $w \
                --pickle ./features/ \
                --independence 32 \
                --min_cluster $c \
                --decompose $d \
                --percentile $p \
                --method count-sketch \
                --pessimistic \
                --writefile $filename"_count_pessimistic.csv" \
                --cuda \
                > $filename"_count_pessimistic.log"
                # else
                #     echo "skipped count-sketch pessimistic"
                # fi

                # if [ ! -e $filename"_bound.csv" ]; then
                #     python3 sketched_spn.py \
                #     --depth 5 \
                #     --width $w \
                #     --pickle ./features/ \
                #     --independence 32 \
                #     --min_cluster $c \
                #     --decompose $d \
                #     --percentile $p \
                #     --method bound-sketch \
                #     --writefile $filename"_bound.csv" \
                #     --cuda \
                #     > $filename"_bound.log"
                # else
                #     echo "skipped bound-sketch"
                # fi

                # if [ ! -e $filename"_bound_pessimistic.csv" ]; then
                #     python3 sketched_spn.py \
                #     --depth 5 \
                #     --width $w \
                #     --pickle ./features/ \
                #     --independence 32 \
                #     --min_cluster $c \
                #     --decompose $d \
                #     --percentile $p \
                #     --method bound-sketch \
                #     --pessimistic \
                #     --writefile $filename"_bound_pessimistic.csv" \
                #     --cuda \
                #     > $filename"_bound_pessimistic.log"
                # else
                #     echo "skipped bound-sketch pessimistic"
                # fi
            done
        done
    done
done