
mkdir -p ./tests/

# python sketched_spn.py  --depth 5 \
#                         --width 1e6 \
#                         --pickle features/ \
#                         --method count-sketch \
#                         --exact \
#                         --cuda &> tests/exact_count_sketch.log

# python sketched_spn.py  --depth 5 \
#                         --width 1e6 \
#                         --pickle features/ \
#                         --method bound-sketch \
#                         --exact \
#                         --cuda &> tests/exact_bound_sketch.log

# python sketched_spn.py  --depth 5 \
#                         --width 1e6 \
#                         --pickle features/ \
#                         --min_cluster 1e-2 \
#                         --method count-sketch \
#                         --pessimistic \
#                         --cuda &> tests/count_sketch.log

# python sketched_spn.py  --depth 5 \
#                         --width 1e6 \
#                         --pickle features/ \
#                         --min_cluster 1e-2 \
#                         --method bound-sketch \
#                         --pessimistic \
#                         --cuda &> tests/bound_sketch.log

clusters=(1 5e-1 2e-1 1e-1 5e-2 1e-2)
widths=(4096 1e4 1e5 1e6)

for w in "${widths[@]}"
do
    for c in "${clusters[@]}"
    do
        echo "running Stats-CEB: min. cluster = "$c" sketch width = "$w
        echo .
        echo .
        echo .

        filename="./tests/stats-ceb_5x"$w"_min"$c

        # python3 sketched_spn.py \
        # --depth 5 \
        # --width $w \
        # --pickle ./features/ \
        # --independence 32 \
        # --min_cluster $c \
        # --method count-sketch \
        # --writefile $filename"_count.csv" \
        # --cuda \
        # > $filename"_count.log"

        # python3 sketched_spn.py \
        # --depth 5 \
        # --width $w \
        # --pickle ./features/ \
        # --independence 32 \
        # --min_cluster $c \
        # --method count-sketch \
        # --pessimistic \
        # --writefile $filename"_count_pessimistic.csv" \
        # --cuda \
        # > $filename"_count_pessimistic.log"
        
        # python3 sketched_spn.py \
        # --depth 5 \
        # --width $w \
        # --pickle ./features/ \
        # --independence 32 \
        # --min_cluster $c \
        # --method bound-sketch \
        # --writefile $filename"_bound.csv" \
        # --cuda \
        # > $filename"_bound.log"

        # python3 sketched_spn.py \
        # --depth 5 \
        # --width $w \
        # --pickle ./features/ \
        # --independence 32 \
        # --min_cluster $c \
        # --method bound-sketch \
        # --pessimistic \
        # --writefile $filename"_bound_pessimistic.csv" \
        # --cuda \
        # > $filename"_bound_pessimistic.log"

    done

    filename="./tests/stats-ceb_5x"$w
    
    python3 sketched_spn.py \
    --depth 5 \
    --width $w \
    --pickle ./features/ \
    --independence 32 \
    --method count-sketch \
    --exact \
    --writefile $filename"_count_exact.csv" \
    --cuda \
    > $filename"_count_exact.log"

    python3 sketched_spn.py \
    --depth 5 \
    --width $w \
    --pickle ./features/ \
    --independence 32 \
    --method bound-sketch \
    --exact \
    --writefile $filename"_bound_exact.csv" \
    --cuda \
    > $filename"_bound_exact.log"
    
done