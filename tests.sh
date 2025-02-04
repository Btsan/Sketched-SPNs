
mkdir -p tests/

python sketched_spn.py  --depth 5 \
                        --width 1e6 \
                        --pickle features/ \
                        --method count-sketch \
                        --exact \
                        --cuda &> tests/exact_count_sketch.log

python sketched_spn.py  --depth 5 \
                        --width 1e6 \
                        --pickle features/ \
                        --method bound-sketch \
                        --exact \
                        --cuda &> tests/exact_bound_sketch.log

python sketched_spn.py  --depth 5 \
                        --width 1e6 \
                        --pickle features/ \
                        --min_cluster 1e-2 \
                        --method count-sketch \
                        --pessimistic \
                        --cuda &> tests/count_sketch.log

python sketched_spn.py  --depth 5 \
                        --width 1e6 \
                        --pickle features/ \
                        --min_cluster 1e-2 \
                        --method bound-sketch \
                        --pessimistic \
                        --cuda &> tests/bound_sketch.log