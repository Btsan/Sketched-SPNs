
clusters=(1 1e-1 1e-2) # 6 -> 3
widths=(1e6) # 4 -> 1
methods=(count bound) # 2 

for w in "${widths[@]}"
do
    for m in "${methods[@]}"
    do
        for c in "${clusters[@]}"
        do
            base="stats-ceb_5x"$w"_min"$c"_"$m
            echo $base

            psql \
            -d stats \
            -U postgres \
            -c "SET ml_joinest_fname='estimates/"$base".txt';" \
            -f prefix.sql \
            -f stats_CEB.sql \
            &> "runtimes/"$base".log"

            psql \
            -d stats \
            -U postgres \
            -c "SET ml_joinest_fname='estimates/"$base"_pessimistic.txt';" \
            -f prefix.sql \
            -f stats_CEB.sql \
            &> "runtimes/"$base"_pessimistic.log"
        done
        # psql \
        # -d stats \
        # -U postgres \
        # -c "SET ml_joinest_fname='estimates/"$base"_exact.txt';" \
        # -f prefix.sql \
        # -f stats_CEB.sql \
        # &> "runtimes/"$base"_exact.log"
    done
done

# docker exec -it -w /var/lib/pgsql/13.1/data/ ce-benchmark bash