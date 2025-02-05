
clusters=(1 5e-1 2e-1 1e-1 5e-2 1e-2)
widths=(4096 1e4 1e5 1e6)
methods=(count bound)

for w in "${widths[@]}"
do
    for m in "${methods[@]}"
    do
        for c in "${clusters[@]}"
        do
            base="stats-ceb_5x"$w"_min"$c"_"$m

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
        psql \
        -d stats \
        -U postgres \
        -c "SET ml_joinest_fname='estimates/"$base"_exact.txt';" \
        -f prefix.sql \
        -f stats_CEB.sql \
        &> "runtimes/"$base"_exact.log"
    done
done

# docker exec -it -w /var/lib/pgsql/13.1/data/ ce-benchmark bash