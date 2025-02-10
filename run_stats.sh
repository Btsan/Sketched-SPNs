for method in estimates/stats*.txt; do
    base=$(basename ${method%.*})
    echo $base

    if [ ! -f runtimes/$base.log ]; then
        echo "Processing "$method

        psql \
        -d stats \
        -U postgres \
        -c "SET ml_joinest_fname='"$method"';" \
        -f prefix.sql \
        -f stats_CEB.sql \
        &> "runtimes/"$base".log"
        
        sleep 11
    else
        echo "Skipping "$method" as runtimes/"$base".log already exists"
    fi
    echo "runtimes/"$base".log: "$(grep -Eo "Time" "runtimes/"$base".log" | wc -l)" query execution times"
    echo "Total execution time: "$(grep -Eo "[0-9]+(.[0-9]+) ms" "runtimes/"$base".log" | awk '{sum += $1} END {print sum / 1e3 /60 / 60" hrs"}')
done
# docker exec -it -w /var/lib/pgsql/13.1/data/ ce-benchmark bash