
csv_dir=./tests/
txt_dir=./estimates/

mkdir -p $txt_dir

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

            python extract_estimates.py \
                --csv $csv_dir/$base".csv" \
                --out $txt_dir/$base".txt"

            python extract_estimates.py \
                --csv $csv_dir/$base"_pessimistic.csv" \
                --out $txt_dir/$base"_pessimistic.txt"
        done

        base="stats-ceb_5x"$w"_"$m"_exact"

        python extract_estimates.py \
            --csv $csv_dir/$base".csv" \
            --out $txt_dir/$base".txt"
    done
done