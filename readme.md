<meta name="robots" content="noindex">

# Sketched Sum-Product Networks


## Requirements

Install python dependencies

```
pip3 install -r requirements.txt
```

Also, install the KWiseHash package from [https://github.com/mikeheddes/fast-multi-join-sketch](https://github.com/mikeheddes/fast-multi-join-sketch/tree/c66c1486679f6eea4f78d703550050a83153360b/kwisehash)

Run the following script to download the Stats and IMDb datasets used to train SPNs

```
bash datasets.sh
```

## Running Experiments

Use either of the provided evaluation scripts to run experiments on the Stats-CEB or JOB-light workload.

```
bash run_stats.sh
bash run_job-light.sh
```


## Usage

You can also provide custom arguments to the python script

```
python3 sketched_spn.py \
        --depth 5 \
        --width 1e4 \
        --min_cluster 0.25 \
        --corr_threshold 0.5 \
        --data ./End-to-End-CardEst-Benchmark-master/datasets/stats_simplified/ \
        --workload ./stats_CEB_sub_queries_corrected.sql \
        --experiment stats-ceb \
        --writefile "out.csv"
```

arguments:
```
usage: sketched_spn.py [-h] [--depth DEPTH] [--width [WIDTH ...]] [--workload WORKLOAD] [--data DATA] [--writefile WRITEFILE] [--k K] [--decompose DECOMPOSE]
                       [--min_cluster MIN_CLUSTER] [--cluster_first] [--primary [PRIMARY ...]] [--sparse] [--dates [DATES ...]] [--experiment {job-light,stats-ceb}]
                       [--find_keys] [--independence INDEPENDENCE] [--pessimistic]

run sketched sum-product networks on a workload

options:
  -h, --help            show this help message and exit
  --depth DEPTH         depth of sketches
  --width [WIDTH ...], --widths [WIDTH ...]
                        width(s) of sketche (widths should be evenly divisible by smaller widths, if multiple are specified)
  --workload WORKLOAD   CSV containing the format (subqueries || parent ID || cardinality)
  --data DATA           path to directory of table files (as CSVs)
  --writefile WRITEFILE
                        name of output csv file
  --k K                 each Sum Node partitions data into k**2 clusters
  --decompose DECOMPOSE, --corr_threshold DECOMPOSE
                        pairs of columns are decomposed with less correlation than this threshold (default 0.3)
  --min_cluster MIN_CLUSTER
                        stop partitioning when data (cluster) is smaller than min_cluster
  --cluster_first       make the root layer a Sum Node (cluster data first) e.g., for large tables with few columns
  --primary [PRIMARY ...]
                        name(s) of table attributes that are primary keys e.g., title.id
  --sparse              use sparse arrays for sketches, i.e., recommended if width >= 1e6
  --dates [DATES ...]   specify date columns (in table.col format)
  --experiment {job-light,stats-ceb}
  --find_keys           analyze columns in workload instead of running estimation e.g., to help prepare experimental setup
  --independence INDEPENDENCE
                        use k-wise independent hashing (recommended k=2**n for n-way joins)
  --pessimistic         use pessimistic (probabilistic upper bound) sketch approximation
```
