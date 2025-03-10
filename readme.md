<meta name="robots" content="noindex">

## Dependencies

Install Python dependencies (listed in [requirements.txt](/requirements.txt))
```
pip3 install -r requirements.txt
```

Optionally, install the KWiseHash package from [https://github.com/mikeheddes/fast-multi-join-sketch](https://github.com/mikeheddes/fast-multi-join-sketch/tree/c66c1486679f6eea4f78d703550050a83153360b/kwisehash).

KWiseHash greatly reduces sketching time, especially when evaluating exact sketches. Otherwise, our Python implementation is used.

## Datasets

Run the following script to download the Stats and IMDb datasets used to train SPNs

```
bash datasets.sh
```

## Usage

Launch [sketched_spn.py](/sketched_spn.py) for both training and inference on a workload.

We can forgo the model and use exact sketches with `--exact_sketch` toggled on.
This may be useful to run first, just to test installed dependencies.

```bash
python sketched_spn.py \
--method count-sketch \
--width 1e6 \
--exact_sketch
```

For training, it is recommended to specify a directory with `--pickle` to save RDC features in, especially for training multiple configurations. 
```bash
python sketched_spn.py \
--pickle ./features/ \
--method count-sketch \
--width 1e5 \
--decompose 1e-2 \
--min_cluster 0.2 
```

If time and memory are concerns, a larger minimum cluster size threshold can be specified with `--min_cluster` to decrease model size, at the expense of accuracy.
Toggling the flag `--exact_preds` can also improve inference time, since the sketch selectivity estimator in leaf nodes is simply implemented on top of computing exact selections.

For example, the fastest model can be trained by specifying the total number of tuples in a relation to be the minimum cluster size via `--min_cluster 1` (100%), which creates the worst-case complete independence assumption model.

Instead of an accurate unbiased median estimator, specify `--percentile 1` to use the maximum count-sketch estimate.
This works well with `--pessimistic` toggled on, which enables the pessimistic sketch approximation in product nodes.
Alternatively, the upper-bounds estimator, Bound-Sketch, may be specified with `--method bound-sketch`. 

By default the program runs Stats-CEB. To specify another dataset, e.g., JOB-light, provide the `--workload`, `--data`, and `--experiment` arguments:
```bash
python sketched_spn.py \
--pickle ./features/ \
--method count-sketch \
--width 1e5 \
--decompose 1e-2 \
--min_cluster 0.1 \
--percentile 1 \
--pessimistic \
--workload ./workloads/job_light_sub_query_with_star_join.sql.txt \
--data ./End-to-End-CardEst-Benchmark-master/datasets/imdb/ \
--experiment job-light
```

New datasets require extending [experiments.py](/experiments.py) with their schemas, e.g., specifying join keys, column names and types.
