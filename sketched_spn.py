from collections import defaultdict
from time import perf_counter_ns
from pathlib import Path

import torch
from torch.fft import fft, ifft

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from SPN import SPN
from RDC import rdc_transform
from Sketches import AMS, CountSketch, BoundSketch

### KWiseHash package by Heddes et al. (SIGMOD 2024)
### https://github.com/mikeheddes/fast-multi-join-sketch - Jul 2024
# from kwisehash import KWiseHash
# class SignHash(object):
#     def __init__(self, depth, k=4) -> None:
#         self.depth = depth
#         self.fn = KWiseHash(depth, k=k)

#     def __call__(self, items: torch.Tensor) -> torch.Tensor:
#         return self.fn.sign(torch.as_tensor(items))
    
# class BinHash(object):
#     def __init__(self, depth, width, k=2) -> None:
#         self.depth = depth
#         self.width = width
#         self.fn = KWiseHash(depth, k=k)

#     def __call__(self, items: torch.Tensor) -> torch.Tensor:
#         # extra modulo to ensure no overflow
#         return self.fn.bin(torch.as_tensor(items), self.width) % self.width

### python implementation (approx 10x slower than KWiseHash)
from hashes import BinHash, SignHash 

def get_hashes(depth, width, k=4):
    binhashes = BinHash(depth, width, k=k)
    signhashes = SignHash(depth, k=k)
    return binhashes, signhashes

def exact_sketch(data, bin_hashes=None, sign_hashes=None, method='count-sketch'):
    depth = bin_hashes[0].depth
    width = bin_hashes[0].width

    if method == 'ams':
        sketch = AMS(data,
                     depth,
                     sign_hashes=sign_hashes,)
    elif method in ('bound-sketch', 'count-min'):
        sketch = BoundSketch(data,
                             depth,
                             width,
                             sign_hashes=sign_hashes,
                             bin_hashes=bin_hashes,)
    else:
        assert method == 'count-sketch'
        sketch = CountSketch(data,
                             depth,
                             width,
                             sign_hashes=sign_hashes,
                             bin_hashes=bin_hashes,)

    return sketch

def cross_correlate(node, query, id2sketch, visited=None):
    if visited is None:
        visited = set()
    id, key = node.split('.')
    visited.add(node)
    sketch = id2sketch[id]

    for other_node in query.joined_nodes(id):
        # skip current node
        if other_node == node:
            continue
        visited.add(other_node)
        tmp = 1
        for joined_node in query.joined_with(other_node):
            tmp = tmp * cross_correlate(joined_node, query, id2sketch, visited=visited)
        sketch = ifft(fft(tmp).conj() * fft(sketch)).real

    for joined_node in query.joined_with(node).difference(visited):
        sketch = sketch * cross_correlate(joined_node, query, id2sketch, visited=visited)
    return sketch

def estimate(query, models, cuda=False, method='count-sketch', percentile=0.5, exact_prob=False):
    inference_times = []
    sketching_times = []
    copy_times = []

    sketches_lo = dict()
    use_count = True
    for id, name in query.table_mapping_iter():
        predicates = dict() if id not in query.selects else query.selects[id]
        keys = query.id2joined_attrs[id]
        components = dict()
        for attr in keys:
            node = f"{id}.{attr}"
            components[attr] = query.node2component[node]
        print(f'{name} components', components)

        t0 = perf_counter_ns()
        if isinstance(models[name], SPN):
            # use iterative inference method of SPN
            output = models[name].iterative(predicates, keys, components=components, count=use_count, exact_prob=exact_prob, cuda=cuda)
        else:
            output = models[name](predicates, keys, components=components, count=use_count, exact_prob=exact_prob, cuda=cuda)
        if len(output) == 2:
            # exact sketches don't have a separate copy time
            estimator, sketch_time = output
            copy_time = 0
        else:
            estimator, sketch_time, copy_time = output
        t1 = perf_counter_ns()
        inference_times.append(t1 - t0 - sketch_time - copy_time)
        sketching_times.append(sketch_time)
        copy_times.append(copy_time)

        # double check before anythng else
        if isinstance(estimator, (int, float)):
            assert estimator == 0, estimator
            print(f"Sketch of {id}({keys.values()}) having {predicates} is 0")
            return (0,
                    pd.Timedelta(sum(inference_times), unit='ns'),
                    pd.Timedelta(sum(sketching_times), unit='ns'),
                    pd.Timedelta(sum(copy_times), unit='ns'),
                    pd.Timedelta(0))
        
        if method == 'bound-sketch':
            print(f"use_count={use_count} returns {type(estimator)}")
            assert (estimator.sketch_lo >= 0).all(), f"negative values in {name} lo sketch"
            use_count = False
        
        # print(estimator)
        sketches_lo[id] = estimator.sketch_lo if not cuda else estimator.sketch_lo.cuda()

    # sum for total sequential inference time
    # max for longest parallel inference time of all models
    max_inference = pd.Timedelta(sum(inference_times), unit='ns')
    total_sketching = pd.Timedelta(sum(sketching_times), unit='ns')
    total_copying = pd.Timedelta(sum(copy_times), unit='ns')
    use_bifocal = estimator.is_bifocal

    t0 = perf_counter_ns()

    start_node = query.random_node()

    # lo-freq estimate
    sketch_estimates = cross_correlate(start_node, query, sketches_lo).sum(dim=1)
    print(f'lo sketch estimates {sketch_estimates.shape}:')
    print(sketch_estimates)

    if method in ('count-min', 'bound-sketch'):
        est = sketch_estimates.min().item()
    else:
        est = sketch_estimates.quantile(percentile).item() # negative estimates are allowed

    t1 = perf_counter_ns()
    estimation_time = pd.Timedelta(t1-t0, unit='ns')
    
    return est, max_inference, total_sketching, total_copying, estimation_time

if __name__ == '__main__':
    import argparse

    from query  import Query, extract_graph
    from dataset import get_dataframe, get_workload
    import experiments
    
    parser = argparse.ArgumentParser(description='run sketched sum-product networks on a workload')
    parser.add_argument('--method', default='count-sketch', choices=['count-sketch', 'count-min', 'bound-sketch'], type=str.lower, help='depth of sketches')
    parser.add_argument('--depth', default=5, type=lambda x: int(float(x)), help='depth of sketches')
    parser.add_argument('--width', default=100000, type=lambda x: int(float(x)), help='width of sketches')
    parser.add_argument('--workload', default=Path('./workloads/stats_CEB_sub_queries_corrected.sql'), type=Path, help='CSV containing the format (subqueries || parent ID || cardinality)')
    parser.add_argument('--data', default=Path('./End-to-End-CardEst-Benchmark-master/datasets/stats_simplified/'), type=Path, help='path to directory containing table CSVs')
    parser.add_argument('--writefile', default=Path('out.csv'), type=Path, help='name of output csv file')
    parser.add_argument('--k', default=2, type=int, help='each Sum Node partitions data into k>=2 clusters')
    parser.add_argument('--decompose', '--rdc_threshold', default=0.01, type=float, help='group columns with pairwise RDC above this threshold')
    parser.add_argument('--min_cluster', default=0.2, type=float, help='minimum clustering size for sum nodes, i.e., treated as a percentage if less than 1')
    parser.add_argument('--cluster_first', action='store_true', help='force the root layer to be a Sum Node (cluster first)')
    parser.add_argument('--experiment', default='stats-ceb', choices=['job-light', 'stats-ceb'])
    parser.add_argument('--find_keys', action='store_true', help='analyze columns in workload instead of running estimation e.g., to prepare experimental config')
    parser.add_argument('--independence', default=4, type=int, help='independence of k-universal hashing for sketches')
    parser.add_argument('--pessimistic', action='store_true', help='use pessimistic approximation (use with --percentile 1 for max estimator)')
    parser.add_argument('--pickle', default=None, type=Path, help='path to directory to save featurized data for faster subsequent runs')
    # parser.add_argument('--bifocal', default=0, type=int, help='number of heavy hitters to track per leaf node for bifocal estimation')
    parser.add_argument('--cuda', action='store_true', help='use GPU for estimation (not recommended unless estimation time is high)')
    parser.add_argument('--exact_sketch', action='store_true', help='use exact sketches for estimation')
    parser.add_argument('--percentile', default=0.5, type=float, help='percentile of [depth] estimates used as final estimate, e.g., 0.5 for median (default) and 1 for max')
    parser.add_argument('--kmeans', action='store_true', help='use kmeans to learn sum nodes (fast, increases model size)')
    parser.add_argument('--exact_preds', action='store_true', help='use exact selectivity of predicates in leaf nodes, instead of sketch estimates')
    args = parser.parse_args()
    print(args)

    if args.find_keys:
        workload = get_workload(args.workload)
        if not args.workload:
            raise SystemExit('Missing --workload argument. Must be specified with --find_keys')
        keys = defaultdict(set)
        preds = defaultdict(set)
        for i, row in enumerate(workload.iloc()):
            query = row['query']
            nodes, edges = extract_graph(query)
            for node in nodes:
                keys[node.name].update(node.keys)
                preds[node.name].update(node.predicates.keys())
        for t, k in keys.items():
            print(f'join keys of {t}: {k}')
        for t, c in preds.items():
            print(f'predicate columns of {t}: {c}')
        exit(0)

    primary, dates, tables = experiments.get_config(args.experiment)

    num_components = len(tables) - 1 # this suffices for acyclic joins
    bin_hashes = [BinHash(args.depth, args.width) for _ in range(num_components)]
    sign_hashes = [SignHash(args.depth, k=args.independence) for _ in range(num_components)]

    workload = get_workload(args.workload)
    workload[f"{args.method}_{args.depth}x{args.width}"] = -1.0
    workload[f"{args.method}_{args.depth}x{args.width}_err"] = -1.0

    workload['num_tables'] = 0
    workload['join_components'] = 0
    workload['inference_time'] = pd.Timedelta(0.0, unit='sec')
    workload['sketching_time'] = pd.Timedelta(0.0, unit='sec')
    workload['copy_overhead'] = pd.Timedelta(0.0, unit='sec')
    workload['estimation_time'] = pd.Timedelta(0.0, unit='sec')
    workload['total_time'] = pd.Timedelta(0.0, unit='sec')
    workload = workload.copy()
    print(f'Generating results into workload ({workload.shape})')

    with torch.inference_mode():
        models = dict()

        training_times = []
        for table, meta in tables.items():
            ts = perf_counter_ns()
            dataset = get_dataframe(f'{args.data}/{table}.csv', names=meta['names'], columns=meta['col_types'].keys())
            if table in dates:
                for col in dates[table]:
                    dataset[col] = pd.to_datetime(dataset[col])
            delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
            print(f"Loaded {table} ({dataset.memory_usage(deep=True).sum():,} bytes) {delta.total_seconds():>25,.2f}s ({delta})")
            print(dataset.describe().to_string(float_format="{:,.2f}".format))
            print(dataset.memory_usage(deep=True).to_string(float_format="{:,.2f}".format))

            if args.exact_sketch:
                models[table] = exact_sketch(dataset, bin_hashes=bin_hashes, sign_hashes=sign_hashes, method=args.method)
            else:
                ts = perf_counter_ns()
                # extract features before training
                if args.pickle:
                    save_path = args.pickle / f"{table}.pkl"
                    if save_path.exists():
                        rdc_features = pd.read_pickle(save_path)
                        delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
                        print(f"Loaded pickled features from {save_path} ({delta})")
                        assert len(rdc_features) == len(dataset), f"Features ({save_path}) do not match ({args.data/table}.csv)"
                    else:
                        print(f"Extracting features from {table} ...", flush=True)
                        args.pickle.mkdir(parents=True, exist_ok=True)
                        rdc_features = rdc_transform(dataset, meta['col_types'])
                        delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
                        print(f"Extracted features from {table} ({delta})", flush=True)
                        save_path = f"{args.pickle}/{table}.pkl"
                        rdc_features.to_pickle(save_path)
                else:
                    print(f"Extracting features from {table} ...", flush=True)
                    rdc_features = rdc_transform(dataset, meta['col_types'])
                    delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
                    print(f"Extracted features from {table} ({delta})", flush=True)
                
                # minimum size of clusters in sum nodes
                min_cluster = args.min_cluster if args.min_cluster > 1 else abs(args.min_cluster * len(dataset))

                # train on features
                ts = perf_counter_ns()
                models[table] = SPN(dataset, rdc_features, 
                                    bin_hashes=bin_hashes, 
                                    sign_hashes=sign_hashes, 
                                    corr_threshold=args.decompose, 
                                    min_cluster=min_cluster, 
                                    num_clusters=args.k, 
                                    cluster_next=args.cluster_first,
                                    keys=meta['keys'], 
                                    method=args.method, 
                                    pessimistic=args.pessimistic, 
                                    use_kmeans=args.kmeans,)
            delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
            print(f"{'Hashed data' if args.exact_sketch else 'Trained SPN'} ({models[table].memory / 2**20:,.2f} MiB) on {table} ({delta})", flush=True)
            training_times.append(delta)
        total_training = sum(training_times, pd.Timedelta(0))

        for i, row in enumerate(workload.iloc()):
            query_start = perf_counter_ns()
            sql = row['query']
            query = Query(sql)
            nodes, edges = extract_graph(sql)
            num_components = 1 + sum(len(n.keys)-1 for n in nodes)
            # if num_components == 1: continue

            print(f"{i}: {query} ({row['cardinality']:,})")
            
            est, inference_time, sketching_time, copying_time, estimation_time = estimate(query, models, cuda=args.cuda, method=args.method, percentile=args.percentile, exact_prob=args.exact_preds)
            name = f"{args.method}_{args.depth}x{args.width}"
            workload.loc[i, name] = est
            workload.loc[i, name + '_err'] = max(est, 1) / max(row['cardinality'], 1) if est >= row['cardinality'] else max(row['cardinality'], 1) / max(est, 1)
            # max(max(est, 1), row['cardinality']) / max(min(abs(est), row['cardinality']), 1)
            query_time = pd.Timedelta(perf_counter_ns() - query_start, unit='ns')

            workload.loc[i, 'num_tables'] = len(nodes)
            workload.loc[i, 'join_components'] = num_components
            workload.loc[i, 'inference_time'] = inference_time
            workload.loc[i, 'sketching_time'] = sketching_time
            workload.loc[i, 'copy_overhead'] = copying_time
            workload.loc[i, 'estimation_time'] = estimation_time
            workload.loc[i, 'total_time'] = query_time

            print(workload.loc[i].to_string(float_format="{:,.2f}".format))
            print(f'Query {i} finished in {query_time.total_seconds():>25,.2f}s ({query_time})')
            print(flush=True)

        cols = list(workload.columns)
        for x in ('query', 'parent'):
            cols.remove(x)

        workload = workload[workload['num_tables'] > 0]

        drop = ['std', 'min']
        pctl = [0.25, 0.5, 0.75, 0.9]

        workload.to_csv(args.writefile, index=False)
        print(workload[cols].describe(percentiles=pctl).transpose().drop(columns=drop).to_string(float_format="{:,.2f}".format))

        print('\nEquality only:')
        print(workload.query("not `query`.str.contains('>') and not `query`.str.contains('<')", engine='python')[cols].describe(percentiles=pctl).transpose().drop(columns=drop).to_string(float_format="{:,.2f}".format))

        print('\nRange included:')
        print(workload.query("`query`.str.contains('>') or `query`.str.contains('<')", engine='python')[cols].describe(percentiles=pctl).transpose().drop(columns=drop).to_string(float_format="{:,.2f}".format))

        print('\nTransitive Joins only:')
        print(workload.query("`join_components` == 1", engine='python')[cols].describe(percentiles=pctl).transpose().drop(columns=drop).to_string(float_format="{:,.2f}".format))

        print('\nNon-Transitive Joins only:')
        print(workload.query("`join_components` > 1", engine='python')[cols].describe(percentiles=pctl).transpose().drop(columns=drop).to_string(float_format="{:,.2f}".format))

        # todo: iterate over distincts instead of the range
        for i in range(workload['num_tables'].min(), workload['num_tables'].max() + 1):
            print(f'\n{i}-way Joins:')
            print(workload.query(f"`num_tables` == {i}", engine='python')[cols].describe(percentiles=pctl).transpose().drop(columns=drop).to_string(float_format="{:,.2f}".format))

        print(f"\nTotal Sketching Time: {workload['sketching_time'].sum()} (average {workload['sketching_time'].mean()})")
        if not args.exact_sketch:
            print(f"Total Structure Learning Time: {total_training} (average {total_training / len(models)})")
            print(f"Total Model Inference Time: {workload['inference_time'].sum()} (average {workload['inference_time'].mean()})")
            print(f"Total Model Copying Overhead: {workload['copy_overhead'].sum()} (average {workload['copy_overhead'].mean()})")
        print(f"Total Estimation Time: {workload['estimation_time'].sum()} (average {workload['copy_overhead'].mean()})")
        print(f"Total Workload Time: {workload['total_time'].sum()} (average {workload['total_time'].mean()})")

        # compute memory usage due to sketches after running workload
        model_mem_usage = sum([model.memory_usage() for model in models.values()])
        gb = model_mem_usage // 2**30
        mb = (model_mem_usage % 2**30) // 2**20
        kb = (model_mem_usage % 2**20) // 2**10
        b = model_mem_usage % 2**10
        print(f"Total memory usage: {gb:,} GiB  {mb:,} MiB  {kb:,} KiB  {b:,} B  (Total {model_mem_usage:,} bytes)")

        print(f"End results for {args}")