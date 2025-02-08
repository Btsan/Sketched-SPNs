from copy import deepcopy
from itertools import combinations
from collections import defaultdict
from time import perf_counter_ns
from pathlib import Path

import numpy as np
import torch
from torch.fft import fft, ifft

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from SPN import SPN
from RDC import rdc_transform
from Sketches import AMS, CountSketch, BoundSketch

## KWiseHash package by Heddes et al. (SIGMOD 2024)
## https://github.com/mikeheddes/fast-multi-join-sketch - Jul 2024
from kwisehash import KWiseHash
class SignHash(object):
    def __init__(self, depth, k=4) -> None:
        self.fn = KWiseHash(depth, k=k)

    def __call__(self, items: torch.Tensor) -> torch.Tensor:
        return self.fn.sign(torch.as_tensor(items))
    
class BinHash(object):
    def __init__(self, depth, width, k=2) -> None:
        self.num_bins = width
        self.fn = KWiseHash(depth, k=k)

    def __call__(self, items: torch.Tensor) -> torch.Tensor:
        return self.fn.bin(torch.as_tensor(items), self.num_bins)
    
# from hashes import SignHash, BinHash ### uncomment for python alternative (~10x slower)

def get_hashes(depth, width, k=4):
    binhashes = BinHash(depth, width, k=k)
    signhashes = SignHash(depth, k=k)
    return binhashes, signhashes

def exact_sketch(data, bin_hashes=None, sign_hashes=None, bifocal=0, method='ams', convolutional=False):
    depth = bin_hashes[0].fn.seeds.shape[0]
    width = bin_hashes[0].num_bins

    if method == 'ams':
        sketch = AMS(data,
                     depth,
                     sign_hashes=sign_hashes,
                     bifocal=bifocal)
    elif method in ('bound-sketch', 'count-min'):
        sketch = BoundSketch(data,
                             depth,
                             width,
                             sign_hashes=sign_hashes,
                             bin_hashes=bin_hashes,
                             bifocal=bifocal,
                             convolutional=convolutional)
    else:
        sketch = CountSketch(data,
                             depth,
                             width,
                             sign_hashes=sign_hashes,
                             bin_hashes=bin_hashes,
                             bifocal=bifocal,
                             convolutional=convolutional)

    return sketch

def compass_estimate(query, id2sketch):
    id2einsum_indices = defaultdict(lambda: [...])

    for idx, join in enumerate(query.joins):
        left, _, right = join
        id = left.split('.')[0]
        id2einsum_indices[id].append(idx)

        id = right.split('.')[0]
        id2einsum_indices[id].append(idx)

    einsum_args = []
    for id in id2sketch:
        einsum_args.append(id2sketch[id])
        einsum_args.append(id2einsum_indices[id])

    einsum_args.append([...])
    return torch.einsum(*einsum_args)

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

def hadamard(id2sketch):
    estimates = 1
    for sketch in id2sketch.values():
        estimates = estimates * sketch

    return estimates

def multiply_frequencies(node, query, id2series, visited=None, root=True):
    """
    This implementation could be improved to better handle primary keys, but suffices for now.
    E.g., the heavy hitters of primary keys could be defined as the heavy hitters of corresponding foreign keys
    """
    if visited is None:
        visited = set()
    id, key = node.split('.')
    visited.add(node)

    freq = id2series[id].reset_index(name=f"_count_{id}")

    for other_node in query.joined_nodes(id):
        if other_node == node:
            continue

        other_id, other_key = other_node.split('.')
        assert id == other_id
        visited.add(other_node)

        for joined_node in query.joined_with(other_node):
            joined_id, joined_key = joined_node.split('.')
            assert other_id != joined_id
            
            tmp = multiply_frequencies(joined_node, query, id2series, visited, root=False)

            assert other_key in freq.columns, f"{other_node} missing in {freq.columns}"
            assert joined_key in tmp.columns, f'{joined_node} missing in {tmp.columns}'
            freq = pd.merge(freq, tmp, left_on=other_key, right_on=joined_key, suffixes=(None, f"_{joined_id}")).dropna(subset=other_key)

    for joined_node in query.joined_with(node).difference(visited):
        tmp = multiply_frequencies(joined_node, query, id2series, visited, root=False)
        joined_id, joined_key = joined_node.split('.')
        assert key in freq.columns, f"{node} missing in {freq.columns}"
        assert joined_key in tmp.columns, f'{joined_node} missing in {tmp.columns}'

        freq = pd.merge(freq, tmp, left_on=key, right_on=joined_key, suffixes=(None, joined_id)).dropna(subset=key)

    if root:
        freq['_count'] = 1.0
        for id in id2series:
            freq['_count'] *= freq[f"_count_{id}"]
    return freq

def estimate(query, models, primary=None, cuda=False, method='count-sketch'):
    inference_times = []
    sketching_times = []

    exact_hi = dict()
    sketches_hi = dict()
    sketches_lo = dict()
    use_count = True
    for id, name in query.table_mapping_iter():
        predicates = query.selects[id]
        keys = query.id2joined_attrs[id]
        components = dict()
        for attr in keys:
            node = f"{id}.{attr}"
            components[attr] = query.node2component[node]
        print(f'{name} components', components)

        t0 = perf_counter_ns()
        estimator, sketch_time = models[name](predicates, keys, components=components, count=use_count)
        t1 = perf_counter_ns()
        inference_times.append(t1 - t0 - sketch_time)
        sketching_times.append(sketch_time)

        if method == 'bound-sketch':
            print(f"use_count={use_count} returns {type(estimator)}")
            assert (estimator.sketch_lo >= 0).all(), f"negative values in {name} lo sketch"
            use_count = False
        
        # print(estimator)
        sketches_lo[id] = estimator.sketch_lo.cuda() if cuda else estimator.sketch_lo

        if estimator.is_bifocal:
            if cuda:
                sketches_hi[id] = estimator.sketch_hi.cuda() if cuda else estimator.sketch_hi
        exact_hi[id] = estimator.exact_hi

    total_inference = pd.Timedelta(sum(inference_times), unit='ns')
    total_sketching = pd.Timedelta(sum(sketching_times), unit='ns')
    use_bifocal = estimator.is_bifocal
    depth, width = estimator.shape[:2]

    t0 = perf_counter_ns()

    start_node = query.random_node()

    # lo-freq estimate
    if method in ('ams'):
        sketch_estimates = hadamard(sketches_lo).sum(dim=1)
    else:
        sketch_estimates = cross_correlate(start_node, query, sketches_lo).sum(dim=1)
    print(f'lo sketch estimates {sketch_estimates.shape}:')
    print(sketch_estimates)

    if method in ('count-min', 'bound-sketch'):
        join_lo = sketch_estimates.min().item()
    else:
        join_lo = max(0, sketch_estimates.quantile(0.5).item())
    print(f"lo estimate = {join_lo:,.2f}")

    join_hi = 0
    join_hilo = 0
    if use_bifocal:
        # hi-freq estimate
        join_freq = multiply_frequencies(start_node, query, exact_hi)
        join_hi = join_freq['_count'].sum()
        print(f"hi estimates: {join_hi:,.2f}")

        # hi-lo freq estimate
        join_hilo = 0
        for num_swaps in range(1, len(sketches_lo)):
            for combo in combinations(sketches_lo.keys(), num_swaps):

                # copy lo-freq sketches and replace some with hi-freq sketches
                sketches_hilo = sketches_lo.copy()
                # sketches_hilo = deepcopy(sketches_lo)
                for id in combo:
                    sketches_hilo[id] = sketches_hi[id]
                    assert torch.count_nonzero(sketches_hilo[id]) == 0 or (sketches_hilo[id] != sketches_lo[id]).any(), "uh oh time to import deepccopy"

                if method in ('ams'):
                    sketch_estimates = hadamard(sketches_hilo)
                else:
                    sketch_estimates = cross_correlate(start_node, query, sketches_hilo)
                    
                if method in ('count-min', 'bound-sketch'):
                    join_hilo += sketch_estimates.sum(dim=1).min().item()
                else:
                    join_hilo += max(0, sketch_estimates.sum(dim=1).quantile(0.5).item())
        print(f"hi-lo estimates = {join_hilo:,.2f}")

    est = join_lo + join_hilo + join_hi

    t1 = perf_counter_ns()
    estimation_time = pd.Timedelta(t1-t0, unit='ns')
    
    return est, total_inference, total_sketching, estimation_time

def compare_sketches(exact_sketches, approximate_sketches):
    assert set(node.name for node in exact_sketches) == set(node.name for node in approximate_sketches), f'{set(node.name for node in exact_sketches)} not match {set(node.name for node in approximate_sketches)}'
    aggregates = {}
    for node_1, node_2 in zip(exact_sketches, approximate_sketches):
        assert node_1.name == node_2.name, f'table nodes not aligned: {node_1.name} != {node_2.name}'
        table = node_1.name
        exact_sketch = node_1.sketch.to_dense()
        approx_sketch = node_2.sketch.to_dense()
        aggregates[table] = torch.nn.functional.cosine_similarity(exact_sketch, approx_sketch).mean().item()
    return aggregates

if __name__ == '__main__':
    import argparse

    from query  import Query, extract_graph
    from dataset import get_dataframe, get_workload
    import experiments
    
    parser = argparse.ArgumentParser(description='run sketched sum-product networks on a workload')
    parser.add_argument('--method', default='count-sketch', choices=['ams', 'count-sketch', 'count-min', 'bound-sketch'], type=str.lower, help='depth of sketches')
    parser.add_argument('--depth', default=5, type=lambda x: int(float(x)), help='depth of sketches')
    parser.add_argument('--width', default=1024, type=lambda x: int(float(x)), help='width(s) of sketche (widths should be evenly divisible by smaller widths, if multiple are specified)')
    parser.add_argument('--workload', default=Path('./stats_CEB_sub_queries_corrected.sql'), type=Path, help='CSV containing the format (subqueries || parent ID || cardinality)')
    parser.add_argument('--data', default=Path('/ssd/btsan/stats_simplified/'), type=Path, help='path to directory of table files (as CSVs)')
    parser.add_argument('--writefile', default=Path('out.csv'), type=Path, help='name of output csv file')
    parser.add_argument('--k', default=1, type=int, help='each Sum Node partitions data into k**2 clusters')
    parser.add_argument('--decompose', '--rdc_threshold', default=0.3, type=float, help='column pairs with rdc above this threshold are grouped')
    parser.add_argument('--min_cluster', default=1e-2, type=float, help='stop partitioning when data (cluster) is smaller than min_cluster')
    parser.add_argument('--cluster_first', action='store_true', help='make the root layer a Sum Node (cluster data first) e.g., for large tables with few columns')
    parser.add_argument('--primary', 
                        default=['badges.Id',
                                'posts.Id',
                                'postLinks.Id',
                                'postHistory.Id',
                                'comments.Id',
                                'tags.Id',
                                'users.Id',
                                'votes.Id']
                        , nargs='*', type=str, help='name(s) of table attributes that are primary keys e.g., title.id')
    parser.add_argument('--sparse', action='store_true', help='use sparse arrays for sketches, i.e., recommended if width >= 1e6')
    parser.add_argument('--dates', nargs='*', 
                        default=['badges.Date', 
                                'comments.CreationDate', 
                                'postHistory.CreationDate', 
                                'postLinks.CreationDate', 
                                'posts.CreationDate', 
                                'users.CreationDate', 
                                'votes.CreationDate'], 
                        help='specify date columns (in table.col format)')
    parser.add_argument('--experiment', default='stats-ceb', choices=['job-light', 'stats-ceb'])
    parser.add_argument('--find_keys', action='store_true', help='analyze columns in workload instead of running estimation e.g., to help prepare experimental setup')
    parser.add_argument('--independence', default=4, type=int, help='use k-wise independent hashing (recommended k=2**n for n-way joins)')
    parser.add_argument('--pessimistic', action='store_true', help='use pessimistic (probabilistic upper bound) sketch approximation')
    parser.add_argument('--pickle', default=None, type=Path, help='path to directory to save featurized data for faster subsequent runs')
    parser.add_argument('--bifocal', default=0, type=int, help='number of heavy hitters to track per leaf node for bifocal estimation')
    parser.add_argument('--cuda', action='store_true', help='use GPU for estimation')
    parser.add_argument('--exact', action='store_true', help='use exact sketches (of pushdown selections) for estimation')
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

    if args.method == 'ams':
        args.width = 1
    
    # if args.method in ('bound-sketch', 'count-min'):
    #     args.pessimistic = True

    num_components = len(tables) - 1 # this suffices for acyclic joins
    bin_hashes = [BinHash(args.depth, args.width) for _ in range(num_components)]
    sign_hashes = [SignHash(args.depth, k=args.independence) for _ in range(num_components)]

    workload = get_workload(args.workload)
    workload[f"{args.method}_{args.depth}x{args.width}{'_primary' * bool(primary)}"] = -1.0
    workload[f"{args.method}_{args.depth}x{args.width}{'_primary' * bool(primary)}_err"] = -1.0

    workload['num_tables'] = 0
    workload['join_attributes'] = 0
    # workload['similarity'] = 0
    # workload['exact_time'] = pd.Timedelta(0.0, unit='sec')
    workload['inference_time'] = pd.Timedelta(0.0, unit='sec')
    workload['sketching_time'] = pd.Timedelta(0.0, unit='sec')
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
            print(f"Loaded {table} ({dataset.memory_usage(deep=True).sum():,} bytes) {delta.total_seconds():>25,.2f}s ({delta})", flush=True)
            print(dataset.describe().to_string(float_format="{:,.2f}".format))
            print(dataset.memory_usage(deep=True).to_string(float_format="{:,.2f}".format))

            min_cluster = args.min_cluster if args.min_cluster > 1 else abs(args.min_cluster * len(dataset))

            # extract features before training
            ts = perf_counter_ns()
            if args.pickle:
                save_path = args.pickle / f"{table}.pkl"
                if save_path.exists():
                    rdc_features = pd.read_pickle(save_path)
                    delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
                    print(f"Loaded pickled features from {save_path} ({delta})")
                    assert len(rdc_features) == len(dataset), f"Features ({save_path}) do not match ({args.data/table}.csv)"
                else:
                    args.pickle.mkdir(parents=True, exist_ok=True)
                    rdc_features = rdc_transform(dataset, meta['col_types'])
                    delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
                    print(f"Extracted features from {table} ({delta})")
                    save_path = f"{args.pickle}/{table}.pkl"
                    rdc_features.to_pickle(save_path)
            else:
                rdc_features = rdc_transform(dataset, meta['col_types'])
                delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
                print(f"Extracted features from {table} ({delta})")

            ts = perf_counter_ns()
            if args.exact:
                models[table] = exact_sketch(dataset, bin_hashes=bin_hashes, sign_hashes=sign_hashes, bifocal=args.bifocal, method=args.method)
            else:
                models[table] = SPN(dataset, rdc_features, bin_hashes=bin_hashes, sign_hashes=sign_hashes, corr_threshold=args.decompose, min_cluster=min_cluster, cluster_nbits=args.k, cluster_next=args.cluster_first, sparse=args.sparse, keys=meta['keys'], method=args.method, bifocal=args.bifocal, pessimistic=args.pessimistic)
            delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
            print(f"{'Hashed data' if args.exact else 'Trained SPN'} ({models[table].memory / 2**20:,.2f} MB) on {table} ({delta})", flush=True)
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
            
            # exact_estimates, sketches_1, exact_time = exact(exact_sketches, nodes, edges, args.width, primary=primary, same_sign=(num_components == 1)) 
            # for k, est in exact_estimates.items():
            #     workload.loc[i, k] = est
            #     workload.loc[i, k + '_err'] = max(est, row['cardinality']) / max(min(est, row['cardinality']), 1)
            est, inference_time, sketching_time, estimation_time = estimate(query, models, primary=primary, cuda=args.cuda, method=args.method)
            name = f"{args.method}_{args.depth}x{args.width}{'_primary' * bool(primary)}"
            workload.loc[i, name] = est
            workload.loc[i, name + '_err'] = max(est, row['cardinality']) / max(min(est, row['cardinality']), 1)
            query_time = pd.Timedelta(perf_counter_ns() - query_start, unit='ns')
            # similarities = compare_sketches(sketches_1, sketches_2)
            # for table, sim in similarities.items():
            #     print(f'sketch of table {table} has approximation similarity {sim}')

            # workload.loc[i, 'similarity'] = sum(similarities.values()) / len(similarities)
            workload.loc[i, 'num_tables'] = len(nodes)
            workload.loc[i, 'join_attributes'] = num_components
            workload.loc[i, 'inference_time'] = inference_time
            workload.loc[i, 'sketching_time'] = sketching_time
            workload.loc[i, 'estimation_time'] = estimation_time
            workload.loc[i, 'total_time'] = query_time

            print(workload.loc[i].to_string(float_format="{:,.2f}".format))
            print(f'Query {i} finished in {query_time.total_seconds():>25,.2f}s ({query_time})')
            print(flush=True)

        cols = list(workload.columns)
        for x in ('query', 'parent'):
            cols.remove(x)

        workload = workload[workload['num_tables'] > 0]

        drop = ['min']
        pctl = [0.25, 0.5, 0.75, 0.9]

        workload.to_csv(args.writefile, index=False)
        print(workload[cols].describe(percentiles=pctl).transpose().drop(columns=drop).to_string(float_format="{:,.2f}".format))

        print('\nEquality predicates only:')
        print(workload.query("not `query`.str.contains('>') and not `query`.str.contains('<')", engine='python')[cols].describe(percentiles=pctl).transpose().drop(columns=drop).to_string(float_format="{:,.2f}".format))

        print('\nRange predicates only:')
        print(workload.query("`query`.str.contains('>') or `query`.str.contains('<')", engine='python')[cols].describe(percentiles=pctl).transpose().drop(columns=drop).to_string(float_format="{:,.2f}".format))

        print('\nSingle-attribute Joins only:')
        print(workload.query("`join_attributes` == 1", engine='python')[cols].describe(percentiles=pctl).transpose().drop(columns=drop).to_string(float_format="{:,.2f}".format))

        print('\nMulti-attribute Joins only:')
        print(workload.query("`join_attributes` > 1", engine='python')[cols].describe(percentiles=pctl).transpose().drop(columns=drop).to_string(float_format="{:,.2f}".format))

        # todo: iterate over distincts instead of the range
        for i in range(workload['num_tables'].min(), workload['num_tables'].max() + 1):
            print(f'\n{i}-way Joins:')
            print(workload.query(f"`num_tables` == {i}", engine='python')[cols].describe(percentiles=pctl).transpose().drop(columns=drop).to_string(float_format="{:,.2f}".format))

        print(f"\nTotal Model Training Time: {total_training} (average {total_training / len(models)})")
        print(f"Total Sketching Time: {workload['sketching_time'].sum()} (average {workload['sketching_time'].mean()})")
        print(f"Total Model Inference Time: {workload['inference_time'].sum()} (average {workload['inference_time'].mean()})")
        print(f"Total Estimation Time: {workload['estimation_time'].sum()} (average {workload['estimation_time'].mean()})")
        print(f"Total Workload Time: {workload['total_time'].sum()} (average {workload['total_time'].mean()})")

        # compute memory usage due to sketches after running workload
        model_mem_usage = sum([model.memory_usage() for model in models.values()])

        print(f"Total size model: {model_mem_usage:,.2f} Bytes = {model_mem_usage / 2**10:,.2f} KB = {model_mem_usage / 2**20:,.2f} = {model_mem_usage / 2**30:,.2f}")

        gb = model_mem_usage // 2**30
        mb = (model_mem_usage % 2**30) // 2**20
        kb = (model_mem_usage % 2**20) // 2**10
        b = model_mem_usage % 2**10
        print(f"Total size model: {gb:,} GB, {mb:,} MB, {kb:,} KB, {b:,} B ({model_mem_usage:,} Bytes)")