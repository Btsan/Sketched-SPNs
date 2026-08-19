from collections import defaultdict
from time import perf_counter_ns
from pathlib import Path

import torch
from torch.fft import fft, ifft

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import RDC
from SPN import SPN
from Sketches import AMS, FastAGMS, BoundSketch
from clustering import kwisehash_transform

### KWiseHash package by Heddes et al. (SIGMOD 2024)
### https://github.com/mikeheddes/fast-multi-join-sketch - Jul 2024
from kwisehash import KWiseHash
from predicate_to_string import clear_canonical_cache
class SignHash(object):
    def __init__(self, depth, k=4) -> None:
        self.depth = depth
        self.fn = KWiseHash(depth, k=k)
    def __call__(self, arr: torch.Tensor) -> torch.Tensor:
        signs = self.fn.sign(torch.as_tensor(arr.flatten()))
        signs = signs.reshape(self.depth, *arr.shape)
        return signs
    
class BinHash(object):
    def __init__(self, depth, width, k=2) -> None:
        self.depth = depth
        self.width = width
        self.fn = KWiseHash(depth, k=k)
    def __call__(self, arr: torch.Tensor) -> torch.Tensor:
        # extra modulo to ensure no overflow
        bins = self.fn.bin(torch.as_tensor(arr.flatten()), self.width) % self.width
        bins = bins.reshape(self.depth, *arr.shape)
        return bins

### python implementation (at least 2x slower than KWiseHash)
# from hashes import BinHash, SignHash 

def get_hashes(depth, width, k=4):
    binhashes = BinHash(depth, width, k=k)
    signhashes = SignHash(depth, k=k)
    return binhashes, signhashes

def exact_sketch(data, bin_hashes=None, sign_hashes=None, method='count-sketch', sparse=False, sample_sketch=None):
    depth = bin_hashes[0].depth
    width = bin_hashes[0].width

    # if method == 'ams':
    #     sketch = AMS(data,
    #                  depth,
    #                  sign_hashes=sign_hashes,
    #                  exact_preds=True,)
    # elif method in ('bound-sketch', 'bound-sketch-v2', 'count-min'):
    #     sketch = BoundSketch(data,
    #                          depth,
    #                          width,
    #                          sign_hashes=sign_hashes,
    #                          bin_hashes=bin_hashes,
    #                          exact_preds=True, sparse=sparse)
    # else:
    #     assert method == 'count-sketch'
    sketch = FastAGMS(data,
                            depth,
                            width,
                            sign_hashes=sign_hashes,
                            bin_hashes=bin_hashes,
                            exact_preds=True, sparse=sparse,
                            sample_sketch=sample_sketch,
                            method=method)

    return sketch

def cross_correlate(node, query, alias2sketch, visited=None):
    """Should not modify the sketches. Avoid in-place operations, e.g., *=, +=..."""
    if visited is None:
        visited = set()
    alias, key = node.split('.')
    visited.add(node)
    sketch = alias2sketch[alias]

    for other_node in query.joined_nodes(alias):
        # skip current node
        if other_node == node:
            continue
        visited.add(other_node)
        tmp = 1
        correlated = False
        for joined_node in query.joined_with(other_node):
            if joined_node not in visited:
                tmp = tmp * cross_correlate(joined_node, query, alias2sketch, visited=visited)
                correlated = True
        if correlated:
            sketch = ifft(fft(tmp).conj() * fft(sketch)).real

    for joined_node in query.joined_with(node).difference(visited):
        sketch = sketch * cross_correlate(joined_node, query, alias2sketch, visited=visited)
    return sketch

def bound_unfiltered_degree_estimate(query, models, exact, cuda=False, exact_prob=False, exact_degree=False):
    inference_times = []
    sketching_times = []
    copy_times = []

    sketch_list = []
    tables = tuple(query.table_mapping_iter())
    for alias_count, name_count in tables:
        sketches = dict()

        predicates = dict() if alias_count not in query.selects else query.selects[alias_count]
        keys = query.alias2joined_attrs[alias_count]
        components = dict()
        for attr in keys:
            node = f"{alias_count}.{attr}"
            components[attr] = query.node2component[node]
        print(f'{name_count} components', components)

        t0 = perf_counter_ns()
        if isinstance(models[name_count], SPN):
            # use iterative inference method of SPN
            output = models[name_count].iterative(predicates, keys, components=components, count=True, exact_prob=exact_prob, cuda=cuda)
        else:
            output = models[name_count](predicates, keys, components=components, count=True, exact_prob=exact_prob, cuda=cuda)
        if len(output) == 2:
            # exact sketches don't have a separate copy time
            sketch, sketch_time = output
            copy_time = 0
        else:
            sketch, sketch_time, copy_time = output
        t1 = perf_counter_ns()
        inference_times.append(t1 - t0 - sketch_time - copy_time)
        sketching_times.append(sketch_time)
        copy_times.append(copy_time)

        # double check before anythng else
        if isinstance(sketch, (int, float)):
            assert sketch == 0, sketch
            print(f"Sketch of {alias_count}({keys.values()}) having {predicates} is 0")
            return (0,
                    pd.Timedelta(sum(inference_times), unit='ns'),
                    pd.Timedelta(sum(sketching_times), unit='ns'),
                    pd.Timedelta(sum(copy_times), unit='ns'),
                    pd.Timedelta(0))
        
        # print(sketch)
        sketches[alias_count] = sketch if not cuda else sketch.cuda()

        for alias_degree, name_degree in tables:
            if alias_count == alias_degree:
                continue
            if alias_degree in query.selects and exact_degree:
                predicates = query.selects[alias_degree]
            else:
                predicates = dict()
            keys = query.alias2joined_attrs[alias_degree]
            components = dict()
            for attr in keys:
                node = f"{alias_degree}.{attr}"
                components[attr] = query.node2component[node]
            print(f'{name_degree} components', components)

            t0 = perf_counter_ns()
            output = exact[name_degree](predicates, keys, components=components, count=False, exact_prob=exact_prob, cuda=cuda)
            if len(output) == 2:
                # exact sketches don't have a separate copy time
                sketch, sketch_time = output
                copy_time = perf_counter_ns() - t0
            else:
                sketch, sketch_time, copy_time = output
            sketching_times.append(sketch_time)
            copy_times.append(copy_time)

            # double check before anythng else
            if isinstance(sketch, (int, float)):
                assert sketch == 0, sketch
                print(f"Sketch of {alias_degree}({keys.values()}) having {predicates} is 0")
                return (0,
                        pd.Timedelta(sum(inference_times), unit='ns'),
                        pd.Timedelta(sum(sketching_times), unit='ns'),
                        pd.Timedelta(sum(copy_times), unit='ns'),
                        pd.Timedelta(0))
            
            # print(sketch)
            sketches[alias_degree] = sketch if not cuda else sketch.cuda()

        sketch_list.append(sketches)

    total_inference = pd.Timedelta(sum(inference_times), unit='ns')
    total_sketching = pd.Timedelta(sum(sketching_times), unit='ns')
    total_copying = pd.Timedelta(sum(copy_times), unit='ns')

    estimates = []

    t0 = perf_counter_ns()

    for sketches in sketch_list:
        start_node = query.random_node()

        sketch_estimates = cross_correlate(start_node, query, sketches).sum(dim=1)
        print(f'sketch estimates {sketch_estimates.shape}:')
        print(sketch_estimates)

        estimates.append(sketch_estimates.min().item())

    t1 = perf_counter_ns()
    estimation_time = pd.Timedelta(t1-t0, unit='ns')
    
    return min(estimates), total_inference, total_sketching, total_copying, estimation_time


def count_estimate(query, models, cuda=False, method='count-sketch', percentile=[0.5, 0.75, 1.0], exact_prob=False, mean=False, exact=None, independence=None, risk_adaptive=False, non_negative=False):
    clear_canonical_cache()
    inference_times = []
    sketching_times = []
    copy_times = []

    sketches = dict()
    use_count = True
    l1_bounds = dict()
    for alias, name in query.table_mapping_iter():
        predicates = query.selection_predicates.get(alias)
        keys = query.alias2joined_attrs[alias]
        components = dict()
        for attr in keys:
            node = f"{alias}.{attr}"
            components[attr] = query.node2component[node]
        print(f'{name} components', components)

        t0 = perf_counter_ns()
        if isinstance(models[name], SPN):
            # use iterative inference method of SPN
            output = models[name](predicates, keys, components=components, count=use_count, exact_prob=exact_prob, cuda=cuda)
        else:
            output = models[name](predicates, keys, components=components, count=use_count, exact_prob=exact_prob, cuda=cuda)
        if len(output) == 2:
            # exact sketches don't have a separate copy time
            sketch, sketch_time = output
            copy_time = 0
        else:
            sketch, sketch_time, copy_time = output
        t1 = perf_counter_ns()

        inference_times.append(t1 - t0 - sketch_time - copy_time)
        sketching_times.append(sketch_time)
        copy_times.append(copy_time)
        print(f'\t|___ {"Inference Time:":<20} {pd.Timedelta(inference_times[-1], unit="ns")}')
        print(f'\t|___ {"Sketching Time:":<20} {pd.Timedelta(sketching_times[-1], unit="ns")}')
        print(f'\t\\___ {"Copy Time:":<20} {pd.Timedelta(copy_times[-1], unit="ns")}')

        # double check before anythng else
        if isinstance(sketch, (int, float)):
            assert sketch == 0, sketch
            print(f"Sketch of {alias}({keys.values()}) having {predicates} is 0")
            return (0,
                    pd.Timedelta(sum(inference_times), unit='ns'),
                    pd.Timedelta(sum(sketching_times), unit='ns'),
                    pd.Timedelta(sum(copy_times), unit='ns'),
                    pd.Timedelta(0))
        
        # check error bound between approximate and exact count sketch
        if predicates is not None and use_count and exact and independence:
            best_case, _ = exact[name](predicates, keys, components=components, count=use_count, exact_prob=exact_prob)
            worst_case, _, _ = independence[name].iterative(predicates, keys, components=components, count=use_count, exact_prob=exact_prob)
            l1_dist = abs(best_case - sketch).sum().item()
            l1_upper = abs(best_case - worst_case).sum().item()
            l1_bounds[name] = (l1_dist, l1_upper)

        # print(sketch)
        sketches[alias] = sketch if not cuda else sketch.cuda()

    total_inference = pd.Timedelta(sum(inference_times), unit='ns')
    total_sketching = pd.Timedelta(sum(sketching_times), unit='ns')
    total_copying = pd.Timedelta(sum(copy_times), unit='ns')

    factors = None
    if method == 'bound-sketch':
        # setup combinations of degree and count sketches
        combinations = {alias: [] for alias in sketches.keys()}
        for idx, (alias, bound_sketch) in enumerate(sketches.items()):
            combinations[alias] += [bound_sketch[:,:,1], bound_sketch[:,:,0]]
            for other, other_sketch in sketches.items():
                if other != alias:
                    combinations[other] += [other_sketch[:,:,0], other_sketch[:,:,1]]
        sketches = {alias: torch.concat(combo, 0) for alias, combo in combinations.items()} 
    elif method == 'factorjoin':
        # won't work: can't track minimum factor elementwise due to FFT
        # determine scaling factor: min of all counts/degrees
        candidate_list = []
        for idx, (alias, bound_sketch) in enumerate(sketches.items()):
            quotient = torch.where(bound_sketch[:,:,0] != 0, bound_sketch[:,:,1] / bound_sketch[:,:,0], 0)
            candidate_list.append(quotient)
        candidates = torch.concat(candidate_list, 1)
        factors, _ = torch.max(candidates, dim=1)
        sketches = {alias: s[:,:,0] for alias, s in sketches.items()}


    sketches = {alias: s.to_dense() if s.is_sparse else s for alias, s in sketches.items()}
    
    # estimate Chernoff error bound for estimates (e.g., if estimate Z < -epsilon, the estimate is too noisy to be trusted)
    epsilon = 1
    for alias, sketch in sketches.items():
        epsilon *= (sketch ** 2).sum(dim=1).median().item()
    epsilon = (epsilon / sketches[alias].shape[1]) ** 0.5
    print(f'epsilon bound for estimates: {epsilon}')

    t0 = perf_counter_ns()
    sketch_products = cross_correlate(query.random_node(), query, sketches)
    sketch_estimates = sketch_products.sum(dim=1)
    if factors is not None:
        assert factors.shape == sketch_estimates.shape, factors.shape
        sketch_estimates *= factors

    print(f'sketch estimates {sketch_estimates.shape}:')
    print(sketch_estimates)

    t1 = perf_counter_ns()
    estimation_time = pd.Timedelta(t1-t0, unit='ns')
    
    est = {f"p{p}": sketch_estimates.quantile(p).item() for p in percentile}

    if mean:
        est['mean'] = sketch_estimates.mean().item()

    if risk_adaptive:
        max_est = sketch_estimates.quantile(1).item()
        median_est = sketch_estimates.quantile(0.5).item()
        spread = (max_est - median_est) / median_est
        if spread > 1.0:
            # if max is twice the median, use max
            est['adaptive'] = max_est
        else:
            est['adaptive'] = median_est
    
    if non_negative:
        sketch_estimates = sketch_products.clip(min=0).sum(dim=1)
        if factors is not None:
            sketch_estimates *= factors
        print(f'non-negative sketch estimates {sketch_estimates.shape}:')
        print(sketch_estimates)
        est |= {f"p{p}_nn": sketch_estimates.quantile(p).item() for p in percentile}
        if mean:
            est['mean_nn'] = sketch_estimates.mean().item()
        if risk_adaptive:
            max_est = sketch_estimates.quantile(1).item()
            median_est = sketch_estimates.quantile(0.5).item()
            spread = (max_est - median_est) / median_est if median_est != 0 else float('inf')
            if spread > 1.0:
                # if max is twice the median, use max
                est['adaptive_nn'] = max_est
            else:
                est['adaptive_nn'] = median_est
    
    return est, total_inference, total_sketching, total_copying, estimation_time, l1_bounds, epsilon

def bound_estimate(query, models, cuda=False, exact_prob=False, exact=None, independence=None):
    inference_times = []
    sketching_times = []
    copy_times = []

    count_sketches = dict()
    degree_sketches = dict()
    l1_bounds = dict()
    for alias, name in query.table_mapping_iter():
        predicates = dict() if alias not in query.selects else query.selects[alias]
        keys = query.alias2joined_attrs[alias]
        components = dict()
        for attr in keys:
            node = f"{alias}.{attr}"
            components[attr] = query.node2component[node]
        print(f'{name} components', components)

        t0 = perf_counter_ns()
        if isinstance(models[name], SPN):
            # use iterative inference method of SPN
            output = models[name].iterative(predicates, keys, components=components, count=True, exact_prob=exact_prob, cuda=cuda)
        else:
            output = models[name](predicates, keys, components=components, count=True, exact_prob=exact_prob, cuda=cuda)
        if len(output) == 2:
            # exact sketches don't have a separate copy time
            sketch, sketch_time = output
            copy_time = 0
        else:
            sketch, sketch_time, copy_time = output
        t1 = perf_counter_ns()
        count_inference_time = t1 - t0 - sketch_time - copy_time
        sketching_times.append(sketch_time)
        copy_times.append(copy_time)

        # double check before anythng else
        if isinstance(sketch, (int, float)):
            assert sketch == 0, sketch
            print(f"Sketch of {alias}({keys.values()}) having {predicates} is 0")
            return (0,
                    pd.Timedelta(sum(inference_times), unit='ns'),
                    pd.Timedelta(sum(sketching_times), unit='ns'),
                    pd.Timedelta(sum(copy_times), unit='ns'),
                    pd.Timedelta(0))
        
        # check error bound between approximate and exact count sketch
        if predicates and exact and independence:
            best_case, _ = exact[name](predicates, keys, components=components, count=True, exact_prob=exact_prob)
            worst_case, _, _ = independence[name].iterative(predicates, keys, components=components, count=True, exact_prob=exact_prob)
            l1_dist = abs(best_case - sketch).sum().item()
            l1_upper = abs(best_case - worst_case).sum().item()
            l1_bounds[name] = (l1_dist, l1_upper)

        # print(sketch)
        count_sketches[alias] = sketch if not cuda else sketch.cuda()

        # repeat for degree sketches
        t0 = perf_counter_ns()
        if isinstance(models[name], SPN):
            # use iterative inference method of SPN
            output = models[name].iterative(predicates, keys, components=components, count=False, exact_prob=exact_prob, cuda=cuda)
        else:
            output = models[name](predicates, keys, components=components, count=False, exact_prob=exact_prob, cuda=cuda)
        if len(output) == 2:
            # exact sketches don't have a separate copy time
            sketch, sketch_time = output
            copy_time = 0
        else:
            sketch, sketch_time, copy_time = output
        t1 = perf_counter_ns()
        inference_times.append(count_inference_time + t1 - t0 - sketch_time - copy_time)
        sketching_times.append(sketch_time)
        copy_times.append(copy_time)

        # double check before anythng else
        if isinstance(sketch, (int, float)):
            assert sketch == 0, sketch
            print(f"Sketch of {alias}({keys.values()}) having {predicates} is 0")
            return (0,
                    pd.Timedelta(sum(inference_times), unit='ns'),
                    pd.Timedelta(sum(sketching_times), unit='ns'),
                    pd.Timedelta(sum(copy_times), unit='ns'),
                    pd.Timedelta(0))

        degree_sketches[alias] = sketch if not cuda else sketch.cuda()

    total_inference = pd.Timedelta(sum(inference_times), unit='ns')
    total_sketching = pd.Timedelta(sum(sketching_times), unit='ns')
    total_copying = pd.Timedelta(sum(copy_times), unit='ns')

    estimates = []

    t0 = perf_counter_ns()
    for count_alias, count_sketch in count_sketches.items():
        # combine sketches, such that only one sketch is a count sketch
        sketches = {degree_alias: degree_sketch for degree_alias, degree_sketch in degree_sketches.items() if degree_alias != count_alias}
        # add count sketch to the sketches
        sketches[count_alias] = count_sketch

        # cross correlate sketches
        start_node = query.random_node()

        sketch_estimates = cross_correlate(start_node, query, sketches).sum(dim=1)
        print(f'sketch estimates {sketch_estimates.shape}:')
        print(sketch_estimates)

        estimates.append(sketch_estimates.min().item())

    t1 = perf_counter_ns()
    estimation_time = pd.Timedelta(t1-t0, unit='ns')
    
    return min(estimates), total_inference, total_sketching, total_copying, estimation_time, l1_bounds, None

def same_sign_estimate(query, models, cuda=False, percentile=0.5, exact_prob=False, mean=False):
    """not implemented for non-transitive joins yet"""
    inference_times = []
    sketching_times = []
    copy_times = []

    sketches = dict()
    negatives = dict()
    num_factors = 0
    for alias, name in query.table_mapping_iter():
        predicates = dict() if alias not in query.selects else query.selects[alias]
        keys = query.alias2joined_attrs[alias]
        components = dict()
        for attr in keys:
            node = f"{alias}.{attr}"
            components[attr] = query.node2component[node]
        print(f'{name} components', components)

        num_factors += len(keys)

        t0 = perf_counter_ns()
        if isinstance(models[name], SPN):
            # use iterative inference method of SPN
            output = models[name].iterative(predicates, keys, components=components, exact_prob=exact_prob, cuda=cuda, separate_negatives=True)
        else:
            output = models[name](predicates, keys, components=components, exact_prob=exact_prob, cuda=cuda, separate_negatives=True)
        if len(output) == 2:
            # exact sketches don't have a separate copy time
            sketch, sketch_time = output
            copy_time = 0
        else:
            sketch, sketch_time, copy_time = output
        t1 = perf_counter_ns()
        inference_times.append(t1 - t0 - sketch_time - copy_time)
        sketching_times.append(sketch_time)
        copy_times.append(copy_time)

        # double check before anythng else
        if isinstance(sketch, (int, float)):
            assert sketch == 0, sketch
            print(f"Sketch of {alias}({keys.values()}) having {predicates} is 0")
            return (0,
                    pd.Timedelta(sum(inference_times), unit='ns'),
                    pd.Timedelta(sum(sketching_times), unit='ns'),
                    pd.Timedelta(sum(copy_times), unit='ns'),
                    pd.Timedelta(0))
        
        # print(sketch)
        width = sketch.shape[-1]
        sketches[alias] = sketch[:, :width//2] if not cuda else sketch[:, :width//2].cuda()
        negatives[alias] = sketch[:, width//2:] if not cuda else sketch[:, width//2:].cuda()

    # sum for total sequential inference time
    # max for longest parallel inference time of all models
    total_inference = pd.Timedelta(sum(inference_times), unit='ns')
    total_sketching = pd.Timedelta(sum(sketching_times), unit='ns')
    total_copying = pd.Timedelta(sum(copy_times), unit='ns')

    t0 = perf_counter_ns()
    start_node = query.random_node()

    sketch_products = cross_correlate(start_node, query, sketches)
    # sketch_estimates = torch.where(negative_components < 0, sketch_products - 2 * negative_components, sketch_products).sum(dim=1)
    if num_factors % 2 == 0:
        sketch_estimates = sketch_products.sum(dim=1)
    else:
        negative_components = cross_correlate(start_node, query, negatives)
        assert negative_components.max() <= 0, f"negative components {negative_components.max()}"
        corrected_product = sketch_products - 2 * negative_components
        sketch_estimates = corrected_product.sum(dim=1)
    print(f'sketch estimates {sketch_estimates.shape}:')
    print(sketch_estimates)

    if mean:
        est = sketch_estimates.mean().item()
    else:
        est = sketch_estimates.quantile(percentile).item() # negative estimates are allowed

    t1 = perf_counter_ns()
    estimation_time = pd.Timedelta(t1-t0, unit='ns')
    
    return est, total_inference, total_sketching, total_copying, estimation_time

if __name__ == '__main__':
    import argparse

    from query_modernized import Query
    from dataset import get_dataframe, get_workload
    import experiments
    
    parser = argparse.ArgumentParser(description='run sketched sum-product networks on a workload')
    parser.add_argument('--method', type=str.lower, default='count-sketch', choices=['ams', 'count-sketch', 'count-min', 'bound-sketch', 'factorjoin'], help='depth of sketches')
    parser.add_argument('--depth', default=5, type=lambda x: int(float(x)), help='depth of sketches')
    parser.add_argument('--width', default=100000, type=lambda x: int(float(x)), help='width of sketches')
    parser.add_argument('--workload', default=Path('./workloads/stats_CEB_sub_queries_corrected.sql'), type=Path, help='CSV containing the format (subqueries || parent ID || cardinality)')
    parser.add_argument('--data', default=Path('./End-to-End-CardEst-Benchmark-master/datasets/stats_simplified/'), type=Path, help='path to directory containing table CSVs')
    parser.add_argument('--writefile', default=Path('out.csv'), type=Path, help='name of output csv file')
    parser.add_argument('--k', default=2, type=int, help='each Sum Node partitions data into k>=2 clusters')
    parser.add_argument('--decompose', '--rdc_threshold', default=0.01, type=float, help='group columns with pairwise RDC above this threshold')
    parser.add_argument('--min_cluster', default=0.1, type=float, help='minimum clustering size for sum nodes, i.e., treated as a percentage if less than 1')
    parser.add_argument('--cluster_first', action='store_true', help='force the root layer to be a Sum Node (cluster first)')
    parser.add_argument('--experiment', type=str.lower, default='stats-ceb', choices=['job-light', 'stats-ceb', 'job', 'stats-sqlstorm'])
    parser.add_argument('--independence', default=4, type=int, help='independence of k-universal hashing for sketches')
    parser.add_argument('--pessimistic', action='store_true', help='use pessimistic approximation (use with --percentile 1 for max estimator)')
    parser.add_argument('--pickle', default=None, type=Path, help='path to directory to save featurized data for faster subsequent runs')
    parser.add_argument('--cuda', action='store_true', help='use GPU for estimation (may reduce estimation time with larger sketches)')
    parser.add_argument('--exact_sketch', action='store_true', help='use exact sketches for estimation')
    parser.add_argument('--percentile', default=[0.5], type=float, nargs='*', help='percentile of [depth] estimates used as final estimate, e.g., 0.5 for median (default) and 1 for max')
    parser.add_argument('--kmeans', action='store_true', help='use K-means to learn sum nodes (slightly faster, might increase model size)')
    # parser.add_argument('--exact_preds', action='store_true', help='use exact selectivity of predicates in leaf nodes, instead of sketch estimates')
    parser.add_argument('--mean', action='store_true', help='use mean estimator')
    parser.add_argument('--risk_adaptive', action='store_true', help='use risk-adaptive strategy for count sketch estimator')
    parser.add_argument('--non_negative', action='store_true', help='use non-negative constraint for count sketch estimator')
    parser.add_argument('--selectivity_estimator', type=str.lower, default='count-min', choices=['exact', 'count-min', 'count-sketch'], help='selectivity estimator in leaf nodes')
    parser.add_argument('--check_error', action='store_true', help='compute exact sketch and independence assumption sketch for comparison (best used with exact selectivity)')
    parser.add_argument('--sparse', action='store_true', help='store sketches as sparse tensors (best if sketch width is also large)')
    parser.add_argument('--transform', default='rdc', choices=['rdc', 'hash'])
    parser.add_argument('--skip_to', default=0, type=int, help='skip to this query index in the workload')
    parser.add_argument('--parent', default=None, help='only run queries with this parent ID in the workload')
    parser.add_argument('--truncate_tables', default=None, type=lambda x: int(float(x)), help='truncate this many rows from each table for training (for faster experiments)')
    parser.add_argument('--sample_selectivity', default=None, type=float, help='sample this percentage of data in leaf nodes (Exact selectivity -> approximation)')
    parser.add_argument('--sample_model', default=None, type=float, help='sample this percentage of data in leaf nodes (Exact selectivity -> approximation)')
    parser.add_argument('--sample_sketch', default=None, type=float, help='sample this percentage of data for sketching (does not affect model training)')
    args = parser.parse_args()

    if args.method == 'ams':
        args.width = 1
    elif args.width == 1 and args.method == 'count-sketch':
        args.method = 'ams'
    elif args.method in ('bound-sketch', 'count-min', 'factorjoin') and (0 not in args.percentile):
        args.percentile.append(0)
    args.percentile.sort()

    print(args)

    dates = experiments.get_date_cols(args.experiment)
    strings = experiments.get_string_cols(args.experiment)
    intervals = experiments.get_range_intervals(args.experiment)
    tables = experiments.get_tables_cols(args.experiment)

    num_components = len(tables) # this suffices for acyclic joins
    bin_hashes = [BinHash(args.depth, args.width) for _ in range(num_components)]
    sign_hashes = [SignHash(args.depth, k=args.independence) for _ in range(2 * num_components)]

    workload = get_workload(args.workload)

    if args.parent is not None:
        workload = workload[workload['parent'] == args.parent]
        workload = workload.reset_index(drop=True)
        assert not workload.empty, f"No queries with parent ID {args.parent}"

    for percentile in args.percentile:
        workload[f"{args.method}_{args.depth}x{args.width}_p{percentile}"] = -1.0
        workload[f"{args.method}_{args.depth}x{args.width}_p{percentile}_err"] = -1.0
        if args.non_negative:
            workload[f"{args.method}_{args.depth}x{args.width}_p{percentile}_nn"] = -1.0
            workload[f"{args.method}_{args.depth}x{args.width}_p{percentile}_nn_err"] = -1.0

    if args.mean:
        workload[f"{args.method}_{args.depth}x{args.width}_mean"] = -1.0
        workload[f"{args.method}_{args.depth}x{args.width}_mean_err"] = -1.0
        if args.non_negative:
            workload[f"{args.method}_{args.depth}x{args.width}_mean_nn"] = -1.0
            workload[f"{args.method}_{args.depth}x{args.width}_mean_nn_err"] = -1.0

    if args.risk_adaptive:
        workload[f"{args.method}_{args.depth}x{args.width}_adaptive"] = -1.0
        workload[f"{args.method}_{args.depth}x{args.width}_adaptive_err"] = -1.0
        if args.non_negative:
            workload[f"{args.method}_{args.depth}x{args.width}_adaptive_nn"] = -1.0
            workload[f"{args.method}_{args.depth}x{args.width}_adaptive_nn_err"] = -1.0

    workload['num_tables'] = 0
    workload['join_components'] = 0
    workload['inference_time'] = pd.Timedelta(0.0, unit='sec')
    workload['sketching_time'] = pd.Timedelta(0.0, unit='sec')
    workload['copy_overhead'] = pd.Timedelta(0.0, unit='sec')
    workload['estimation_time'] = pd.Timedelta(0.0, unit='sec')
    workload['total_time'] = pd.Timedelta(0.0, unit='sec')
    workload['query_memory'] = 0
    workload['epsilon'] = 0
    if args.check_error:
        workload['L1'] = pd.NA
        workload['L1_bound'] = pd.NA
    workload = workload.copy()
    print(f'Generating results into workload ({workload.shape})')

    with torch.inference_mode():
        if args.exact_sketch:
            models = exact = dict()
        else:
            models = dict()
            exact = dict()
        independence_models = dict()

        training_times = []
        for table, meta in tables.items():
            ts = perf_counter_ns()
            dataset = get_dataframe(f'{args.data}/{table}.csv',
                                    names=meta['names'],
                                    columns=meta['col_types'].keys(),
                                    dates=dates.get(table),
                                    strings=strings.get(table),)
            if args.truncate_tables is not None and len(dataset) > args.truncate_tables:
                dataset = dataset.sample(n=args.truncate_tables, random_state=42).reset_index(drop=True)
            delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
            print(f"Loaded {table} ({dataset.memory_usage(deep=True).sum() / 2**20:,} MiB) {delta.total_seconds():>25,.2f}s ({delta})")
            print(dataset.describe().to_string(float_format="{:,.2f}".format))
            print(dataset.memory_usage(deep=True).to_string(float_format="{:,.2f}".format))

            if args.exact_sketch or args.method == 'bound-sketch-unfiltered' or args.check_error:
                # bound sketch approximation still uses exact degrees without pushdown
                exact[table] = exact_sketch(dataset, bin_hashes=bin_hashes, sign_hashes=sign_hashes, method=args.method, sparse=args.sparse, sample_sketch=args.sample_sketch)
                
            scale_factor=1
            if args.sample_model is not None:
                if 0 < args.sample_model < 1:
                    # Treat as percentage
                    sample_size = int(len(dataset) * args.sample_model)
                    sample_size = max(sample_size, 1000)
                elif len(dataset) > args.sample_model >= 1:
                    # Treat as absolute number
                    sample_size = int(args.sample_model)
                else:
                    # Ignore
                    sample_size = len(dataset)
                sample_size = min(len(dataset), sample_size)
                scale_factor = len(dataset) / (sample_size)
                dataset = dataset.sample(n=sample_size).reset_index(drop=True)
                assert scale_factor >= 1

            if not args.exact_sketch:
                ts = perf_counter_ns()
                # extract features before training
                if args.pickle and not (args.sample_model or args.truncate_tables):
                    save_path = args.pickle / f"{table}.pkl"
                    if save_path.exists():
                        features = pd.read_pickle(save_path)
                        delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
                        print(f"Loaded pickled features from {save_path} ({delta})")
                        assert len(features) == len(dataset), f"Features ({save_path}) do not match ({args.data/table}.csv)"
                    else:
                        print(f"Extracting features from {table} ...", flush=True)
                        args.pickle.mkdir(parents=True, exist_ok=True)
                        if args.transform == 'rdc':
                            features = RDC.generate_rdc_features_inplace(dataset, meta['col_types'])
                        else:
                            features = kwisehash_transform(dataset,
                                                           bin_hash=bin_hashes[0],
                                                           intervals=intervals[table] if table in intervals else None)
                        delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
                        print(f"Extracted features from {table} ({delta})", flush=True)
                        save_path = f"{args.pickle}/{table}.pkl"
                        features.to_pickle(save_path)
                else:
                    print(f"Extracting features from {table} ...", flush=True)
                    if args.transform == 'rdc':
                        features = RDC.generate_rdc_features_inplace(dataset, meta['col_types'])
                    else:
                        features = kwisehash_transform(dataset,
                                                        bin_hash=bin_hashes[0],
                                                        intervals=intervals[table] if table in intervals else None)
                    delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
                    print(f"Extracted features from {table} ({delta})", flush=True)
                
                # minimum size of clusters in sum nodes
                min_cluster = args.min_cluster if args.min_cluster > 1 else abs(args.min_cluster * len(dataset))

                # convert dataframes to use pyarrow backend for faster ops
                # dataset = dataset.convert_dtypes(dtype_backend='pyarrow')
                # features = features.convert_dtypes(dtype_backend='pyarrow') # don't convert this one

                # create independence assumption models for error checking
                if args.check_error:
                    independence_models[table] = SPN(dataset, features, 
                                              bin_hashes=bin_hashes, 
                                              sign_hashes=sign_hashes, 
                                              corr_threshold=1, 
                                              min_cluster=len(dataset), 
                                              num_clusters=args.k, 
                                              cluster_next=args.cluster_first,
                                              keys=meta['keys'], 
                                              method=args.method, 
                                              pessimistic=args.pessimistic, 
                                              use_kmeans=args.kmeans,
                                              meta_types=meta['col_types'],
                                              intervals=intervals[table] if table in intervals else None,
                                              selectivity_estimator=args.selectivity_estimator,
                                              scale_factor=scale_factor)

                # train SPN on features
                ts = perf_counter_ns()
                models[table] = SPN(dataset, features, 
                                    bin_hashes=bin_hashes, 
                                    sign_hashes=sign_hashes, 
                                    corr_threshold=args.decompose, 
                                    min_cluster=min_cluster, 
                                    num_clusters=args.k, 
                                    cluster_next=args.cluster_first,
                                    keys=meta['keys'], 
                                    method=args.method, 
                                    pessimistic=args.pessimistic, 
                                    use_kmeans=args.kmeans,
                                    meta_types=meta['col_types'],
                                    intervals=intervals[table] if table in intervals else None,
                                    selectivity_estimator=args.selectivity_estimator,
                                    sparse=args.sparse,
                                    sample_selectivity=args.sample_selectivity,
                                    sample_sketch=args.sample_sketch,
                                    scale_factor=scale_factor)
                del features # features are no longer necessary
                
            delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
            print(f"{'Hashed data' if args.exact_sketch else f'Trained SPN(scale={scale_factor:.2f})'} ({models[table].memory / 2**20:,.2f} MiB) on {table} ({delta})", flush=True)
            training_times.append(delta)

        total_training = sum(training_times, pd.Timedelta(0))

        if not args.exact_sketch:
            print(f"Total Structure Learning Time: {total_training} (avg. {total_training / len(models)})")
            print(f"Models: {list(models.keys())}")

        cum_sketching  = pd.Timedelta(0)
        cum_inference  = pd.Timedelta(0)
        cum_copying    = pd.Timedelta(0)
        cum_estimation = pd.Timedelta(0)
        cum_total      = pd.Timedelta(0)
        queries_done   = 0
        total_queries  = len(workload) - args.skip_to

        query = None
        num_selections = 0
        for i, row in enumerate(workload.iloc()):
            if i < args.skip_to or (args.parent is not None and row['parent'] != args.parent):
                continue
            query_start = perf_counter_ns()
            sql = row['query']
            query = Query(sql, history=query)
            # nodes, edges = extract_graph(sql)
            # num_components = 1 + sum(len(n.keys)-1 for n in nodes)
            # if query.num_components == 1: continue

            print(f"{i}: {query} ({row['cardinality']:,})")

            mem_before = sum(m.memory_usage() for m in models.values())
            l1_bounds = dict()
            # if args.method == 'bound-sketch-unfiltered':
            #     est, inference_time, sketching_time, copying_time, estimation_time = bound_unfiltered_degree_estimate(query, models, exact, cuda=args.cuda, exact_degree=args.exact_sketch)
            # elif args.method == 'bound-sketch':
            #     est, inference_time, sketching_time, copying_time, estimation_time, l1_bounds = bound_estimate(query, models, cuda=args.cuda, exact=exact, independence=independence_models)
            # else:
            est, inference_time, sketching_time, copying_time, estimation_time, l1_bounds, epsilon = count_estimate(query, models, cuda=args.cuda, method=args.method, percentile=args.percentile, mean=args.mean,
                                                                                                    exact=exact, independence=independence_models, risk_adaptive=args.risk_adaptive, non_negative=args.non_negative)
            mem_after = sum(m.memory_usage() for m in models.values())
            for estimator, value in est.items():
                name = f"{args.method}_{args.depth}x{args.width}_{estimator}"
                assert name in workload.columns, f"{name} not in workload columns"
                workload.at[i, name] = value
                workload.at[i, name + '_err'] = max(value, 1) / max(row['cardinality'], 1) if value >= row['cardinality'] else max(row['cardinality'], 1) / max(value, 1)
            # max(max(est, 1), row['cardinality']) / max(min(abs(est), row['cardinality']), 1)
            query_time = pd.Timedelta(perf_counter_ns() - query_start, unit='ns')

            workload.at[i, 'num_tables'] = len(query.alias2joined_attrs)
            workload.at[i, 'join_components'] = query.num_components
            workload.at[i, 'inference_time'] = inference_time
            workload.at[i, 'sketching_time'] = sketching_time
            workload.at[i, 'copy_overhead'] = copying_time
            workload.at[i, 'estimation_time'] = estimation_time
            workload.at[i, 'total_time'] = query_time
            workload.at[i, 'query_memory'] = mem_after - mem_before
            workload.at[i, 'epsilon'] = epsilon
            if len(l1_bounds) > 0:
                workload.at[i, 'L1'] = sum([L1_dist[0] for L1_dist in l1_bounds.values()])
                workload.at[i, 'L1_bound'] = sum([L1_dist[1] for L1_dist in l1_bounds.values()])
                num_selections += len(l1_bounds)

            cum_sketching  += sketching_time
            cum_inference  += inference_time
            cum_copying    += copying_time
            cum_estimation += estimation_time
            cum_total      += query_time
            queries_done   += 1

            print(workload.loc[i].to_string(float_format="{:,.2f}".format))
            print(f'Query {i} finished in {query_time.total_seconds():>25,.2f}s ({query_time})')
            print(f"--- Cumulative ({queries_done}/{total_queries} queries) ---")
            print(f"  {'Sketching:':<14} {str(cum_sketching):>26}  (avg {cum_sketching / queries_done})")
            print(f"  {'Inference:':<14} {str(cum_inference):>26}  (avg {cum_inference / queries_done})")
            print(f"  {'Copy:':<14} {str(cum_copying):>26}  (avg {cum_copying / queries_done})")
            print(f"  {'Estimation:':<14} {str(cum_estimation):>26}  (avg {cum_estimation / queries_done})")
            print(f"  {'Total:':<14} {str(cum_total):>26}  (avg {cum_total / queries_done})")
            print(f"  {'Memory:':<14} {mem_after / 2**30:>25.3f} GiB")
            print(flush=True)

        cols = list(workload.columns)
        for x in ('query', 'parent'):
            cols.remove(x)

        workload = workload[workload['num_tables'] > 0]

        drop = ['std']
        pctl = [0.25, 0.5, 0.75]

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

        print(f"\nTotal Sketching Time: {workload['sketching_time'].sum()} (avg. {workload['sketching_time'].mean()})")
        if not args.exact_sketch:
            print(f"Total Structure Learning Time: {total_training} (avg. {total_training / len(models)})")
            print(f"Total Model Inference Time: {workload['inference_time'].sum()} (avg. {workload['inference_time'].mean()})")
            print(f"Total Model Copying Overhead: {workload['copy_overhead'].sum()} (avg. {workload['copy_overhead'].mean()})")
        print(f"Total Estimation Time: {workload['estimation_time'].sum()} (avg. {workload['estimation_time'].mean()})")
        print(f"Total Workload Time: {workload['total_time'].sum()} (avg. {workload['total_time'].mean()})")

        # compute memory usage due to sketches after running workload
        model_mem_usage = sum([model.memory_usage() for model in models.values()])
        gb = model_mem_usage // 2**30
        mb = (model_mem_usage % 2**30) // 2**20
        kb = (model_mem_usage % 2**20) // 2**10
        b = model_mem_usage % 2**10
        print(f"Total memory usage: {gb:,} GiB  {mb:,} MiB  {kb:,} KiB  {b:,} B  (Total {model_mem_usage:,} bytes)")

        if args.check_error:
            print(f"Total approximation error (L1 distance) for {num_selections:,} selections: {workload['L1'].sum():,.2f} (avg. {workload['L1'].sum() / num_selections :,.2f}) <= worst-case {workload['L1_bound'].sum() :,.2f} (avg. {workload['L1_bound'].sum() / num_selections :,.2f})")

        if query is not None:
            # find maximum number of components and join edges in query history
            max_components = 0
            max_edges = 0
            for alias, sketch_ids in query.sketch_history.items():
                for component_id, edges in sketch_ids:
                    if component_id > max_components:
                        max_components = component_id
                    if max(edges) > max_edges:
                        max_edges = max(edges)

        print(f"End results for {args}")