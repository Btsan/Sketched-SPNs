from copy import deepcopy
from time import perf_counter_ns
from itertools import combinations
from typing import Set, List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sqlglot import expressions as exp

from RDC import rdc
from Sketches import AMS, FastAGMS, BoundSketch, ExactSelectivity, CountSketch, CountMin, StringCountMin
from predicate_to_string import _or_nodes_cache

def get_component(node, adjacency_mat):
    group = set()
    to_visit = {node,}
    while to_visit:
        curr_node = to_visit.pop()
        adjacency_mat[:, curr_node] = False
        neighbors = np.argwhere(adjacency_mat[curr_node, :]).flatten().tolist()
        to_visit.update(neighbors)
        group.add(curr_node)
    return group

def decompose(data, features, pairwise_corr, corr_thresh=0.3, min_cluster=1e5, terminate=False, keys={}):
    # print(f"keys {keys}, columns {data.columns}, intersection {keys.intersection(data.columns)}")
    if data.shape[0] <= max(1, min_cluster) or terminate is True:
        # completely decompose data regardless of correlations
        keys = keys.intersection(data.columns)
        if keys:
            k = list(keys)
            components = [data[col] for col in data.columns if col not in keys] + [data[k]]
            component_features = [features[col] for col in features.columns if col not in keys] + [features[k]]
        else:
            components = [data[col] for col in data.columns]
            component_features = [features[col] for col in features.columns]
        return components, component_features

    N, D = data.shape

    ungrouped = set(range(D))
    groups = []

    adjacency_mat = pairwise_corr > corr_thresh
    # group of columns that have a dependency on any key(s)
    # must be computed before other groups
    if keys.intersection(data.columns):
        g = set()
        for key in keys:
            node = data.columns.get_loc(key)
            if node in ungrouped:
                g |= get_component(node, adjacency_mat)
                adjacency_mat[list(g), :] = False
                adjacency_mat[:, list(g)] = False
                ungrouped -= g
        groups.append([data.columns[i] for i in g])

    # group remaining columns
    while ungrouped:
        node = ungrouped.pop()
        g = get_component(node, adjacency_mat)
        adjacency_mat[list(g), :] = adjacency_mat[:, list(g)] = False
        ungrouped -= g
        groups.append([data.columns[i] for i in g])

    components = [data[g] for g in groups]
    component_features = [features[g] for g in groups]
    return components, component_features

def cluster(data, features, k=2, gmm=None, max_sample_size=10000, use_kmeans=False):
    assert k >= 2, f"Invalid number of clusters: {k}"
    assert len(data) > 1, f"Not enough points ({len(data)}) to cluster"
    feat = features.values

    # t0 = perf_counter_ns()
    if use_kmeans:
        labels = KMeans(n_clusters=k).fit_predict(feat)
    else:
        if len(features) > max_sample_size:
            # t2 = perf_counter_ns()
            rng = np.random.default_rng()
            sample = rng.choice(feat, size=max_sample_size, replace=False)
            # t3 = perf_counter_ns()
            # print(f"Sampling time: {(t3 - t2) / 1e6:.2f} ms")
        else:
            sample = feat
        # default to EM
        if gmm is None:
            # increasing reg_covar may be necessary if sklearn raises an error
            gmm = GaussianMixture(n_components=k, covariance_type='diag', reg_covar=1e-4).fit(sample.astype(np.float64))
        else:
            assert isinstance(gmm, GaussianMixture), f"Invalid GMM type: {type(gmm)}"
            gmm = gmm.fit(sample)
        labels = gmm.predict(feat)
        # print(f"gmm labels {np.bincount(labels)}")

    # in case, prevents failure to cluster
    if (np.bincount(labels) > 0).sum() < 2:
        # fallback to random projection LSH
        rng = np.random.default_rng()
        normals = rng.random((feat.shape[1], 1)) - 0.5
        rand_proj = np.matmul(feat, normals)
        labels = np.sin(rand_proj).flatten() > 0
        # print(f"LSH labels {np.bincount(labels)}")

    if (np.bincount(labels) > 0).sum() < 2:
        # final fallback to even split
        labels = np.arange(feat.shape[0]) % 2
        # print(f"split labels {np.bincount(labels)}")

    # t1 = perf_counter_ns()
    # print(f"Clustering time: {(t1 - t0) / 1e6:.2f} ms")

    # t0 = perf_counter_ns()
    clusters = []
    cluster_features = []
    for cluster, group in data.groupby(labels.flatten()):
        if not group.empty:
            clusters.append(group)
    
    for cluster, group in features.groupby(labels.flatten()):
        if not group.empty:
            cluster_features.append(group)

    # t1 = perf_counter_ns()
    # print(f"Cluster extraction time: {(t1 - t0) / 1e6:.2f} ms")

    assert len(clusters) > 1, \
        f'cluster labels {labels}'
    return clusters, cluster_features, gmm

class SPN(object):
    """Mixed Sum-Product Networks (Molina et al., 2017)
    https://arxiv.org/pdf/1710.03297.pdf
    """
    def __init__(self, data, features, bin_hashes=None, sign_hashes=None, corr_threshold=0, min_cluster=1e5, num_clusters=2, cluster_next=False, level=0, verbose=True, keys=None, method='count-sketch', pessimistic=False, gmm=None, use_kmeans=False, meta_types=None, intervals=None, selectivity_estimator='count-min', sparse=False, sample_selectivity=None, sample_sketch=None, scale_factor=1.0):
        self.exact_preds = (selectivity_estimator == 'exact')
        if keys is None:
            keys = set()
        self.size = len(data)
        self.sketch_method = method
        self.scale_factor = scale_factor

        if isinstance(data, pd.Series):
            self.columns = {data.name,}
            self.dtypes = {data.name: data.dtype}
        else:
            self.columns = set(data.columns)
            self.dtypes = dict()
            for col in self.columns:
                self.dtypes[col] = data[col].dtype
                
        # if verbose: print(f'keys {keys}')
        if len(data.shape) == 1 or len(data.columns) == 1:
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'leaf node {data.name if isinstance(data, pd.Series) else data.columns}', end='')
            level += 1
            self.node = UnivariateLeaf(data,
                                       bin_hashes=bin_hashes, sign_hashes=sign_hashes, method=method, keys=keys, intervals=intervals,
                                       selectivity_estimator=selectivity_estimator, sparse=sparse, sample_selectivity=sample_selectivity, sample_sketch=sample_sketch)
            if verbose: print(f'({type(self.node.sketch)} {self.node.memory:,} bytes)')
        elif set(data.columns) == set(keys):
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'join node {tuple(data.columns)}', end='')
            level += 1
            self.node = JoinLeaf(data,
                                 bin_hashes=bin_hashes, sign_hashes=sign_hashes, method=method, sparse=sparse, sample_sketch=sample_sketch)
            if verbose: print(f'({type(self.node.sketch)} {self.node.memory:,} bytes)')
        elif cluster_next:
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'sum node {tuple(data.columns)}')
            clusters, cluster_features, gmm = cluster(data, features, k=num_clusters, gmm=gmm, use_kmeans=use_kmeans)
            level += 1
            self.node = SumNode(clusters, cluster_features,
                                bin_hashes=bin_hashes, sign_hashes=sign_hashes, corr_threshold=corr_threshold, min_cluster=min_cluster, num_clusters=num_clusters, level=level, keys=keys, method=method, pessimistic=pessimistic, gmm=gmm, use_kmeans=use_kmeans, verbose=verbose,
                                meta_types=meta_types, intervals=intervals, selectivity_estimator=selectivity_estimator, sparse=sparse, sample_selectivity=sample_selectivity, sample_sketch=sample_sketch)
        else:
            # print(f"2 {meta_types}")
            contained_types = {meta_types[col] for col in data.columns} if meta_types is not None else None
            corr_type = 'corr'
            sample_size = 4096
            if len(data) <= max(1, min_cluster):
                # skip rdc calculation and assume independence
                pairwise_corr = np.eye(data.shape[1])
            elif corr_threshold < 0:
                pairwise_corr = np.ones((data.shape[1], data.shape[1]))
            else:
                # print(f"3 {meta_types}")
                corr_type = 'RDC'
                assert len(data) == len(features)
                pairwise_corr = rdc(data=data,
                                    rdc_features=features.sample(sample_size) if len(features) > sample_size else features,
                                    meta_types=meta_types)
            # print(pairwise_corr, thresh)
            min_corr = pairwise_corr.min()
            components, component_features = decompose(data, features, pairwise_corr, corr_thresh=corr_threshold, min_cluster=min_cluster, keys=keys)
            if len(components) > 1:
                if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'product node {tuple(data.columns)}{data.shape}(min. {corr_type}={min_corr:.2e})')
                level += 1
                self.node = ProductNode(components, component_features, 
                                        bin_hashes=bin_hashes, sign_hashes=sign_hashes,
                                        corr_threshold=corr_threshold, min_cluster=min_cluster, num_clusters=num_clusters, level=level, keys=keys, method=method, pessimistic=pessimistic, use_kmeans=use_kmeans, verbose=verbose,
                                        meta_types=meta_types, intervals=intervals, selectivity_estimator=selectivity_estimator, sparse=sparse, sample_selectivity=sample_selectivity, sample_sketch=sample_sketch)
            else:
                if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'sum node {tuple(data.columns)}{data.shape}(min. {corr_type}={min_corr:.2e})')
                clusters, cluster_features, gmm = cluster(data, features, k=num_clusters, gmm=gmm, use_kmeans=use_kmeans)
                level += 1
                self.node = SumNode(clusters, cluster_features,
                                    bin_hashes=bin_hashes, sign_hashes=sign_hashes, corr_threshold=corr_threshold, min_cluster=min_cluster, num_clusters=num_clusters, level=level, keys=keys, method=method, pessimistic=pessimistic, gmm=gmm, use_kmeans=use_kmeans, verbose=verbose,
                                    meta_types=meta_types, intervals=intervals, selectivity_estimator=selectivity_estimator, sparse=sparse, sample_selectivity=sample_selectivity, sample_sketch=sample_sketch)

        self.memory = self.node.memory

        self.saved = dict()
        return
    
    def memory_usage(self):
        return self.node.memory_usage()
    
    def insert(self, data):
        """ Insert new data (a single tuple) into the SPN """
        pass

    def cast_predicates(self, predicates):
        """cast predicate values (presumably strings) to their correct datatype"""
        predicates = deepcopy(predicates)
        for col in predicates.keys():
            if col in self.dtypes:
                print(f"cast {col} to {self.dtypes[col]}")
                # if type is a datetime, convert to nanoseconds since last epoch
                use_nanoseconds = pd.api.types.is_datetime64_any_dtype(self.dtypes[col])
                for op, val in predicates[col].items():
                    print(f"\tcast {col} {op} {val} to {self.dtypes[col]}")
                    if str.upper(op) == 'BETWEEN':
                        val_1, val_2 = val.split(' AND ')
                        print(f"\t\tsplit {col} {op} {val} to {col} >= {val_1} AND {col} <= {val_2}")
                        if use_nanoseconds:
                            predicates[col]['>='] = pd.to_datetime(val_1).value
                            predicates[col]['<='] = pd.to_datetime(val_2).value
                        else:
                            predicates[col]['>='] = self.dtypes[col].type(val_1)
                            predicates[col]['<='] = self.dtypes[col].type(val_2)
                    elif use_nanoseconds:
                        predicates[col][op] = pd.to_datetime(val).value
                    else:
                        predicates[col][op] = self.dtypes[col].type(val)
        return predicates

    def __call__(self, predicates, key, components, _root=True, **kwargs):
        # if _root and not self.exact_preds:
        #     # cast predicate values to the correct type
        #     # do not cast if using exact selectivity
        #     predicates = self.cast_predicates(predicates)

        # col_in_preds = self.columns.intersection(predicates.keys())
        # col_in_keys = self.columns.intersection(key.keys())
        # sketch_id = frozenset(key.items()).union(components.items()).union(kwargs.items())
        # if not col_in_preds and not col_in_keys:
        #     return 1, 0, 0
        # elif _root and not col_in_preds and sketch_id in self.saved:
        #     return self.saved[sketch_id] if isinstance(self.saved[sketch_id], (int, float)) else self.saved[sketch_id].to_dense(), 0, 0

        sketch_or_prob, sketch_time, copy_time = self.node(predicates, key, components, _root=False, **kwargs)
        # if _root and not col_in_preds:
        #     # save sketch for later use
        #     if isinstance(sketch_or_prob, (int, float)):
        #         self.saved[sketch_id] = sketch_or_prob
        #     else:
        #         self.saved[sketch_id] = sketch_or_prob.to_sparse()
        return (self.scale_factor * sketch_or_prob), sketch_time, copy_time
    
    def iterative(self, predicates, key, components, **kwargs):
        # if not self.exact_preds:
        #     # cast predicate values to the correct type
        #     # do not cast if using exact selectivity
        #     predicates = self.cast_predicates(predicates)

        # col_in_preds = self.columns.intersection(predicates.keys())
        # sketch_id = frozenset(key.items()).union(components.items()).union(kwargs.items())
        # # print(col_in_preds, sketch_id, self.saved)
        # # print(type(col_in_preds), type(sketch_id), type(self.saved))
        # if not col_in_preds and sketch_id in self.saved:
        #     return self.saved[sketch_id] if isinstance(self.saved[sketch_id], (int, float)) else self.saved[sketch_id].to_dense(), 0, 0

        results = dict()

        # depth-first traversal
        stack = [self.node]
        visited = []
        
        # extract time for copying sketches from leaf nodes e.g., cpu to cuda
        copy_time = 0

        while stack:
            node = stack.pop()

            if isinstance(node, (SumNode, ProductNode)):
                if node not in visited:
                    # depth-first descending
                    visited.append(node)
                    stack.append(node)
                    for child in node.children:
                        stack.append(child.node)
                else:
                    # depth-first ascending
                    if isinstance(node, SumNode):
                        total_sketch_time = 0
                        freq = 0
                        sketch = None
                        for child in node.children:
                            sketch_or_prob, sketch_time = results.pop(child.node)
                            
                            total_sketch_time += sketch_time
                            if isinstance(sketch_or_prob, (int, float)):
                                freq += sketch_or_prob * child.size
                            elif sketch is None:
                                sketch = sketch_or_prob
                            else:
                                sketch += sketch_or_prob
                        results[node] = ((freq / node.size) if isinstance(sketch_or_prob, (int, float)) else sketch, total_sketch_time)
                    elif isinstance(node, ProductNode):
                        probs = [1]
                        sketch = None
                        sketch_time = 0
                        for child in node.children:
                            sketch_or_prob, sketch_time = results.pop(child.node)

                            if isinstance(sketch_or_prob, (int, float)):
                                probs.append(sketch_or_prob)
                            else:
                                assert sketch == None
                                sketch = sketch_or_prob
                        prob = min(probs) if node.pessimistic else np.prod(probs).item()
                        assert 0 <= prob <= 1, f"probability {prob} out of bounds"
                        results[node] = (sketch * prob if sketch is not None else prob, sketch_time)
            else:
                # leaf node
                t0 = perf_counter_ns()
                sketch_or_prob, sketch_time = results[node] = node.sketch(predicates, key, components, **kwargs)
                t1 = perf_counter_ns()
                if not isinstance(sketch_or_prob, (int, float)):
                    # overhead from sketch copying
                    copy_time += t1 - t0 - sketch_time

        # if not col_in_preds:
        #     # save sketch for later use
        #     if isinstance(results[self.node][0], (int, float)):
        #         self.saved[sketch_id] = results[self.node][0]
        #     else:
        #         self.saved[sketch_id] = results[self.node][0].to_sparse()
        return *results[self.node], copy_time

class UnivariateLeaf(object):
    def __init__(self, data, bin_hashes=None, sign_hashes=None, method='count-sketch', keys=None, intervals=None, selectivity_estimator='count-min', sparse=False, sample_selectivity=None, sample_sketch=None):
        if type(data) is pd.DataFrame:
            data = data[data.columns[0]]
        # self.data = data
        self.name = data.name
        self.size = len(data)

        """
        series.groupby(level=0) groups by the index rather than values
        the correct groupby method (for series) os series.groupby(series)

        for dataframes, it'd be df.groupby([str_name])
        """

        # if data is not a join key, use selectivity estimator instead of join sketches
        if keys is not None and self.name not in keys:
            # check if data is numeric or a date
            is_numeric = pd.api.types.is_numeric_dtype(data) or pd.api.types.is_datetime64_any_dtype(data)
            # if using exact selectivity or type is not supported
            if selectivity_estimator == 'exact':
                print(f"Exact selectivity for {self.name} ({data.shape}) {data.dtype})")
                self.sketch = ExactSelectivity(data, sample_selectivity=sample_selectivity)
            elif is_numeric and selectivity_estimator == 'count-sketch':
                self.sketch = CountSketch(data,
                                        depth=bin_hashes[0].depth,
                                        width=max(1000, bin_hashes[0].width // 1000),
                                        sign_hash=sign_hashes[0],
                                        bin_hash=bin_hashes[0],
                                        intervals=intervals[self.name] if intervals is not None and self.name in intervals else None,)
            elif is_numeric and selectivity_estimator == 'count-min':
                self.sketch = CountMin(data,
                                    depth=bin_hashes[0].depth,
                                    width=max(1000, bin_hashes[0].width // 1000),
                                    bin_hash=bin_hashes[0],
                                    intervals=intervals[self.name] if intervals is not None and self.name in intervals else None)
            else:
                self.sketch = ExactSelectivity(data, sample_selectivity=sample_selectivity)
                # self.sketch = StringCountMin(data,
                #                              depth=bin_hashes[0].depth,
                #                              width=max(1000, bin_hashes[0].width // 1000),
                #                              sign_hash=sign_hashes[0],
                #                              bin_hash=bin_hashes[0],
                #                              intervals=intervals[self.name] if intervals is not None and self.name in intervals else None,)
                # raise ValueError(f'Uknown selecitivity estimator {selectivity_estimator}')
        else:
            # if method == 'ams':
            #     self.sketch = AMS(data,
            #                     width=bin_hashes[0].width,
            #                     depth=bin_hashes[0].depth,
            #                     sign_hashes=sign_hashes,
            #                     exact_preds=True,
            #                     sparse=sparse)
            # elif method in ('bound-sketch', 'count-min', 'bound-sketch-unfiltered'):
            #     self.sketch = BoundSketch(data,
            #                             depth=bin_hashes[0].depth,
            #                             width=bin_hashes[0].width,
            #                             bin_hashes=bin_hashes,
            #                             exact_preds=True,
            #                             sparse=sparse)
            # else:
            self.sketch = FastAGMS(data,
                                    depth=bin_hashes[0].depth,
                                    width=bin_hashes[0].width,
                                    sign_hashes=sign_hashes,
                                    bin_hashes=bin_hashes,
                                    exact_preds=True,
                                    sparse=sparse,
                                    sample_sketch=sample_sketch,
                                    method=method)
            
        self.memory = self.sketch.memory
        return
    
    def memory_usage(self):
        return self.sketch.memory_usage()

    def __call__(self, predicates, key, components, **kwargs):
        t0 = perf_counter_ns()
        estimator, sketch_time = self.sketch(predicates, key, components, **kwargs)
        t1 = perf_counter_ns()
        # exclude sketching time from copy overhead
        copy_time = t1 - t0 - sketch_time
        return estimator, sketch_time, copy_time

class JoinLeaf(object):
    def __init__(self, data, bin_hashes=None, sign_hashes=None, method='count-sketch', sparse=False, sample_sketch=None):
        self.columns = set(data.columns)
        self.size = len(data)

        # if method == 'ams':
        #     self.sketch = AMS(data,
        #                     width=bin_hashes[0].width,
        #                     depth=bin_hashes[0].depth,
        #                     sign_hashes=sign_hashes,
        #                     exact_preds=True,
        #                     sparse=sparse)
        # elif method in ('bound-sketch', 'count-min', 'bound-sketch-unfiltered'):
        #     self.sketch = BoundSketch(data,
        #                               width=bin_hashes[0].width,
        #                               depth=bin_hashes[0].depth,
        #                               bin_hashes=bin_hashes,
        #                               exact_preds=True,
        #                               sparse=sparse)
        # else:
        self.sketch = FastAGMS(data,
                                    depth=bin_hashes[0].depth,
                                    width=bin_hashes[0].width,
                                    sign_hashes=sign_hashes,
                                    bin_hashes=bin_hashes,
                                    exact_preds=True,
                                    sparse=sparse,
                                    sample_sketch=sample_sketch,
                                    method=method)

        self.memory = self.sketch.memory
        return
    
    def memory_usage(self):
        return self.sketch.memory_usage()
    
    def __call__(self, predicates, keys, components, **kwargs):
        t0 = perf_counter_ns()
        estimator, sketch_time = self.sketch(predicates, keys, components, **kwargs)
        t1 = perf_counter_ns()
        # exclude sketching time from copy overhead
        copy_time = t1 - t0 - sketch_time
        return estimator, sketch_time, copy_time

class SumNode(object):
    def __init__(self, clusters, features, **kwargs):
        self.method = kwargs['method']
        self.columns = clusters[0].columns
        self.children = []
        self.size = 0
        self.memory = 0
        for i, (c, index) in enumerate(zip(clusters, features)):
            self.children.append(SPN(c, index, **kwargs))
            self.size += len(c)
            self.memory += self.children[i].memory
        return
    
    def memory_usage(self):
        nbytes = sum([child.memory_usage() for child in self.children])
        return nbytes

    def __call__(self, predicates, key, components, **kwargs):
        total_sketch_time = 0
        total_copy_time = 0
        freq = 0
        sketch = None
        for child in self.children:
            sketch_or_prob, sketch_time, copy_time = child(predicates, key, components, **kwargs)
            total_sketch_time += sketch_time
            total_copy_time += copy_time
            if isinstance(sketch_or_prob, (int, float)):
                freq += sketch_or_prob * child.size
            elif sketch is None:
                sketch = sketch_or_prob
            else:
                ### Doesn't work: causes underestimation
                # if self.method in ('bound-sketch', 'factorjoin'):
                #     assert sketch.ndim == 3, sketch.shape
                #     replace_degree = (sketch[:,:,0] < sketch_or_prob[:,:,0])
                #     sketch[:,:,0] = sketch[:,:,0].where(replace_degree, sketch_or_prob[:,:,0])
                #     sketch[:,:,1] += sketch_or_prob[:,:,1]
                # else:
                if sketch.is_sparse and not sketch_or_prob.is_sparse:
                    sketch = sketch.to_dense()
                sketch += sketch_or_prob
        if isinstance(sketch_or_prob, (int, float)):
            return  (freq / self.size), total_sketch_time, total_copy_time
        return sketch, total_sketch_time, total_copy_time
    
class ProductNode(object):
    def __init__(self, components, features, pessimistic=False, **kwargs):
        self.children = []
        self.memory = 0
        for i, (comp, feat) in enumerate(zip(components, features)):
            self.children.append(SPN(comp, feat, pessimistic=pessimistic, cluster_next=True, **kwargs))
            self.memory += self.children[i].memory
        self.pessimistic = pessimistic
        return
    
    def memory_usage(self):
        nbytes = sum([child.memory_usage() for child in self.children])
        return nbytes
    
    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================
    
    def __call__(self, predicates, key, components, **kwargs):
        """
        Evaluate predicates on ProductNode.
        
        Optimized for efficiency:
        - Conjunctive predicates: Single call per child with full predicate
        - Disjunctive predicates: Inclusion-Exclusion with conjunctions as atomic units
        
        Args:
            predicates: sqlglot Expression tree
            key: Key columns for sketch lookup
            components: Component information
        
        Returns:
            (sketch, sketch_time, copy_time) if sketch child exists
            (probability, sketch_time, copy_time) otherwise
        """
        # Identify sketch child (the one matching key)
        sketch_child = None
        sketch_child_idx = None
        if key:
            for i, child in enumerate(self.children):
                if child.columns.intersection(key):
                    assert sketch_child is None, "Only one child may return a sketch in a product node"
                    sketch_child = child
                    sketch_child_idx = i
        
        # Separate children into sketch child and others
        if sketch_child is not None:
            # Evaluate sketch child
            sketch, sketch_time, copy_time = sketch_child(predicates, key, components, **kwargs)
            
            # Evaluate other children to get probabilities
            other_children = [child for i, child in enumerate(self.children) if i != sketch_child_idx]
            other_prob = self._evaluate_children_probability(predicates, other_children, **kwargs)
            
            # Scale sketch by other probabilities
            scaled_sketch = sketch * float(other_prob)
            return scaled_sketch, sketch_time, copy_time
        
        else:
            # No sketch child, evaluate all children for probability
            total_prob = self._evaluate_children_probability(predicates, self.children, **kwargs)
            return total_prob, 0, 0
    
    # ========================================================================
    # PROBABILITY EVALUATION (Handles both AND and OR)
    # ========================================================================
    
    def _evaluate_children_probability(self, predicates: exp.Expression, 
                                       children: List, **kwargs) -> float:
        """
        Evaluate predicates across multiple children to get combined probability.
        
        Handles:
        - Conjunctive predicates (AND) → call each child once with full predicate
        - Disjunctive predicates (OR) → apply Inclusion-Exclusion with conjunctions as units
        
        Args:
            predicates: sqlglot Expression
            children: List of children to evaluate
        
        Returns:
            Combined probability
        """
        if not children or not predicates:
            return 1.0
        
        # Check if predicates contain cross-child OR
        if self._contains_cross_child_or(predicates, children):
            # Use Inclusion-Exclusion (optimized)
            return self._evaluate_with_inclusion_exclusion(predicates, children, **kwargs)
        else:
            # Pure conjunction - evaluate efficiently
            return self._evaluate_conjunction(predicates, children, **kwargs)
    
    def _evaluate_conjunction(self, predicates: exp.Expression, 
                              children: List, **kwargs) -> float:
        """
        Evaluate conjunctive predicates efficiently.
        
        Key optimization: Call each child ONCE with the full conjunction.
        Each child filters to its relevant predicates.
        
        Args:
            predicates: Conjunctive expression (or leaf)
            children: List of children
        
        Returns:
            Product of child probabilities
        
        Example:
            predicates = "x=1 AND y=2 AND z=3"
            children = [Child1{x}, Child2{y}, Child3{z}]
            
            Child1(x=1 AND y=2 AND z=3) → filters to x=1 → 0.3
            Child2(x=1 AND y=2 AND z=3) → filters to y=2 → 0.5
            Child3(x=1 AND y=2 AND z=3) → filters to z=3 → 0.4
            
            Result: 0.3 × 0.5 × 0.4 = 0.06
            
            Total calls: 3 (one per child)
        """
        probs = []
        for child in children:
            # Each child evaluates the full predicate (filters to relevant parts)
            p, _, _ = child(predicates, None, None, **kwargs)
            probs.append(p)
        
        if self.pessimistic:
            return min(probs) if probs else 1.0
        else:
            result = np.prod(probs).item() if probs else 1.0
            return max(0.0, min(1.0, result))
    
    # ========================================================================
    # DISJUNCTION DETECTION AND EVALUATION (OPTIMIZED)
    # ========================================================================
    
    def _contains_cross_child_or(self, predicates: exp.Expression, 
                                  children: List) -> bool:
        """
        Check if predicates contain OR that spans multiple children.
        
        Args:
            predicates: sqlglot Expression
            children: List of children to check
        
        Returns:
            True if OR spans multiple of these children
        """
        # Find all OR nodes
        or_nodes = self._find_or_nodes(predicates)
        
        if not or_nodes:
            return False
        
        # Check each OR node
        for or_node in or_nodes:
            or_columns = self._get_columns_from_expr(or_node)
            
            # Check how many of the given children are involved
            children_involved = [child for child in children 
                               if child.columns.intersection(or_columns)]
            
            if len(children_involved) > 1:
                return True
        
        return False
    
    def _evaluate_with_inclusion_exclusion(self, predicates: exp.Expression,
                                           children: List, **kwargs) -> float:
        """
        Evaluate predicates with Inclusion-Exclusion.
        
        OPTIMIZED: Conjunctions are treated as atomic units.
        Each conjunction results in one call per child (not decomposed).
        
        Args:
            predicates: sqlglot Expression with OR
            children: List of children to evaluate
        
        Returns:
            Probability
        """
        # Extract the OR structure and apply I-E
        return self._apply_inclusion_exclusion_optimized(predicates, children, **kwargs)
    
    def _apply_inclusion_exclusion_optimized(self, expr: exp.Expression,
                                            children: List, **kwargs) -> float:
        """
        Apply Inclusion-Exclusion with optimized conjunction evaluation.
        
        Key optimization: Each term in the I-E formula (which is a conjunction)
        is evaluated as a single unit by calling each child once.
        
        Args:
            expr: Expression (potentially with OR at top level)
            children: List of children
        
        Returns:
            Probability
        
        Example:
            expr = "(x=2 AND y>10) OR (x=1 AND y<10)"
            children = [Child1{x}, Child2{y}]
            
            Terms: [x=2 AND y>10, x=1 AND y<10]
            
            Size 1:
              Term 1: x=2 AND y>10
                Child1(x=2 AND y>10) → 0.3
                Child2(x=2 AND y>10) → 0.5
                Result: 0.15
              
              Term 2: x=1 AND y<10
                Child1(x=1 AND y<10) → 0.2
                Child2(x=1 AND y<10) → 0.4
                Result: 0.08
            
            Size 2:
              Term 3: (x=2 AND y>10) AND (x=1 AND y<10)
                Flatten: x=2 AND y>10 AND x=1 AND y<10
                Child1(x=2 AND y>10 AND x=1 AND y<10) → 0.0
                Child2(x=2 AND y>10 AND x=1 AND y<10) → 0.0
                Result: 0.0
            
            Final: 0.15 + 0.08 - 0.0 = 0.23
            
            Total calls: 6 (3 terms × 2 children)
        """
        # If it's an OR at top level, extract terms
        if isinstance(expr, exp.Or):
            terms = self._extract_or_terms(expr)
        else:
            # Not an OR - evaluate as single conjunction
            return self._evaluate_conjunction(expr, children, **kwargs)
        
        # Apply Inclusion-Exclusion formula
        result_prob = 0.0
        
        # Generate all non-empty subsets
        for size in range(1, len(terms) + 1):
            for subset in combinations(terms, size):
                # Create conjunction of all terms in subset
                conjunction = self._build_conjunction(subset)
                
                # Evaluate conjunction as atomic unit (ONE CALL PER CHILD)
                prob = self._evaluate_conjunction(conjunction, children, **kwargs)
                
                # Apply sign: (-1)^(size+1)
                sign = 1 if size % 2 == 1 else -1
                result_prob += sign * prob
        
        return max(0.0, min(1.0, result_prob))
    
    def _build_conjunction(self, terms: Tuple[exp.Expression]) -> exp.Expression:
        """
        Build a single AND expression from multiple terms.
        
        Args:
            terms: Tuple of expressions to AND together
        
        Returns:
            Single AND expression
        
        Example:
            terms = (x=1, y=2, z=3)
            result = x=1 AND y=2 AND z=3
        """
        if len(terms) == 1:
            return terms[0]
        
        # Flatten any nested ANDs in the terms
        flat_terms = []
        for term in terms:
            if isinstance(term, exp.And):
                # Extract all AND operands
                flat_terms.extend(self._flatten_and(term))
            else:
                flat_terms.append(term)
        
        # Build left-associative AND chain
        result = flat_terms[0]
        for term in flat_terms[1:]:
            result = exp.And(this=result, expression=term)
        
        return result
    
    def _flatten_and(self, and_expr: exp.And) -> List[exp.Expression]:
        """
        Flatten nested AND expression into list of operands.
        
        Args:
            and_expr: AND expression
        
        Returns:
            List of operands
        
        Example:
            (x=1 AND y=2) AND z=3 → [x=1, y=2, z=3]
        """
        operands = []
        
        # Left side
        if isinstance(and_expr.this, exp.And):
            operands.extend(self._flatten_and(and_expr.this))
        else:
            operands.append(and_expr.this)
        
        # Right side
        if isinstance(and_expr.expression, exp.And):
            operands.extend(self._flatten_and(and_expr.expression))
        else:
            operands.append(and_expr.expression)
        
        return operands
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _find_or_nodes(self, expr: exp.Expression) -> List[exp.Expression]:
        """Find all OR nodes in expression tree."""
        key = id(expr)
        cached = _or_nodes_cache.get(key)
        if cached is not None:
            return cached
        result = list(expr.find_all(exp.Or))
        _or_nodes_cache[key] = result
        return result
    
    def _extract_or_terms(self, or_expr: exp.Or) -> List[exp.Expression]:
        """
        Extract all terms from nested OR.
        
        Example: (A OR B) OR C → [A, B, C]
        """
        terms = []
        
        # Left side
        if isinstance(or_expr.this, exp.Or):
            terms.extend(self._extract_or_terms(or_expr.this))
        else:
            terms.append(or_expr.this)
        
        # Right side
        if isinstance(or_expr.expression, exp.Or):
            terms.extend(self._extract_or_terms(or_expr.expression))
        else:
            terms.append(or_expr.expression)
        
        return terms
    
    def _get_columns_from_expr(self, expr: exp.Expression) -> Set[str]:
        """Extract all column names from expression."""
        columns = set()
        for col in expr.find_all(exp.Column):
            columns.add(col.name)
        return columns