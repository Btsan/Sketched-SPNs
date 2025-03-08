from time import perf_counter_ns

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from RDC import rdc
from Sketches import AMS, CountSketch, BoundSketch

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
    flattened = np.concatenate([np.stack(features[col]) for col in features], axis=-1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(flattened)
    if scaled.shape[0] > max_sample_size:
        rng = np.random.default_rng()
        sample = rng.choice(scaled, size=max_sample_size, replace=False)
    else:
        sample = scaled
    if use_kmeans:
        kmeans = KMeans(n_clusters=k).fit(sample)
        labels = kmeans.predict(scaled)[:, None]
    else:
        # default to EM
        if gmm is None:
            gmm = GaussianMixture(n_components=k).fit(sample)
        else:
            assert isinstance(gmm, GaussianMixture), type(gmm)
            gmm = gmm.fit(sample)
        labels = gmm.predict(scaled)[:, None]
    uniques = np.unique(labels, axis=0)
    # assert len(uniques) <= k, f"Unexpectedly many clusters ({len(uniques)}): {uniques}"

    clusters = []
    cluster_features = []
    for value in uniques:
        rows = np.all(labels == value, axis=1)
        if np.any(rows):
            clusters.append(data.iloc[rows])
            cluster_features.append(features.iloc[rows])

    # in case, prevents failure to cluster
    if len(clusters) == 1:
        # print(f"Clustering method only produced 1 unique label: {uniques}")
        # print("Falling back to even splits")
        size = len(data) // k
        clusters = [data.iloc[i:i + size] for i in range(0, len(data), size)]
        cluster_features = [features.iloc[i:i + size] for i in range(0, len(data), size)]
    return clusters, cluster_features, gmm

class SPN(object):
    """Mixed Sum-Product Networks (Molina et al., 2017)
    https://arxiv.org/pdf/1710.03297.pdf
    """
    def __init__(self, data, features, bin_hashes=None, sign_hashes=None, corr_threshold=0.3, min_cluster=1e5, num_clusters=2, cluster_next=False, level=0, verbose=True, sparse=False, keys=None, method='count-sketch', bifocal=0, pessimistic=False, gmm=None, use_kmeans=False):
        if keys is None:
            keys = set()
        self.size = len(data)
        self.sketch_method = method
        # if features is None:
        #     features = data.copy(deep=True)
        #     # index.fillna(-42, inplace=True)
        #     # index = index.map(hash)
        #     features.iloc[:, :] = rdc(features, types)
        # assert data.shape == features.shape, f'data {data.shape} mismatch with features {features.shape}'

        if isinstance(data, pd.Series):
            self.columns = {data.name,}
            self.types = {data.name: data.dtype.type}
            self.bounds = {data.name: (data.min(), data.max())}
        else:
            self.columns = set(data.columns)
            self.types = dict()
            self.bounds = dict()
            for col in self.columns:
                self.types[col] = data[col].dtype.type
                self.bounds[col] = (data[col].min(), data[col].max())
                
        # if verbose: print(f'keys {keys}')
        if len(data.shape) == 1 or len(data.columns) == 1:
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'leaf node {data.name if isinstance(data, pd.Series) else data.columns}{data.shape}', end='')
            level += 1
            self.node = UnivariateLeaf(data,
                                       bin_hashes=bin_hashes, sign_hashes=sign_hashes, level=level, sparse=sparse, method=method, bifocal=bifocal)
            if verbose: print(f'({self.node.memory:,} bytes)')
        elif set(data.columns) == set(keys):
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'join node {tuple(data.columns)}{data.shape}', end='')
            level += 1
            self.node = JoinLeaf(data,
                                 bin_hashes=bin_hashes, sign_hashes=sign_hashes, level=level, sparse=sparse, method=method, bifocal=bifocal)
            if verbose: print(f'({self.node.memory:,} bytes)')
        elif cluster_next:
            self.columns = data.columns
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'sum node {tuple(data.columns)}{data.shape}')
            clusters, indices, gmm = cluster(data, features, k=num_clusters, gmm=gmm, use_kmeans=use_kmeans)
            level += 1
            self.node = SumNode(clusters, indices,
                                bin_hashes=bin_hashes, sign_hashes=sign_hashes, corr_threshold=corr_threshold, min_cluster=min_cluster, num_clusters=num_clusters, level=level, sparse=sparse, keys=keys, method=method, bifocal=bifocal, pessimistic=pessimistic, gmm=gmm, use_kmeans=use_kmeans, verbose=verbose)
        else:
            if data.shape[0] <= max(1, min_cluster):
                # skip rdc calculation
                pairwise_corr = np.eye(data.shape[1])
            else:
                assert data.shape == features.shape
                sample_size = 10000
                pairwise_corr = rdc(rdc_features=features.sample(sample_size) if len(features) > sample_size else features)
            # pairwise_corr = data.corr(method='spearman').abs().values
            # thresh = corr_threshold + (0.25 * level // 5) # relax threshold
            # print(pairwise_corr, thresh)
            min_corr = pairwise_corr.min()
            components, indices = decompose(data, features, pairwise_corr, corr_thresh=corr_threshold, min_cluster=min_cluster, keys=keys)
            if len(components) > 1:
                if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'product node {tuple(data.columns)}{data.shape}(min. corr={min_corr:.2f})')
                level += 1
                self.node = ProductNode(components, indices, 
                                        bin_hashes=bin_hashes, sign_hashes=sign_hashes, 
                                        corr_threshold=corr_threshold, min_cluster=min_cluster, num_clusters=num_clusters, level=level, sparse=sparse, keys=keys, method=method, bifocal=bifocal, pessimistic=pessimistic, use_kmeans=use_kmeans, verbose=verbose)
            else:
                if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'sum node {tuple(data.columns)}{data.shape}(min. corr={min_corr:.2f})')
                clusters, indices, gmm = cluster(data, features, k=num_clusters, gmm=gmm, use_kmeans=use_kmeans)
                level += 1
                self.node = SumNode(clusters, indices,
                                    bin_hashes=bin_hashes, sign_hashes=sign_hashes, corr_threshold=corr_threshold, min_cluster=min_cluster, num_clusters=num_clusters, level=level, sparse=sparse, keys=keys, method=method, bifocal=bifocal, pessimistic=pessimistic, gmm=gmm, use_kmeans=use_kmeans, verbose=verbose)


        self.memory = self.node.memory
        return
    
    def memory_usage(self):
        return self.node.memory_usage()

    def __call__(self, predicates, key, **kwargs):
        col_in_preds = self.columns.intersection(predicates.keys())
        col_in_keys = self.columns.intersection(key.keys())
        if not col_in_preds and not col_in_keys:
            return 1, 0, 0
        """
        elif col_in_preds:
            # check if predicates are out of bounds
            for col in col_in_preds:
                t = self.types[col]
                left, right = self.bounds[col]
                for op, val in predicates[col].items():
                    # print(f"col {col}, op {op}, val {type(val)}{val}, min {self.data[col].min()}, max {self.data[col].max()}")
                    if op == '==':
                        if t(val) < left or t(val) > right:
                            print(f"{col}{op}{val} out of bounds [{left}, {right}]")
                            return 0, 0
                    elif op == '<':
                        if t(val) <= left:
                            print(f"{col}{op}{val} out of bounds [{left}, {right}]")
                            return 0, 0
                    elif op == '>':
                        if t(val) >= right:
                            print(f"{col}{op}{val} out of bounds [{left}, {right}]")
                            return 0, 0
                    elif op == '<=':
                        if t(val) < left:
                            print(f"{col}{op}{val} out of bounds [{left}, {right}]")
                            return 0, 0
                    elif op == '>=':
                        if t(val) > right:
                            print(f"{col}{op}{val} out of bounds [{left}, {right}]")
                            return 0, 0
        """ # todo: this optimization is causing errors (estimator returns 0)

        sketch_or_prob, sketch_time, copy_time = self.node(predicates, key, **kwargs)
        return sketch_or_prob, sketch_time, copy_time
    
    def iterative(self, predicates, key, **kwargs):
        results = dict()

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
                results[node] = node.sketch(predicates, key, **kwargs)
                t1 = perf_counter_ns()
                copy_time += t1 - t0 - results[node][1]
            
            # print()
        return *results[self.node], copy_time

class UnivariateLeaf(object):
    def __init__(self, data, bin_hashes=None, sign_hashes=None, level=0, sparse=False, method='count-sketch', bifocal=0):
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

        if method == 'ams':
            self.sketch = AMS(data,
                            width=bin_hashes[0].num_bins,
                            depth=bin_hashes[0].fn.seeds.shape[0],
                            sign_hashes=sign_hashes,
                            bifocal=bifocal)
        elif method in ('bound-sketch', 'count-min'):
            self.sketch = BoundSketch(data,
                                      depth=bin_hashes[0].fn.seeds.shape[0],
                                      width=bin_hashes[0].num_bins,
                                      bin_hashes=bin_hashes,
                                      bifocal=bifocal)
        else:
            self.sketch = CountSketch(data,
                                      depth=bin_hashes[0].fn.seeds.shape[0],
                                      width=bin_hashes[0].num_bins,
                                      sign_hashes=sign_hashes,
                                      bin_hashes=bin_hashes,
                                      bifocal=bifocal)
        
        self.memory = self.sketch.memory_usage()
        return
    
    def memory_usage(self):
        return self.sketch.memory_usage()

    def __call__(self, predicates, key, **kwargs):
        t0 = perf_counter_ns()
        estimator, sketch_time = self.sketch(predicates, key, **kwargs)
        t1 = perf_counter_ns()
        # exclude sketching time from copy overhead
        copy_time = t1 - t0 - sketch_time
        return estimator, sketch_time, copy_time

class JoinLeaf(object):
    def __init__(self, data, bin_hashes=None, sign_hashes=None, level=0, sparse=False, method='count-sketch', bifocal=0):
        self.columns = set(data.columns)
        self.size = len(data)

        if method == 'ams':
            self.sketch = AMS(data,
                            width=bin_hashes[0].num_bins,
                            depth=bin_hashes[0].fn.seeds.shape[0],
                            sign_hashes=sign_hashes,
                            bifocal=bifocal)
        elif method in ('bound-sketch', 'count-min'):
            self.sketch = BoundSketch(data,
                                      width=bin_hashes[0].num_bins,
                                      depth=bin_hashes[0].fn.seeds.shape[0],
                                      bin_hashes=bin_hashes,
                                      bifocal=bifocal)
        else:
            self.sketch = CountSketch(data,
                                      depth=bin_hashes[0].fn.seeds.shape[0],
                                      width=bin_hashes[0].num_bins,
                                      sign_hashes=sign_hashes,
                                      bin_hashes=bin_hashes,
                                      bifocal=bifocal)

        self.memory = self.sketch.memory_usage()
        return
    
    def memory_usage(self):
        return self.sketch.memory_usage()
    
    def __call__(self, predicates, keys, **kwargs):
        t0 = perf_counter_ns()
        estimator, sketch_time = self.sketch(predicates, keys, **kwargs)
        t1 = perf_counter_ns()
        # exclude sketching time from copy overhead
        copy_time = t1 - t0 - sketch_time
        return estimator, sketch_time, copy_time

class SumNode(object):
    def __init__(self, clusters, features, **kwargs):
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

    def __call__(self, predicates, key, **kwargs):
        total_sketch_time = 0
        total_copy_time = 0
        freq = 0
        sketch = None
        for child in self.children:
            sketch_or_prob, sketch_time, copy_time = child(predicates, key, **kwargs)
            total_sketch_time += sketch_time
            total_copy_time += copy_time
            if isinstance(sketch_or_prob, (int, float)):
                freq += sketch_or_prob * child.size
            elif sketch is None:
                sketch = sketch_or_prob
            else:
                sketch += sketch_or_prob
        if isinstance(sketch_or_prob, (int, float)):
            return  (freq / self.size), total_sketch_time, total_copy_time
        return sketch, total_sketch_time, total_copy_time
            
class ProductNode(object):
    def __init__(self, components, features, pessimistic=False, **kwargs):
        self.children = []
        self.memory = 0
        for i, (comp, feat) in enumerate(zip(components, features)):
            self.children.append(SPN(comp, feat, pessimistic=pessimistic, **kwargs))
            self.memory += self.children[i].memory
        self.pessimistic = pessimistic
        return
    
    def memory_usage(self):
        nbytes = sum([child.memory_usage() for child in self.children])
        return nbytes
    
    def __call__(self, predicates, key, **kwargs):
        probs = [1]
        sketch = None
        sketch_time = 0
        copy_time = 0
        for child in self.children:
            if child.columns.intersection(key):
                assert sketch is None, "Only one child may return a sketch in a product node"
                sketch, sketch_time, copy_time = child(predicates, key, **kwargs)
                if sketch == 0:
                    return sketch, sketch_time, copy_time
            elif min(probs) > 0 and child.columns.intersection(predicates.keys()):
                p, _, _ = child(predicates, key, **kwargs)
                probs.append(p)
        prob = min(probs) if self.pessimistic else np.prod(probs).item()
        assert 0 <= prob <= 1, f"probability {prob} out of bounds"
        if sketch is not None:
            sketch *= prob
            return sketch, sketch_time, copy_time
        return prob, sketch_time, copy_time