from time import perf_counter_ns

import numpy as np
import pandas as pd
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

    adjacency_mat = pairwise_corr >= corr_thresh
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

def cluster(data, features, nbits=1):
    k = 2 ** nbits
    flattened = np.concatenate([np.stack(features[col]) for col in features], axis=-1)
    kmeans = KMeans(n_clusters=k).fit(flattened)
    labels = kmeans.labels_

    clusters = []
    cluster_features = []
    for value in np.unique(labels):
        rows = (labels == value)
        if np.any(rows):
            clusters.append(data.iloc[rows])
            cluster_features.append(features.iloc[rows])

    # in case, prevents failure to cluster
    if len(clusters) == 1:
        size = len(data) // k
        clusters = [data.iloc[i:i + size] for i in range(0, len(data), size)]
        cluster_features = [features.iloc[i:i + size] for i in range(0, len(data), size)]
    return clusters, cluster_features

class SPN(object):
    """Mixed Sum-Product Networks (Molina et al., 2017)
    https://arxiv.org/pdf/1710.03297.pdf
    """
    def __init__(self, data, features, bin_hashes=None, sign_hashes=None, corr_threshold=0.3, min_cluster=1e5, cluster_nbits=1, cluster_next=False, level=0, verbose=True, sparse=False, keys=None, method='count-sketch', bifocal=0, pessimistic=False):
        if keys is None:
            keys = set()
        self.size = len(data)
        self.sketch_method = method
        # if features is None:
        #     features = data.copy(deep=True)
        #     # index.fillna(-42, inplace=True)
        #     # index = index.map(hash)
        #     features.iloc[:, :] = rdc(features, types)
        assert data.shape == features.shape, f'data {data.shape} mismatch with features {features.shape}'

        # if verbose: print(f'keys {keys}')
        if len(data.shape) == 1 or len(data.columns) == 1:
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'leaf node {data.name if type(data) is pd.Series else data.columns}{data.shape}', end='')
            level += 1
            self.node = UnivariateLeaf(data, features,
                                       bin_hashes=bin_hashes, sign_hashes=sign_hashes, level=level, sparse=sparse, method=method, bifocal=bifocal)
            self.columns = [self.node.name]
            print(f'({self.node.memory:,} bytes)')
        elif set(data.columns) == set(keys):
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'join node {tuple(data.columns)}{data.shape}', end='')
            level += 1
            self.node = JoinLeaf(data, features,
                                 bin_hashes=bin_hashes, sign_hashes=sign_hashes, level=level, sparse=sparse, method=method, bifocal=bifocal)
            self.columns = data.columns
            print(f'({self.node.memory:,} bytes)')
        elif cluster_next:
            self.columns = data.columns
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'sum node {tuple(data.columns)}{data.shape}')
            clusters, indices = cluster(data, features, nbits=cluster_nbits)
            level += 1
            self.node = SumNode(clusters, indices,
                                bin_hashes=bin_hashes, sign_hashes=sign_hashes, corr_threshold=corr_threshold, min_cluster=min_cluster, cluster_nbits=cluster_nbits, level=level, sparse=sparse, keys=keys, method=method, bifocal=bifocal, pessimistic=pessimistic)
        else:
            self.columns = data.columns
            pairwise_corr = rdc(rdc_features=features, sample_size=1e4)
            # print(pairwise_corr)
            components, indices = decompose(data, features, pairwise_corr, corr_thresh=corr_threshold, min_cluster=min_cluster, keys=keys)
            if len(components) > 1:
                if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'product node {tuple(data.columns)}{data.shape}')
                level += 1
                self.node = ProductNode(components, indices, 
                                        bin_hashes=bin_hashes, sign_hashes=sign_hashes, 
                                        corr_threshold=corr_threshold, min_cluster=min_cluster, cluster_nbits=cluster_nbits, level=level, sparse=sparse, keys=keys, method=method, bifocal=bifocal, pessimistic=pessimistic)
            else:
                if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'sum node {tuple(data.columns)}{data.shape}')
                clusters, indices = cluster(data, features, nbits=cluster_nbits)
                level += 1
                self.node = SumNode(clusters, indices,
                                    bin_hashes=bin_hashes, sign_hashes=sign_hashes, corr_threshold=corr_threshold, min_cluster=min_cluster, cluster_nbits=cluster_nbits, level=level, sparse=sparse, keys=keys, method=method, bifocal=bifocal, pessimistic=pessimistic)
        self.columns = set(self.columns)
        self.memory = self.node.memory
        return
    
    # todo: rename key to sketch_attrs
    def __call__(self, predicates, key, **kwargs):
        if not self.columns.intersection(predicates.keys()) and not self.columns.intersection(key):
            return 1, 0
        sketch_or_prob, sketch_time = self.node(predicates, key, **kwargs)
        return sketch_or_prob, sketch_time

class UnivariateLeaf(object):
    def __init__(self, data, features, bin_hashes=None, sign_hashes=None, level=0, sparse=False, method='count-sketch', bifocal=0):
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

    def __call__(self, predicates, key, **kwargs):
        estimator, sketch_time = self.sketch(predicates, key, **kwargs)
        return estimator, sketch_time

class JoinLeaf(object):
    def __init__(self, data, features, bin_hashes=None, sign_hashes=None, level=0, sparse=False, method='count-sketch', bifocal=0):
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
    
    def __call__(self, predicates, keys, **kwargs):
        estimator, sketch_time = self.sketch(predicates, keys, **kwargs)
        return estimator, sketch_time

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

    def __call__(self, predicates, key, **kwargs):
        freq = 0
        sketch = None
        sketch_times = []
        for child in self.children:
            sketch_or_prob, sketch_time = child(predicates, key, **kwargs)
            sketch_times.append(sketch_time)
            if isinstance(sketch_or_prob, (int, float)):
                freq += sketch_or_prob * child.size
            elif sketch is None:
                sketch = sketch_or_prob
            else:
                sketch += sketch_or_prob
        if isinstance(sketch_or_prob, (int, float)):
            return  (freq / self.size), sum(sketch_times)
        return sketch, sum(sketch_times)
            
class ProductNode(object):
    def __init__(self, components, features, pessimistic=False, **kwargs):
        self.children = []
        self.memory = 0
        for i, (comp, feat) in enumerate(zip(components, features)):
            self.children.append(SPN(comp, feat, pessimistic=pessimistic, **kwargs))
            self.memory += self.children[i].memory
        self.pessimistic = pessimistic
        return
    
    def __call__(self, predicates, key, **kwargs):
        probs = [1]
        sketch = None
        sketch_time = 0
        for child in self.children:
            if child.columns.intersection(key):
                assert sketch is None, "Only one child may return a sketch in a product node"
                sketch, sketch_time = child(predicates, key, **kwargs)
            elif child.columns.intersection(predicates.keys()):
                p, _ = child(predicates, key, **kwargs)
                probs.append(p)
        prob = min(probs) if self.pessimistic else np.prod(probs).item()
        assert 0 <= prob <= 1, f"probability {prob} out of bounds"
        if sketch is not None:
            sketch *= prob
            return sketch, sketch_time
        return prob, sketch_time