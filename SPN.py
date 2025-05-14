from time import perf_counter_ns

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from RDC import rdc
from Sketches import AMS, FastAGMS, BoundSketch, ExactSelectivity, CountSketch, CountMin

from scipy.stats import chi2_contingency

def cramers_v_matrix(df):
    """
    Computes the pairwise Cramer's V correlation matrix for categorical variables in a pandas DataFrame.
    
    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing categorical variables.

    Returns:
        pd.DataFrame: A DataFrame containing the pairwise Cramer's V values.
    """
    cols = df.columns
    n = len(cols)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1  # Perfect correlation with itself
            else:
                # Check for all unique values in either column
                if df[cols[i]].nunique() >= 0.99 * len(df[cols[i]]) or df[cols[j]].nunique() >= 0.99 * len(df[cols[j]]):
                    matrix[i, j] = 0  # No correlation possible
                else:
                    # Build the contingency table
                    contingency_table = pd.crosstab(df[cols[i]], df[cols[j]])
                    # Compute the chi-squared statistic
                    chi2, _, _, _ = chi2_contingency(contingency_table)
                    # Compute Cramer's V
                    total = contingency_table.sum().sum()
                    min_dim = min(contingency_table.shape) - 1
                    
                    # Check for zero divisor
                    if total == 0 or min_dim == 0 or chi2 < 0:
                        matrix[i, j] = 0
                    else:
                        value = chi2 / (total * min_dim)
                        matrix[i, j] = np.sqrt(value) if value > 0 else 0

    return matrix

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
    # t0 = perf_counter_ns()
    # feat = np.concatenate([np.stack(features[col]) for col in features], axis=-1)
    # feat = np.column_stack([np.array(features[col].to_list()) for col in features])
    feat = features.values
    # t1 = perf_counter_ns()
    # print(f"Feature ({features.shape} -> {feat.shape}) extraction time: {(t1 - t0) / 1e6:.2f} ms")

    # t0 = perf_counter_ns()
    if use_kmeans:
        # kmeans = KMeans(n_clusters=k).fit(sample)
        # labels = kmeans.predict(scaled)[:, None]
        labels = KMeans(n_clusters=k).fit_predict(feat)[:, None]
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
            gmm = GaussianMixture(n_components=k, covariance_type='diag').fit(sample)
        else:
            assert isinstance(gmm, GaussianMixture), f"Invalid GMM type: {type(gmm)}"
            gmm = gmm.fit(sample)
            # gmm.fit(scaled)
        labels = gmm.predict(feat)[:, None]
    # t1 = perf_counter_ns()
    # print(f"Clustering time: {(t1 - t0) / 1e6:.2f} ms")

    # t0 = perf_counter_ns()
    clusters = []
    cluster_features = []
    # for value in uniques:
    #     rows = np.all(labels == value, axis=1)
    #     if np.any(rows):
    #         clusters.append(data.iloc[rows])
    #         cluster_features.append(features.iloc[rows])
    # data.loc[:, '_cluster'] = labels.flatten()
    # features.loc[:, '_cluster'] = labels.flatten()
    for cluster, group in data.groupby(labels.flatten()):
        if not group.empty:
            clusters.append(group)
    
    for cluster, group in features.groupby(labels.flatten()):
        if not group.empty:
            cluster_features.append(group)
        # cluster_features.append(features.iloc[group.index])

    # t1 = perf_counter_ns()
    # print(f"Cluster extraction time: {(t1 - t0) / 1e6:.2f} ms")

    # in case, prevents failure to cluster
    if len(clusters) == 1:
        # fallback to random projection LSH
        clusters = []
        cluster_features = []

        # hashes
        rng = np.random.default_rng()
        normals = np.random.rand(feat.shape[1], 1) - 0.5
        rand_proj = np.matmul(feat, normals)
        labels = np.sin(rand_proj).flatten() > 0

        # group by LSH labels
        for cluster, group in data.groupby(labels):
            if not group.empty:
                clusters.append(group)

        for cluster, group in features.groupby(labels):
            if not group.empty:
                cluster_features.append(group)
    
    if len(clusters) == 1:
        # final fallback to even split
        clusters = []
        cluster_features = []

        labels = np.arange(feat.shape[0]) % 2
        for cluster, group in data.groupby(labels):
            if not group.empty:
                clusters.append(group)

        for cluster, group in features.groupby(labels):
            if not group.empty:
                cluster_features.append(group)

    assert len(clusters) > 1, \
        f'cluster labels {labels}'
    return clusters, cluster_features, gmm

class SPN(object):
    """Mixed Sum-Product Networks (Molina et al., 2017)
    https://arxiv.org/pdf/1710.03297.pdf
    """
    def __init__(self, data, features, bin_hashes=None, sign_hashes=None, corr_threshold=0.3, min_cluster=1e5, num_clusters=2, cluster_next=False, level=0, verbose=True, sparse=False, keys=None, method='count-sketch', bifocal=0, pessimistic=False, gmm=None, use_kmeans=False, exact_preds=False, meta_types=None, pearsons=False, intervals=None):
        self.exact_preds = exact_preds
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
            self.dtypes = {data.name: data.dtype}
            # self.bounds = {data.name: (data.min(), data.max())}
        else:
            self.columns = set(data.columns)
            self.dtypes = dict()
            # self.bounds = dict()
            for col in self.columns:
                self.dtypes[col] = data[col].dtype
                # self.bounds[col] = (data[col].min(), data[col].max())
                
        # if verbose: print(f'keys {keys}')
        if len(data.shape) == 1 or len(data.columns) == 1:
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'leaf node {data.name if isinstance(data, pd.Series) else data.columns}', end='')
            level += 1
            self.node = UnivariateLeaf(data,
                                       bin_hashes=bin_hashes, sign_hashes=sign_hashes, level=level, sparse=sparse, method=method, bifocal=bifocal, exact_preds=exact_preds, keys=keys, intervals=intervals)
            if verbose: print(f'({self.node.memory:,} bytes)')
        elif set(data.columns) == set(keys):
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'join node {tuple(data.columns)}', end='')
            level += 1
            self.node = JoinLeaf(data,
                                 bin_hashes=bin_hashes, sign_hashes=sign_hashes, level=level, sparse=sparse, method=method, bifocal=bifocal, exact_preds=exact_preds,)
            if verbose: print(f'({self.node.memory:,} bytes)')
        elif cluster_next:
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'sum node {tuple(data.columns)}')
            clusters, indices, gmm = cluster(data, features, k=num_clusters, gmm=gmm, use_kmeans=use_kmeans)
            level += 1
            self.node = SumNode(clusters, indices,
                                bin_hashes=bin_hashes, sign_hashes=sign_hashes, corr_threshold=corr_threshold, min_cluster=min_cluster, num_clusters=num_clusters, level=level, sparse=sparse, keys=keys, method=method, bifocal=bifocal, pessimistic=pessimistic, gmm=gmm, use_kmeans=use_kmeans, verbose=verbose, exact_preds=exact_preds,
                                meta_types=meta_types, pearsons=pearsons, intervals=intervals)
        else:
            contained_types = {meta_types[col] for col in data.columns} if meta_types is not None else None
            corr_type = 'corr'
            sample_size = 5000
            if len(data) <= max(1, min_cluster):
                # skip rdc calculation
                pairwise_corr = np.eye(data.shape[1])
            # elif pearsons:
            #     corr_type = 'Pearsons'
            #     pairwise_corr = (data.sample(sample_size) if len(data) > sample_size else data).corr(method='pearson').abs().values
            # elif meta_types is not None and contained_types == {'CONTINUOUS'}:
            #     corr_type = 'Spearmans'
            #     pairwise_corr = (data.sample(sample_size) if len(data) > sample_size else data).corr(method='spearman').abs().values
            # elif meta_types is not None and contained_types == {'DISCRETE'}:
            #     corr_type = 'cramers_v'
            #     pairwise_corr = abs(cramers_v_matrix(data.sample(sample_size) if len(data) > sample_size else data))
            else:
                corr_type = 'RDC'
                assert len(data) == len(features)
                pairwise_corr = rdc(data=data,
                                    rdc_features=features.sample(sample_size) if len(features) > sample_size else features,
                                    meta_types=meta_types)
            # pairwise_corr = data.corr(method='spearman').abs().values
            # thresh = corr_threshold + (0.25 * level // 5) # relax threshold
            # print(pairwise_corr, thresh)
            min_corr = pairwise_corr.min()
            components, indices = decompose(data, features, pairwise_corr, corr_thresh=corr_threshold, min_cluster=min_cluster, keys=keys)
            if len(components) > 1:
                if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'product node {tuple(data.columns)}{data.shape}(min. {corr_type}={min_corr:.2e})')
                level += 1
                self.node = ProductNode(components, indices, 
                                        bin_hashes=bin_hashes, sign_hashes=sign_hashes,
                                        corr_threshold=corr_threshold, min_cluster=min_cluster, num_clusters=num_clusters, level=level, sparse=sparse, keys=keys, method=method, bifocal=bifocal, pessimistic=pessimistic, use_kmeans=use_kmeans, verbose=verbose, exact_preds=exact_preds,
                                        meta_types=meta_types, pearsons=pearsons, intervals=intervals)
            else:
                if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'sum node {tuple(data.columns)}{data.shape}(min. {corr_type}={min_corr:.2e})')
                clusters, indices, gmm = cluster(data, features, k=num_clusters, gmm=gmm, use_kmeans=use_kmeans)
                level += 1
                self.node = SumNode(clusters, indices,
                                    bin_hashes=bin_hashes, sign_hashes=sign_hashes, corr_threshold=corr_threshold, min_cluster=min_cluster, num_clusters=num_clusters, level=level, sparse=sparse, keys=keys, method=method, bifocal=bifocal, pessimistic=pessimistic, gmm=gmm, use_kmeans=use_kmeans, verbose=verbose, exact_preds=exact_preds,
                                    meta_types=meta_types, pearsons=pearsons, intervals=intervals)

        self.memory = self.node.memory

        self.saved = dict()
        return
    
    def memory_usage(self):
        return self.node.memory_usage()

    def __call__(self, predicates, key, components, _root=True, **kwargs):
        if _root and not self.exact_preds:
            # cast predicate values to the correct type
            for col in predicates.keys():
                if col in self.dtypes:
                    print(f"cast {col} to {self.dtypes[col]}")
                    # if type is a datetime, convert to nanoseconds since last epoch
                    use_nanoseconds = pd.api.types.is_datetime64_any_dtype(self.dtypes[col])
                    for op, val in predicates[col].items():
                        print(f"\tcast {col} {op} {val} to {self.dtypes[col]}")
                        if use_nanoseconds:
                            predicates[col][op] = pd.to_datetime(val).value
                        else:
                            predicates[col][op] = self.dtypes[col].type(val)

        col_in_preds = self.columns.intersection(predicates.keys())
        col_in_keys = self.columns.intersection(key.keys())
        sketch_id = frozenset(key.items()).union(components.items()).union(kwargs.items())
        if not col_in_preds and not col_in_keys:
            return 1, 0, 0
        elif _root and not col_in_preds and sketch_id in self.saved:
            return self.saved[sketch_id] if isinstance(self.saved[sketch_id], (int, float)) else self.saved[sketch_id].to_dense(), 0, 0
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
        sketch_or_prob, sketch_time, copy_time = self.node(predicates, key, components, _root=False, **kwargs)
        if _root and not col_in_preds:
            # save sketch for later use
            if isinstance(sketch_or_prob, (int, float)):
                self.saved[sketch_id] = sketch_or_prob
            else:
                self.saved[sketch_id] = sketch_or_prob.to_sparse()
        return sketch_or_prob, sketch_time, copy_time
    
    def iterative(self, predicates, key, components, **kwargs):
        # cast predicate values to the correct type
        if not self.exact_preds:
            for col in predicates.keys():
                if col in self.dtypes:
                    print(f"cast {col} to {self.dtypes[col]}")
                    # if type is a datetime, convert to nanoseconds since last epoch
                    use_nanoseconds = pd.api.types.is_datetime64_any_dtype(self.dtypes[col])
                    for op, val in predicates[col].items():
                        print(f"\tcast {col} {op} {val} to {self.dtypes[col]}")
                        if use_nanoseconds:
                            predicates[col][op] = pd.to_datetime(val).value
                        else:
                            predicates[col][op] = self.dtypes[col].type(val)

        col_in_preds = self.columns.intersection(predicates.keys())
        sketch_id = frozenset(key.items()).union(components.items()).union(kwargs.items())
        # print(col_in_preds, sketch_id, self.saved)
        # print(type(col_in_preds), type(sketch_id), type(self.saved))
        if not col_in_preds and sketch_id in self.saved:
            return self.saved[sketch_id] if isinstance(self.saved[sketch_id], (int, float)) else self.saved[sketch_id].to_dense(), 0, 0

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

        if not col_in_preds:
            # save sketch for later use
            if isinstance(results[self.node][0], (int, float)):
                self.saved[sketch_id] = results[self.node][0]
            else:
                self.saved[sketch_id] = results[self.node][0].to_sparse()
        return *results[self.node], copy_time

class UnivariateLeaf(object):
    def __init__(self, data, bin_hashes=None, sign_hashes=None, level=0, sparse=False, method='count-sketch', bifocal=0, exact_preds=False, keys=None, intervals=None):
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
            is_numeric = pd.api.types.is_numeric_dtype(data)
            is_datetime = pd.api.types.is_datetime64_any_dtype(data)
            # if using exact selectivity or type is not supported
            if exact_preds or not (is_numeric or is_datetime):
                print(f"Exact selectivity for {self.name} ({data.shape}) {data.dtype} {data.min()}-{data.max()})")
                self.sketch = ExactSelectivity(data)
            else:
                self.sketch = CountSketch(data,
                                          depth=bin_hashes[0].depth,
                                          width=max(1000, bin_hashes[0].width // 100),
                                          sign_hash=sign_hashes[0],
                                          bin_hash=bin_hashes[0],
                                          intervals=intervals[self.name] if intervals is not None and self.name in intervals else None,)
                # self.sketch = CountMin(data,
                #                       depth=bin_hashes[0].depth,
                #                       width=bin_hashes[0].width // 100,
                #                       bin_hash=bin_hashes[0],
                #                       intervals=intervals[self.name] if intervals is not None and self.name in intervals else None,)
        else:
            if method == 'ams':
                self.sketch = AMS(data,
                                width=bin_hashes[0].width,
                                depth=bin_hashes[0].depth,
                                sign_hashes=sign_hashes,
                                exact_preds=exact_preds)
            elif method in ('bound-sketch', 'count-min'):
                self.sketch = BoundSketch(data,
                                        depth=bin_hashes[0].depth,
                                        width=bin_hashes[0].width,
                                        bin_hashes=bin_hashes,
                                        exact_preds=exact_preds)
            else:
                self.sketch = FastAGMS(data,
                                        depth=bin_hashes[0].depth,
                                        width=bin_hashes[0].width,
                                        sign_hashes=sign_hashes,
                                        bin_hashes=bin_hashes,
                                        exact_preds=exact_preds)
            
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
    def __init__(self, data, bin_hashes=None, sign_hashes=None, level=0, sparse=False, method='count-sketch', bifocal=0, exact_preds=False):
        self.columns = set(data.columns)
        self.size = len(data)

        if method == 'ams':
            self.sketch = AMS(data,
                            width=bin_hashes[0].width,
                            depth=bin_hashes[0].depth,
                            sign_hashes=sign_hashes,
                            bifocal=bifocal,
                            exact_preds=exact_preds)
        elif method in ('bound-sketch', 'count-min'):
            self.sketch = BoundSketch(data,
                                      width=bin_hashes[0].width,
                                      depth=bin_hashes[0].depth,
                                      bin_hashes=bin_hashes,
                                      bifocal=bifocal,
                                      exact_preds=exact_preds)
        else:
            self.sketch = FastAGMS(data,
                                      depth=bin_hashes[0].depth,
                                      width=bin_hashes[0].width,
                                      sign_hashes=sign_hashes,
                                      bin_hashes=bin_hashes,
                                      bifocal=bifocal,
                                      exact_preds=exact_preds)

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
    
    def __call__(self, predicates, key, components, **kwargs):
        probs = [1]
        sketch = None
        sketch_time = 0
        copy_time = 0
        for child in self.children:
            if child.columns.intersection(key):
                assert sketch is None, "Only one child may return a sketch in a product node"
                sketch, sketch_time, copy_time = child(predicates, key, components, **kwargs)
                # if (isinstance(sketch, (int, float)) and sketch == 0) or (sketch == 0).all():
                #     return sketch, sketch_time, copy_time
            elif min(probs) > 0 and child.columns.intersection(predicates.keys()):
                p, _, _ = child(predicates, key, components, **kwargs)
                # if p == 0:
                #     return p, sketch_time, copy_time
                probs.append(p)
        prob = min(probs) if self.pessimistic else np.prod(probs).item()
        assert 0 <= prob <= 1, f"probability {prob} out of bounds"
        if sketch is not None:
            sketch *= prob
            return sketch, sketch_time, copy_time
        return prob, sketch_time, copy_time