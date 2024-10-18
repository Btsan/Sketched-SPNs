from copy import deepcopy
from itertools import product, combinations
from collections import defaultdict
from time import perf_counter_ns

import numpy as np
import torch
from torch.fft import fft, ifft
from scipy.stats import rankdata

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# KWiseHash package by Heddes et al. (SIGMOD 2024)
# https://github.com/mikeheddes/fast-multi-join-sketch - Jul 2024
from kwisehash import KWiseHash
class SignHash(object):
    fn: KWiseHash

    def __init__(self, *size, k=2) -> None:
        self.fn = KWiseHash(*size, k=k)

    def __call__(self, items: torch.Tensor) -> torch.Tensor:
        return self.fn.sign(torch.as_tensor(items))
    
class BinHash(object):
    fn: KWiseHash

    def __init__(self, *size, bins, k=2) -> None:
        self.num_bins = bins
        self.fn = KWiseHash(*size, k=k)

    def __call__(self, items: torch.Tensor) -> torch.Tensor:
        return self.fn.bin(torch.as_tensor(items), self.num_bins)

def get_hashes(depth, width, k=4):
    binhashes = BinHash(depth, bins=width, k=k)
    signhashes = SignHash(depth, k=k)
    return binhashes, signhashes

def ecdf(X):
    # Empirical cumulative distribution function
    # for data X (one dimensional, if not it is linearized first)
    # https://github.com/SPFlow/SPFlow/blob/master/src/spn/algorithms/splitting/RDC.py - Feb 2024
    mv_ids = np.isnan(X)

    N = X.shape[0]
    X = X[~mv_ids]
    R = rankdata(X, method="max") / len(X)
    X_r = np.zeros(N)
    X_r[~mv_ids] = R
    return X_r

def decompose(data, index, threshold=0.3, min_cluster=1e5, terminate=False, keys={}):
    if data.shape[0] <= max(1, min_cluster) or terminate:
        # completely decompose data regardless
        if keys & set(data.columns):
            k = list(keys)
            components = [data[col] for col in data.columns if col not in keys] + [data[k]]
            indices = [index[col] for col in index.columns if col not in keys] + [index[k]]
        else:
            components = [data[col] for col in data.columns]
            indices = [index[col] for col in index.columns]
        return components, indices

    N, D = data.shape
    hashes = index.to_numpy() # % 2
    hashes = np.apply_along_axis(ecdf, 0, hashes)
    correlation = np.corrcoef(hashes, rowvar=False) # -1 to 1

    ungrouped = [i for i in range(0, D)]
    groups = []
    g = []
    if keys & set(data.columns):
        for i in range(0, D):
            if data.columns[i] in keys and i in ungrouped:
                g.append(i)
                ungrouped.remove(i)
                for j in range(0, D):
                    if i != j and abs(correlation[i][j]) >= threshold and j in ungrouped:
                        g.append(j)
                        ungrouped.remove(j)
        groups.append([data.columns[x] for x in g])
        g.clear()
    for i in range(0, D):
        if i not in ungrouped:
            continue # skip
        g.append(i)
        ungrouped.remove(i)
        for j in range(i+1, D):
            if abs(correlation[i][j]) >= threshold and j in ungrouped:
                g.append(j)
                ungrouped.remove(j)
        groups.append([data.columns[x] for x in g])
        g.clear()
    components = [data[g] for g in groups]
    indices = [index[g] for g in groups]
    return components, indices

def cluster(data, index, nbits=4):
    # random projection lsh
    hashes = index.to_numpy()
    normals = np.random.rand(nbits, len(data.columns)) - 0.5
    rand_proj = np.matmul(hashes, normals.T) # [n, nbits]
    rand_proj = np.sin(rand_proj) > 0 # boolean ndarray

    clusters = []
    indices = []
    for combo in product((True, False), repeat=nbits):
        rows = np.all(rand_proj == combo, axis=1)
        if np.any(rows):
            clusters.append(data.iloc[rows])
            indices.append(index.iloc[rows])

    # in case, prevents failure to cluster
    if len(clusters) == 1:
        size = len(data) // (2 ** nbits)
        clusters = [data.iloc[i:i + size] for i in range(0, len(data), size)]
        indices = [index.iloc[i:i + size] for i in range(0, len(data), size)]
    return clusters, indices

class SPN(object):
    """Mixed Sum-Product Networks (Molina et al., 2017)
    https://arxiv.org/pdf/1710.03297.pdf
    """
    def __init__(self, data, index=None, bin_hashes=None, sign_hashes=None, corr_threshold=0.3, min_cluster=1e5, cluster_nbits=1, cluster_next=False, level=0, verbose=True, sparse=False, keys=None):
        if keys is None:
            keys = set()
        self.size = len(data)

        if index is None:
            index = data.copy(deep=True)
            index.fillna(-42, inplace=True)
            index = index.applymap(hash)
        assert data.shape == index.shape, f'data {data.shape} mismatch with hashed index {index.shape}'

        # if verbose: print(f'keys {keys}')
        if len(data.shape) == 1 or len(data.columns) == 1:
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'leaf node {data.name if type(data) is pd.Series else data.columns}{data.shape}', end='')
            level += 1
            self.node = UnivariateLeaf(data, index, bin_hashes, sign_hashes, level=level, sparse=sparse)
            self.columns = [self.node.name]
            print(f'({self.node.memory:,} bytes)')
        elif set(data.columns) == set(keys):
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'join node {tuple(data.columns)}{data.shape}', end='')
            level += 1
            self.node = JoinLeaf(data, index, bin_hashes, sign_hashes, level=level, sparse=sparse)
            self.columns = data.columns
            print(f'({self.node.memory:,} bytes)')
        elif cluster_next:
            self.columns = data.columns
            if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'sum node {tuple(data.columns)}{data.shape}')
            clusters, indices = cluster(data, index, nbits=cluster_nbits)
            level += 1
            self.node = SumNode(clusters, indices, bin_hashes, sign_hashes, corr_threshold=corr_threshold, min_cluster=min_cluster, cluster_nbits=cluster_nbits, level=level, sparse=sparse, keys=keys)
        else:
            self.columns = data.columns
            components, indices = decompose(data, index, threshold=corr_threshold, min_cluster=min_cluster, keys=keys)
            if len(components) > 1:
                if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'product node {tuple(data.columns)}{data.shape}')
                level += 1
                self.node = ProductNode(components, indices, bin_hashes, sign_hashes, corr_threshold=corr_threshold, min_cluster=min_cluster, cluster_nbits=cluster_nbits, level=level, sparse=sparse, keys=keys)
            else:
                if verbose: print('|   ' * max(0, level-1) + '\\-- ' * min(1, level) + f'sum node {tuple(data.columns)}{data.shape}')
                clusters, indices = cluster(data, index, nbits=cluster_nbits)
                level += 1
                self.node = SumNode(clusters, indices, bin_hashes, sign_hashes, corr_threshold=corr_threshold, min_cluster=min_cluster, cluster_nbits=cluster_nbits, level=level, sparse=sparse, keys=keys)
        self.memory = self.node.memory
        return
    
    def sketch(self, predicates, key, all=False, level=0, num_joins=defaultdict(lambda: 1), sparse=False, pessimistic=False):
        return self.node.sketch(predicates, key, all, level=level, num_joins=num_joins, sparse=sparse, pessimistic=pessimistic)

class UnivariateLeaf(object):
    def __init__(self, data, index, bin_hashes=None, sign_hashes=None, level=0, sparse=False):
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
        counts = data.groupby(data).size()
        self.frequencies = counts

        self.bin_hashes = bin_hashes
        self.sign_hashes = sign_hashes
        depth = bin_hashes.fn.seeds.shape[0]
        width = bin_hashes.num_bins 
        
        sketch = torch.zeros((depth, 2 * width), dtype=torch.float32)
        x = index.values
        if x.ndim > 1: x = x.squeeze(1)
        negatives = sign_hashes(x) < 0
        bins = bin_hashes(x).abs()
        assert bins.min() >= 0 and bins.max() < width, f'bins has min {bins.min()} and max {bins.max()}'
        bins += (width * negatives)
        sketch.scatter_reduce_(-1, bins, torch.ones(1).expand(bins.shape), reduce='sum')
        
        self.memory = self.frequencies.memory_usage()
        if sparse:
            self.pos = sketch[:, :width].to_sparse()
            self.neg = sketch[:, width:].to_sparse()

            # memory of a sparse COO tensor is at least
            # (ndim * 8 + <size of element type in bytes>) * nse bytes
            self.memory += (2 * 8 + 4) * (self.pos.values().numel() + self.neg.values().numel())
        else:
            self.pos = sketch[:, :width].detach()
            self.neg = sketch[:, width:].detach()
            self.memory += (2 * depth * width) * 4
        return

    def sketch(self, predicates, keys, all=False, level=0, num_joins=defaultdict(lambda: 1), sparse=False, pessimistic=False):
        if self.name in keys:
            if num_joins[self.name] % 2 == 0:
                s = (self.pos + self.neg)
                # print(f'univariate leaf: returning count-min sketch of {self.name}')
            else:
                s = (self.pos - self.neg)
                # print(f'univariate leaf: returning fast-agms sketch of {self.name}')
            if sparse:
                return s.to_sparse()
            else:
                return s.to_dense()
            
        elif self.name in predicates:
            preds = predicates[self.name]
            if '=' in preds:
                count = self.frequencies[self.frequencies.index.values == preds['=']].sum()
            elif '=' in preds:
                count = self.frequencies[self.frequencies.index.values != preds['=']].sum()
            else:
                index = np.array([True] * len(self.frequencies.index))
                if '>' in preds:
                    index *= self.frequencies.index > preds['>']
                if '>=' in preds:
                    index *= self.frequencies.index >= preds['>=']
                if '<' in preds:
                    index *= self.frequencies.index < preds['<']
                if '<=' in preds:
                    index *= self.frequencies.index <= preds['<=']
                count = self.frequencies.iloc[index].sum()
            prob = count / self.size
            return prob
        return 1

class JoinLeaf(object):
    def __init__(self, data, index, bin_hashes=None, sign_hashes=None, level=0, sparse=False):
        self.columns = data.columns
        self.size = len(data)

        self.bin_hashes = bin_hashes
        self.sign_hashes = sign_hashes
        depth = bin_hashes.fn.seeds.shape[0]
        width = bin_hashes.num_bins 

        x = index.values
        bins = [bin_hashes(col).abs() for col in x.T] # [d, N] x C
        bins = torch.stack(bins, dim=2)
        assert bins.min() >= 0 and bins.max() <= width, f'bins has min {bins.min()} and max {bins.max()}'

        signs = [sign_hashes(col) for col in x.T] # [d, N] x C
        signs = torch.stack(signs, dim=2).type(torch.float32)

        self.memory = 0

        self.pos = {}
        self.neg = {}
        for i in range(len(data.columns)):
            for col_idx in combinations(range(len(data.columns)), i+1):
                combo = tuple(data.columns[idx] for idx in col_idx)

                combined_signs = signs[:, :, col_idx].prod(dim=-1)
                combined_bins = bins[:, :, col_idx].sum(dim=-1) % width
                assert combined_signs.min() >= -1 and combined_signs.max() <= 1, f'signs are [{combined_signs.min()},{combined_signs.max()}]'

                sketches = torch.zeros((depth, 2 * width), dtype=torch.float32)
                combined_bins = combined_bins + (width * (combined_signs < 0))

                assert combined_bins.shape == (depth, len(data)), f'shape of bin hashes is incorrect {combined_bins.shape} (exepcted {depth},{len(data)} hashes of {combo})'
                assert combined_signs.shape == (depth, len(data)), f'shape of sign hashes is incorrect {combined_signs.shape} (exepcted {depth},{len(data)} hashes of {combo})'
                assert combined_bins.shape == combined_signs.shape, f'shape of hashes do not match {combined_bins.shape} is not {combined_signs.shape}'

                sketches.scatter_reduce_(-1, combined_bins, torch.ones(1).expand(combined_bins.shape), reduce='sum')
                if sparse:
                    self.pos[combo] = sketches[:, :width].to_sparse()
                    self.neg[combo] = sketches[:, width:].to_sparse()
                    self.memory += (2 * 8 + 4) * (self.pos[combo].values().numel() + self.neg[combo].values().numel())
                else:
                    self.pos[combo] = sketches[:, :width].detach()
                    self.neg[combo] = sketches[:, width:].detach()
                    self.memory += (2 * depth * width) * 4

        self.frequencies = {}
        for col in data.columns:
            self.frequencies[col] = data.groupby(col).size()
            self.memory += self.frequencies[col].memory_usage()
        return
    
    def sketch(self, predicates, keys, all=False, level=0, num_joins=defaultdict(lambda: 1), sparse=False, pessimistic=False):
        if set(self.columns) & set(keys):
            s = None
            included_keys = [key for key in keys if num_joins[key] % 2 != 0]

            if len(included_keys) == 0:
                for combo in self.pos:
                    if set(combo) == set(keys):
                        s = (self.pos[combo] + self.neg[combo])
                        # print(f'join leaf: returning convoluted count-min of {keys} out of {keys}')
                        break
            else:
                for combo in self.pos:
                    if set(combo) == set(included_keys):
                        s = (self.pos[combo] - self.neg[combo])
                        # print(f'join leaf: returning convoluted fast-agms of {included_keys} out of {keys}')
                        break
            
            assert s is not None, f'Missing sketch for join key(s) {keys} in {list(self.sketches.keys())}'
            if sparse:
                return s.to_sparse()
            else:
                return s.to_dense()
            
        # otherwise, check if columns have any predicates
        prob = 1
        for col in self.columns:
            if col in predicates:
                preds = predicates[col]
                freq = self.frequencies[col]
                if '=' in preds:
                    count = freq[freq.index.values == preds['=']].sum()
                elif '=' in preds:
                    count = self.frequencies[self.frequencies.index.values != preds['=']].sum()
                else:
                    index = np.array([True] * len(freq.index))
                    if '>' in preds:
                        index *= freq.index > preds['>']
                    if '>=' in preds:
                        index *= freq.index >= preds['>=']
                    if '<' in preds:
                        index *= freq.index < preds['<']
                    if '<=' in preds:
                        index *= freq.index <= preds['<=']
                    count = freq.iloc[index].sum()
                prob *= count / self.size
        return prob

class SumNode(object):
    def __init__(self, clusters, indices, bin_hashes=None, sign_hashes=None, corr_threshold=0.3, min_cluster=1e5, cluster_nbits=4, level=0, sparse=False, keys={}):
        self.columns = clusters[0].columns
        self.children = []
        self.size = 0
        self.memory = 0
        for i, (c, index) in enumerate(zip(clusters, indices)):
            self.children.append(SPN(c, index, bin_hashes, sign_hashes, corr_threshold=corr_threshold, min_cluster=min_cluster, cluster_nbits=cluster_nbits, level=level, sparse=sparse, keys=keys))
            self.size += len(c)
            self.memory += self.children[i].memory
        return

    def sketch(self, predicates, keys, all=False, level=0, num_joins=defaultdict(lambda: 1), sparse=False, pessimistic=False):
        p = 0
        agg = None
        for child in self.children:
            s = child.sketch(predicates, keys, all=all, level=level+1, num_joins=num_joins, sparse=sparse, pessimistic=pessimistic)
            if not hasattr(s, "__len__"):
                p += s * child.size
            elif agg is None:
                agg = s
            else:
                agg += s
        if not hasattr(s, "__len__"):
            return  p / self.size
        return agg
            
class ProductNode(object):
    def __init__(self, components, indices, bin_hashes=None, sign_hashes=None, corr_threshold=0.3, min_cluster=1e5, cluster_nbits=4, level=0, sparse=False, keys={}):
        self.children = []
        self.memory = 0
        for i, (c, index) in enumerate(zip(components, indices)):
            self.children.append(SPN(c, index, bin_hashes, sign_hashes, corr_threshold=corr_threshold, min_cluster=min_cluster, cluster_nbits=cluster_nbits, level=level, cluster_next=True, sparse=sparse, keys=keys))
            self.memory += self.children[i].memory
        return
    
    def sketch(self, predicates, keys, all=False, level=0, num_joins=defaultdict(lambda: 1), sparse=False, pessimistic=False):
        prob = 1
        sketches = None
        for child in self.children:
            if set(keys) & set(child.columns):
                if sketches is None:
                    sketches = child.sketch(predicates, keys, all=all, level=level+1, num_joins=num_joins, sparse=sparse, pessimistic=pessimistic)
                else:
                    sketches = ifft(fft(sketches.to_dense()) * fft( # fft does not support sparse tensors
                        child.sketch(predicates, keys, all=all, level=level+1, num_joins=num_joins, sparse=False, pessimistic=pessimistic))).real
            elif (set(predicates.keys()) & set(child.columns)):
                p = child.sketch(predicates, keys, level+1, pessimistic=pessimistic)
                if pessimistic and p < prob:
                    prob = p
                else:
                    prob *= p
        if sketches is not None:
            if sparse:
                return (prob * sketches).to_sparse()
            else:
                return (prob * sketches).to_dense()
        return prob

class ExactSketch(object):
    def __init__(self, data, bin_hashes=None, sign_hashes=None, index=None, fillna=-4242424242):
        self.fillna = fillna
        if index is None:
            index = data.copy(deep=True)
            na_mask = index.isna()
            index.fillna(fillna, inplace=True)
            for col, dtype in zip(index.columns, index.dtypes):
                if 'datetime' in str(dtype):
                    index[col] = index[col].astype('int64') // 10**9
                elif str(dtype) in 'object':
                    index[col] = index[col].apply(hash)
                elif str(dtype) in 'float':
                    index[col] = index[col] * 10**5
            index[na_mask] = fillna
            index = index.astype(int, copy=False)

        assert data.shape == index.shape, f'data {data.shape} mismatch with hashed index {index.shape}'
        self.data = data
        self.index = index

        self.columns = data.columns
        self.bin_hashes = bin_hashes
        self.sign_hashes = sign_hashes

    def sketch(self, predicates, keys, num_joins=defaultdict(lambda: 1)):
        assert set(keys).issubset(self.columns)
        filters = True
        for col in predicates:
            for op in predicates[col]:
                val = predicates[col][op]
                tmp = True
                if op == '=' or op == '==':
                    tmp &= self.data[col] == val
                elif op == '>=':
                    tmp &= self.data[col] >= val
                    tmp &= self.data[col].notna()
                elif op == '<=':
                    tmp &= self.data[col] <= val
                    tmp &= self.data[col].notna()
                elif op == '>':
                    tmp &= self.data[col] > val
                    tmp &= self.data[col].notna()
                elif op == '<':
                    tmp &= self.data[col] < val
                    tmp &= self.data[col].notna()
                elif op == '<>' or op == '!=':
                    tmp &= self.data[col] != val
                else:
                    raise SystemExit(f'unknown predicate operator in {col}{op}{val}')
                filters &= tmp
        selection = self.index if type(filters) is bool else self.index.loc[filters]
        
        depth = self.bin_hashes.fn.seeds.shape[0]
        width = self.bin_hashes.num_bins

        included_keys = [key for key in keys if num_joins[key] % 2 != 0]

        if len(included_keys):
            print(f'using fast-agms for {included_keys} out of {keys}')
            x = selection[included_keys].values

            bins = [bin_hashes(col) for col in x.T] # [d, N] x C
            bins = torch.stack(bins, dim=2)
            bins = bins.abs_()
            
            signs = [sign_hashes(col) for col in x.T] # [d, N] x C
            signs = torch.stack(signs, dim=2).type(torch.float32)

            combined_bins = bins.sum(dim=-1) % width
            combined_signs = signs.prod(dim=-1)
        else:
            print(f'using convoluted count-min for {keys}')
            x = selection[list(keys)].values
            bins = [bin_hashes(col) for col in x.T] # [d, N] x C
            bins = torch.stack(bins, dim=2)
            bins = bins.abs_()
            
            combined_bins = bins.sum(dim=-1) % width
            combined_signs = torch.ones(1).expand_as(combined_bins)

        sketch = torch.zeros((depth, width), dtype=torch.float32) 
        sketch.scatter_reduce_(-1, combined_bins, combined_signs, reduce='sum')
        return sketch

def process_sketch(sketch, width=None):
    depth, current_width = sketch.shape
    if width and current_width != width:
        sketch = sketch.to_dense()
        sketch = sketch.reshape((depth, width, -1)).sum(dim=2)
        assert width == sketch.shape[1], f'Could not resize sketch shape {sketch.shape} to width {width}'
    return sketch.to_dense()
    
def combine_sketches(table_nodes, join_edges, idx=0, visited=set(), primary={}, root=True):
    # print(f'visited {table_nodes[idx].name} ({idx}) from {visited}')
    visited.add(idx)

    sketch = table_nodes[idx].sketch
    is_primary = table_nodes[idx].name in primary and (primary[table_nodes[idx].name] & set(table_nodes[idx].keys.keys()))
    use_cross = len(table_nodes[idx].keys) > 1

    # primary key correction
    if is_primary and len(table_nodes) > 2:
        if sketch.is_sparse:
            sketch = sketch.coalesce().tanh_()
        else:
            sketch = sketch.clamp_(min=-1, max=1)

    cross_counter = 0
    for j, is_join in enumerate(join_edges[idx]):
        if is_join and j not in visited:
            # print(f'{table_nodes[idx].name} ({idx}) joins with {table_nodes[j].name} ({j})')
            candidate, visited = combine_sketches(table_nodes, join_edges, idx=j, visited=visited, primary=primary, root=False)
            if use_cross and (len(table_nodes[idx].keys) - cross_counter > 1 or not root):
                cross_counter += 1
                # print(f'{table_nodes[idx].name} ({idx}) cross-correlated with {table_nodes[j].name} ({j})')
                sketch = ifft(fft(candidate).conj() * fft(sketch)).real # [dD, w]
            else:
                # print(f'{table_nodes[idx].name} ({idx}) multiplied with {table_nodes[j].name} ({j})')
                assert sketch.shape[-1] == candidate.shape[-1], f'sketch of {table_nodes[idx].name} shape {sketch.shape} should have same width as {candidate.shape}'
                sketch *= candidate

    # visited_keys = set()
    # for v in visited:
    #     visited_keys |= table_nodes[v].keys
    # if visited_keys: assert table_nodes[idx].keys & visited_keys, f'{table_nodes[idx].name} keys {table_nodes[idx].keys} not in {visited_keys}'

    # print(f'return sketch of {table_nodes[idx].name} ({idx})')
    return sketch, visited

# change the products of all negative factors to positives
def combine_sketches_sign_v3(table_nodes, join_edges, idx=0, visited=set(), same_sign=0, signs_count=1, primary={}, root=True):
    # print(f'visited {idx} ({table_nodes[idx].name}) from visited {visited}\nedges {join_edges}\n {idx} in visited {idx in visited}')
    visited.add(idx)
    sketch = table_nodes[idx].sketch
    is_primary = table_nodes[idx].name in primary and (primary[table_nodes[idx].name] & set(table_nodes[idx].keys.keys()))
    use_cross = len(table_nodes[idx].keys) > 1
    # primary key correction
    if is_primary and len(table_nodes) > 2:
        if sketch.is_sparse:
            sketch = sketch.coalesce().tanh_()
        else:
            sketch = sketch.clamp_(min=-1, max=1)
    same_sign = sketch.sign()
 
    cross_counter = 0
    for j, is_join in enumerate(join_edges[idx]):
        if is_join and (j not in visited) and (j != idx):
            # print(f'entering {j}')
            candidate, visited, signs_other, num_other = combine_sketches_sign_v3(table_nodes, join_edges, idx=j, visited=visited, primary=primary, root=False)
            if use_cross and (len(table_nodes[idx].keys) - cross_counter > 1 or not root):
                cross_counter += 1
                if num_other > 1:
                    # odd product correction
                    assert type(signs_other) is torch.Tensor, f'{signs_other} must be a tensor'
                    candidate[(signs_other == -num_other) & (num_other % 2 == 1)] *= -1
                sketch = ifft(fft(candidate).conj() * fft(sketch)).real
            else:
                assert sketch.shape[-1] == candidate.shape[-1], f'sketch of {table_nodes[idx].name} shape {sketch.shape} should have same width as {candidate.shape}'
                # accumulate signs
                same_sign += signs_other
                signs_count += num_other
                sketch *= candidate

    if use_cross:
        # don't include sign of cross-correlated sketch
        return sketch, visited, 0, 0
    elif root and signs_count > 1:
        # odd product bias correction
        sketch[(same_sign == -signs_count) & (signs_count % 2 == 1)] *= -1
        return sketch, visited, 0, -1
    return sketch, visited, same_sign, signs_count

def cross_estimate(params):
    table_nodes, join_edges, method, width, primary, same_sign = deepcopy(params)
    for node in table_nodes:
        # print(node.name, node.sketch)
        node.sketch = process_sketch(node.sketch, width=width)
    medians = table_nodes[0].sketch.shape[0]

    estimation_time = perf_counter_ns()
    # only use same-sign heuristic for single-attribute joins
    if same_sign:# and max(len(n.keys) for n in nodes) == 1:
        sketch, visited, _, _ = combine_sketches_sign_v3(table_nodes, join_edges, primary=primary)
    else:
        sketch, visited = combine_sketches(table_nodes, join_edges, primary=primary)
    assert visited == set(range(len(table_nodes))), f'visited nodes {visited} less than {len(table_nodes)}\nTables: {[t.name for t in table_nodes]}\nJoin graph adjacency matrix:\n{join_edges}\nsame-sign {same_sign}'
    visited.clear() # ??? removing this breaks something
    del visited


    dims = tuple(i for i in range(1, sketch.dim()))
    D, W = sketch.shape
    assert W == width, f'width of sketch products {sketch.shape} does not match expected size {width}'
    est = sketch.sum(dim=dims)
    assert est.numel() % medians == 0, f'number of estimates {est.shape} is not a multiple of {D}'
    estimation_time = perf_counter_ns() - estimation_time

    agg = defaultdict(float)
    for d in range(1, medians+1, 2):
        key = f"{method}_{1}x{d}x{width}{'_primary' * bool(primary)}"
        i = 0
        for i, combo in enumerate(combinations(range(D), d)):
            if method == 'count-min':
                agg[key] += est[combo,].min().item()
            else:
                agg[key] += est[combo,].quantile(0.5).item()
        if i > 0: agg[key] /= i + 1
    return agg, estimation_time

def estimate(models, table_nodes, join_edges, widths, methods=['fast-agms'], primary=dict(), same_sign=False, sparse=True, pessimistic=False):
    inference_times = []
    estimation_times = []
    aggregates = defaultdict(float)
    for method in methods:
        for i, node in enumerate(table_nodes):
            inference_ts = perf_counter_ns()
            node.sketch = models[node.name].sketch(node.predicates, set(node.keys.keys()), num_joins=defaultdict(lambda: 2) if method.lower() == 'count-min' else (defaultdict(lambda: 1) if same_sign else node.keys), sparse=sparse, pessimistic=pessimistic)
            # print(f'{node.name} {node.sketch.shape} approx:\n{node.sketch}')
            inference_times.append(perf_counter_ns() - inference_ts)

        for prod in product((table_nodes,), (join_edges,), (method,), widths, (primary, {},), tuple({same_sign})):
            agg, est_time = cross_estimate(prod)
            aggregates |= agg
            estimation_times.append(est_time)

    return aggregates, deepcopy(table_nodes), pd.Timedelta(sum(inference_times), unit='ns'), pd.Timedelta(sum(estimation_times) / len(estimation_times), unit='ns')

def exact(tables, table_nodes, join_edges, widths, methods=['fast-agms'], primary=dict(), same_sign=False):
    sketching_times = []
    aggregates = defaultdict(float)
    for method in methods:
        for i, node in enumerate(table_nodes):
            sketch_ts = perf_counter_ns()
            node.sketch = tables[node.name].sketch(node.predicates, set(node.keys.keys()), num_joins=defaultdict(lambda: 2) if method.lower() == 'count-min' else (defaultdict(lambda: 1) if same_sign else node.keys))
            sketching_times.append(perf_counter_ns() - sketch_ts)
        for prod in product((table_nodes,), (join_edges,), (method,), widths, (primary, {},), tuple({same_sign})):
            agg, _ = cross_estimate(prod)
            agg = {f'exact_{k}': v for k,v in agg.items()}
            aggregates |= agg
    return aggregates, deepcopy(table_nodes), pd.Timedelta(sum(sketching_times), unit='ns')

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

    from utils import get_dataframe, get_workload, extract_graph
    parser = argparse.ArgumentParser(description='run sketched sum-product networks on a workload')
    parser.add_argument('--depth', default=5, type=lambda x: int(float(x)), help='depth of sketches')
    parser.add_argument('--width', '--widths', nargs='*', default=[1024,], type=lambda x: int(float(x)), help='width(s) of sketche (widths should be evenly divisible by smaller widths, if multiple are specified)')
    parser.add_argument('--workload', default='./stats_CEB_sub_queries_corrected.sql', help='CSV containing the format (subqueries || parent ID || cardinality)')
    parser.add_argument('--data', default='./End-to-End-CardEst-Benchmark-master/datasets/stats_simplified/', help='path to directory of table files (as CSVs)')
    parser.add_argument('--writefile', default='out.csv', help='name of output csv file')
    parser.add_argument('--k', default=1, type=int, help='each Sum Node partitions data into k**2 clusters')
    parser.add_argument('--decompose', '--corr_threshold', default=0.3, type=float, help='pairs of columns are decomposed with less correlation than this threshold (default 0.3)')
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
    parser.add_argument('--independence', default=64, type=int, help='use k-wise independent hashing (recommended k=2**n for n-way joins)')
    parser.add_argument('--pessimistic', action='store_true', help='use pessimistic (probabilistic upper bound) sketch approximation')
    args = parser.parse_args()
    print(args)

    workload = get_workload(args.workload)
    if args.find_keys:
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

    # args.methods = ['fast-agms',] # only fast-agms is supported currently
    for m in ['fast-agms',]:
        for c in [1]:
            for d in range(1, args.depth+1, 2):  
                for w in args.width:  
                    workload[f'{m}_{c}x{d}x{w}'] = -1.0
                    workload[f'{m}_{c}x{d}x{w}_err'] = -1.0
                    workload[f'exact_{m}_{c}x{d}x{w}'] = -1.0
                    workload[f'exact_{m}_{c}x{d}x{w}_err'] = -1.0
                    if args.primary:
                        workload[f'{m}_{c}x{d}x{w}_primary'] = -1.0
                        workload[f'{m}_{c}x{d}x{w}_primary_err'] = -1.0
                        workload[f'exact_{m}_{c}x{d}x{w}_primary'] = -1.0
                        workload[f'exact_{m}_{c}x{d}x{w}_primary_err'] = -1.0

    workload['num_tables'] = 0
    workload['join_attributes'] = 0
    workload['similarity'] = 0
    workload['exact_time'] = pd.Timedelta(0.0, unit='sec')
    workload['inference_time'] = pd.Timedelta(0.0, unit='sec')
    workload['estimation_time'] = pd.Timedelta(0.0, unit='sec')
    workload['total_time'] = pd.Timedelta(0.0, unit='sec')
    workload = workload.copy()
    print(f'Generating results into workload ({workload.shape})')

    if args.experiment == 'job-light':
        args.workload = './job_light_sub_query_with_star_join.sql.txt'
        args.primary = ['title.id']
        args.dates = []
        tables = {'title': {
                        'names': ['id', 'title', 'imdb_index', 'kind_id', 'production_year',
                                'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr',
                                'episode_nr', 'series_years', 'md5sum'],
                        'columns': ['id', 'kind_id', 'series_years', 'production_year',
                                'phonetic_code', 'season_nr', 'episode_nr', 'imdb_index'],
                        'keys': {'id'},},
                'movie_companies': {'names': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                        'columns': ['company_type_id', 'company_id', 'movie_id'], 
                        'keys': {'movie_id'},},
                'movie_info_idx': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                        'columns': ['info_type_id', 'movie_id'], 
                        'keys': {'movie_id'},},
                'movie_keyword': {'names': ['id', 'movie_id', 'keyword_id'],
                        'columns': ['keyword_id', 'movie_id'], 
                        'keys': {'movie_id'},},
                'movie_info': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                        'columns': ['info_type_id', 'movie_id'], 
                        'keys': {'movie_id'},},
                'cast_info': {'names': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                        'columns': ['role_id', 'nr_order', 'movie_id'], 
                        'keys': {'movie_id'},},
                        
                }
    elif args.experiment == 'stats-ceb':
        args.workload = './stats_CEB_sub_queries_corrected.sql'
        args.primary = ['badges.Id',
                                'posts.Id',
                                'postLinks.Id',
                                'postHistory.Id',
                                'comments.Id',
                                'tags.Id',
                                'users.Id',
                                'votes.Id']
        args.dates = ['badges.Date', 
                                'comments.CreationDate', 
                                'postHistory.CreationDate', 
                                'postLinks.CreationDate', 
                                'posts.CreationDate', 
                                'users.CreationDate', 
                                'votes.CreationDate']
        tables = {'badges': defaultdict(lambda: None),
                  'comments': defaultdict(lambda: None),
                  'postHistory': defaultdict(lambda: None),
                  'postLinks': defaultdict(lambda: None),
                  'posts': defaultdict(lambda: None),
                  'tags': defaultdict(lambda: None),
                  'users': defaultdict(lambda: None),
                  'votes': defaultdict(lambda: None),}
        tables['badges']['keys'] = {'UserId'}
        tables['users']['keys'] = {'Id'}
        tables['comments']['keys'] = {'UserId', 'PostId'}
        tables['postHistory']['keys'] = {'UserId', 'PostId'}
        tables['votes']['keys'] = {'UserId', 'PostId'}
        tables['posts']['keys'] = {'OwnerUserId', 'Id'}
        tables['postLinks']['keys'] = {'RelatedPostId', 'PostId'}
        tables['tags']['keys'] = {'ExcerptPostId'}

    if args.primary:
        primary = defaultdict(set)
        for t, c in map(lambda x: x.split('.'), args.primary):
            primary[t].add(c)
        args.primary = primary

    if args.dates:
        dates = defaultdict(set)
        for t, c in map(lambda x: x.split('.'), args.dates):
            dates[t].add(c)
        args.dates = dict(dates)

    bin_hashes, sign_hashes = get_hashes(args.depth, max(args.width))

    with torch.inference_mode():
        models = dict()
        exact_sketches = dict()
        for table, cols in tables.items():
            ts = perf_counter_ns()
            dataset = get_dataframe(f'{args.data}/{table}.csv', names=cols['names'], columns=cols['columns'])
            if table in args.dates:
                for col in args.dates[table]:
                    dataset[col] = pd.to_datetime(dataset[col])
            delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
            print(f"Loaded {table} ({dataset.memory_usage(deep=True).sum():,} bytes) {delta.total_seconds():>25,.2f}s ({delta})", flush=True)
            print(dataset.describe().to_string(float_format="{:,.2f}".format))
            print(dataset.memory_usage(deep=True).to_string(float_format="{:,.2f}".format))

            ts = perf_counter_ns()
            min_cluster = args.min_cluster if args.min_cluster > 1 else abs(args.min_cluster * len(dataset))
            models[table] = SPN(dataset, bin_hashes=bin_hashes, sign_hashes=sign_hashes, min_cluster=min_cluster, cluster_nbits=args.k, cluster_next=args.cluster_first, sparse=args.sparse, keys=cols['keys'])
            delta = pd.Timedelta(perf_counter_ns() - ts, unit='ns')
            print(f"Trained SPN ({models[table].memory:,} bytes) on {table} ({delta})", flush=True)

            # exact sketches for comparison
            exact_sketches[table] = ExactSketch(dataset, bin_hashes=bin_hashes, sign_hashes=sign_hashes)

        for i, row in enumerate(workload.iloc()):
            query_start = perf_counter_ns()
            query = row['query']
            nodes, edges = extract_graph(query)
            max_chains = max(len(n.keys) for n in nodes)
            num_components = 1 + sum(len(n.keys)-1 for n in nodes)
            # if num_components == 1: continue

            print(f"{i}: {query} ({row['cardinality']:,})")
            
            exact_estimates, sketches_1, exact_time = exact(exact_sketches, nodes, edges, args.width, primary=args.primary, same_sign=(num_components == 1)) 
            for k, est in exact_estimates.items():
                workload.loc[i, k] = est
                workload.loc[i, k + '_err'] = max(est, row['cardinality']) / max(min(est, row['cardinality']), 1)
            estimates, sketches_2, inference_time, estimation_time = estimate(models, nodes, edges, args.width, primary=args.primary, same_sign=(num_components == 1), sparse=args.sparse, pessimistic=args.pessimistic)
            for k, est in estimates.items():
                workload.loc[i, k] = est
                workload.loc[i, k + '_err'] = max(est, row['cardinality']) / max(min(est, row['cardinality']), 1)
            query_time = pd.Timedelta(perf_counter_ns() - query_start, unit='ns')
            similarities = compare_sketches(sketches_1, sketches_2)
            for table, sim in similarities.items():
                print(f'sketch of table {table} has approximation similarity {sim}')

            workload.loc[i, 'similarity'] = sum(similarities.values()) / len(similarities)
            workload.loc[i, 'num_tables'] = len(nodes)
            workload.loc[i, 'join_attributes'] = num_components
            workload.loc[i, 'exact_time'] = exact_time
            workload.loc[i, 'inference_time'] = inference_time
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

        for i in range(workload['num_tables'].min(), workload['num_tables'].max() + 1):
            print(f'\n{i}-way Joins:')
            print(workload.query(f"`num_tables` == {i}", engine='python')[cols].describe(percentiles=pctl).transpose().drop(columns=drop).to_string(float_format="{:,.2f}".format))

        print('\nAverage Model Inference Time:')
        print(workload['inference_time'].mean())
        print('\nAverage Estimation Time:')
        print(workload['estimation_time'].mean())
        print('\Total Time:')
        print(workload['total_time'].sum())