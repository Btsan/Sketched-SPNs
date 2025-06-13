from time import perf_counter_ns
import re

import numpy as np
import pandas as pd
import torch

# from Estimators import CountEstimator, DegreeEstimator

class Sketch(object):
    """Base class for sketches."""

    def memory_usage(self):
        return 0
    
    def __call__(self, predicates:dict, keys:dict, **kwargs):
        """
        returns:
            the selectivity of the predicates (float) or the sketch of the keys (Tensor)
        """
        raise NotImplementedError("Subclasses should implement this method.")

class AMS(Sketch):
    def __init__(self, data:pd.DataFrame, depth:int, sign_hashes:list, **kwargs):
        self.depth = depth
        self.nrows = len(data)
        self.sign_hashes = sign_hashes
        self.saved = dict()

        self.columns = {data.name,} if isinstance(data, pd.Series) else set(data.columns)

        # saves the dataframe with only distinct rows and their counts
        self.distincts = data.value_counts(dropna=False).sort_values(ascending=False).reset_index(name='_count')
        
        # hashes for computing the ad-hoc sketches
        self.signs = dict()
        for col in self.columns:
            values = self.distincts[col].map(hash).values + 1 # [N]
            mask = self.distincts[col].notnull().values[None, :] # [1, N]
            self.signs[col] = [sign_hash(values) * mask for sign_hash in sign_hashes]

        self.memory = self.distincts.memory_usage().sum()
        for col in self.columns:
            for hashes in self.signs[col]:
                self.memory += hashes.numel() * hashes.element_size()

        # memory usage of pushdown (exact) sketches
        self.pushdown = dict()

    def memory_usage(self):
        nbytes = sum(self.pushdown.values())
        for sketch in self.saved.values():
            nbytes += sketch.numel() * sketch.element_size()
        
        return nbytes

    def __call__(self, predicates:dict, keys:dict, **kwargs):
        """
        returns:
            the selectivity of the predicates (float) or the sketch of the keys (Estimator)
        """
        col_in_preds = self.columns.intersection(predicates.keys())
        col_in_keys = self.columns.intersection(keys.keys())

        selection = self.distincts
        if col_in_preds:
            preds = []
            for col in col_in_preds:
                for op, val in predicates[col].items():
                    if op == '=':
                        op = '==' # pandas uses '==' for equality
                    # if sel[col].dtype not in (int, float):
                    if not pd.api.types.is_numeric_dtype(selection[col]):
                        val = f"'{val}'"
                    preds.append(f"`{col}`{op}{val}")
            q = " & ".join(preds)
            selection = selection.query(q)

            if col_in_keys:
                signs = 1
                for col, join_indices in keys.items():
                    values = selection[col].map(hash).values + 1
                    for join_idx in join_indices:
                        signs *= self.sign_hashes[join_idx](values)
                    mask = selection[col].notnull().values[None, :]
                    signs *= mask
                assert signs.shape == (self.depth, max(1, len(selection))), f"{signs.shape} == {(self.depth, len(selection))}"
                signs *= selection['_count'].values[None, :]
                sketch = signs.sum(dim=-1, keepdim=True).float()
                
                # record memory usage of pushdown sketches
                # assumes sketch of selection is only ever computed once
                pushdown_id = frozenset(keys.items()).union(preds)
                self.pushdown[pushdown_id] = sketch.numel() * sketch.element_size()

                return sketch
            
            prob = (selection['_count'].sum()) / self.nrows
            return prob, 0
        elif not col_in_keys:
            return 1, 0

        # check if sketch already exists
        sketch_id = frozenset(keys.items())
        if not col_in_preds and sketch_id in self.saved:
            return self.saved[sketch_id].to_dense(), 0
        
        # measure sketcching time
        t0 = perf_counter_ns()
        
        signs = 1
        for key, join_indices in keys.items():
            for join_idx in join_indices:
                signs *= self.signs[key][join_idx]
        assert signs.shape == (self.depth, max(1, len(self.distincts))), f"{signs.shape} == {(self.depth, len(self.distincts))}"
        signs *= self.distincts['_count'].values[None, :]
        sketch = signs.sum(dim=-1, keepdim=True).float()
        
        t1 = perf_counter_ns()
        sketch_time = (t1 - t0)

        # save sketch for reuse, if there were no predicates
        if not col_in_preds:
            self.saved[sketch_id] = sketch.to_sparse()
        return sketch, sketch_time

class FastAGMS(Sketch):
    def __init__(self, data:pd.DataFrame, depth:int, width:int, sign_hashes:list, bin_hashes:list, exact_preds=False, sparse=False, **kwargs):
        self.depth = depth
        self.width = width
        self.nrows = len(data)
        self.sign_hashes = sign_hashes
        self.bin_hashes = bin_hashes
        self.sparse = sparse

        self.columns = [data.name,] if isinstance(data, pd.Series) else list(data.columns)

        # creates a dataframe with only distinct rows and their counts
        self.distincts = data.value_counts(dropna=False).sort_values(ascending=False).reset_index(name='_count')

        self.vhash = np.vectorize(hash)
        values = self.vhash(self.distincts[self.columns].values) + 1 # [N, col]
        mask = self.distincts[self.columns].notnull().values[None, :, :] # [1, N, col]
        signs = [sign_hash(values) * mask for sign_hash in sign_hashes]
        bins = [bin_hash(values) for bin_hash in bin_hashes]
        # print(f"values {values.shape} mask {mask.shape} signs {signs[0].shape} bins {bins[0].shape}")
        self.signs = {col: [signs_all[:, :, i] for signs_all in signs] for i, col in enumerate(self.columns)}
        self.bins = {col: [bins_all[:, :, i] for bins_all in bins] for i, col in enumerate(self.columns)}

        self.columns = set(self.columns)

        self.sketches = dict()
        self.memory = self.distincts.memory_usage().sum()
        for col in self.columns:
            for hashes in self.signs[col]:
                self.memory += hashes.numel() * hashes.element_size()
            for hashes in self.bins[col]:
                self.memory += hashes.numel() * hashes.element_size()
        
        # memory usage of pushdown (exact) sketches
        self.pushdown = dict()

        # Count-Min for predicate selectivity
        self.countmins = {}
        if not exact_preds:
            for col in self.columns:
                values = self.vhash(self.distincts[col].values) + 1 # N
                mask = self.distincts[col].notnull().values[None, :] # 1, N
                # bins = torch.concatenate([bin_hash(values) for bin_hash in bin_hashes], dim=0)
                bins = bin_hashes[0](values) % self.width
                counts = torch.tensor(self.distincts['_count'].values)[None, :].expand_as(bins)
                counts *= mask # don't count nulls
                # assert bins.shape == counts.shape == (self.depth * len(bin_hashes), len(distincts)), \
                #     f"{bins.shape} == {counts.shape} == {self.depth * len(bin_hashes), len(distincts)}"
                assert bins.shape == counts.shape == (self.depth, len(self.distincts)), \
                    f"{bins.shape} == {counts.shape} == {self.depth, len(self.distincts)}"
                # print(f"\n{col} {distincts['_count']}  counts {counts}")
                # print(f"\nvalues {values} bins {bins}")
                # self.countmins[col] = torch.zeros((self.depth * len(bin_hashes), self.width), dtype=torch.long).scatter_add_(1, bins, counts)
                self.countmins[col] = torch.zeros((self.depth, self.width), dtype=torch.long).scatter_add_(1, bins, counts)

    def memory_usage(self):
        nbytes = sum(self.pushdown.values())
        for sketch in self.sketches.values():
            if sketch.is_sparse:
                indices = sketch.indices()
                nbytes += indices.nelement() * indices.element_size()
                values = sketch.values()
                nbytes += values.nelement() * values.element_size()
            else:
                nbytes += sketch.numel() * sketch.element_size()
        return nbytes
    
    def __call__(self, predicates:dict, keys:dict, components:dict, cuda : bool = False, separate_negatives : bool = False, **kwargs):
        """
        returns:
            the selectivity of the predicates (float) or the sketch of the keys (Estimator)
        """
        col_in_preds = self.columns.intersection(predicates.keys())
        col_in_keys = self.columns.intersection(keys.keys())

        if separate_negatives:
            sketch_id = frozenset(keys.keys()).union(components.items())
        else:
            sketch_id = frozenset(keys.items()).union(components.items())
        preds = []
        if not col_in_keys and not col_in_preds:
            # if no selection is needed and not a join key attribute, return 1
            return 1, 0
        elif col_in_preds:
            # otherwise, filter selection is needed
            for col in col_in_preds:
                for op, val in predicates[col].items():
                    if op == '=':
                        op = '==' # pandas uses '==' for equality
                    if not pd.api.types.is_numeric_dtype(self.distincts[col]):
                        val = f"'{val}'"
                    preds.append(f"`{col}`{op}{val}")
            q = " & ".join(preds)
            selection = self.distincts.query(q)
            # print(f"{q} --> {len(selection)}/{len(self.distincts_lo)} {len(sel_hi)}/{len(self.distincts_hi)}")

            if col_in_keys:
                # pushdown sketch of infrequent items
                t0 = perf_counter_ns()
                sketch = torch.zeros((self.depth, self.width * (2 if separate_negatives else 1)), dtype=torch.float)
                if len(selection) > 0:
                    selection = selection.groupby(list(keys.keys())).sum('_count').reset_index()
                    signs = 1
                    negatives = 1
                    bins = 0
                    for key, join_indices in keys.items():
                        values = self.vhash(selection[key].values) + 1
                        bins += self.bin_hashes[components[key]](values)
                        if separate_negatives:
                            temp = self.sign_hashes[0](values)
                            signs *= temp
                            negatives *= temp * (temp < 0)
                        else:
                            for join_idx in join_indices:
                                signs *= self.sign_hashes[join_idx](values)
                        # mask = selection[key].notnull().values[None, :] # [1, N]
                        # signs *= mask
                    assert bins.shape == signs.shape == (self.depth, max(1, len(selection))), f"{bins.shape} == {signs.shape} == {(self.depth, len(selection))}"
                    bins %= self.width
                    signs *= selection['_count'].values[None, :]

                    # assert bins.dtype == torch.int64, f"bins {bins.dtype} {bins.shape} {bins}\nsigns {signs.dtype} {signs.shape} {signs}\ndistincts_lo {self.distincts_lo}"
                    sketch.view(self.depth, -1).scatter_add_(1, bins.long(), signs.float())

                    if separate_negatives:
                        # keep separate counters for purely negative factors
                        bins += self.width
                        sketch.view(self.depth, -1).scatter_add_(1, bins.long(), negatives.float())

                t1 = perf_counter_ns()
                sketch_time = (t1 - t0)

                # record memory usage of pushdown sketches
                # assumes sketch of selection is only ever computed once
                pushdown_id = sketch_id.union(preds)
                self.pushdown[pushdown_id] = sketch.numel() * sketch.element_size()
                return sketch, sketch_time
            else:
                # return probability if not a join key attribute
                if not self.countmins:
                    prob = selection['_count'].sum() / self.nrows
                else:
                    # convert to count-min probability
                    freq = torch.zeros((self.depth, self.width), dtype=torch.long)
                    for col in col_in_preds:
                        if len(selection) > 0:
                            values = selection[col].map(hash).values + 1
                            bins = self.bin_hashes[0](values) # depth, N
                            freq += self.countmins[col].gather(1, bins).sum(dim=1).min().item()
                    prob = freq.sum(dim=-1).min().item() / self.nrows
                return prob, 0

        # check if sketch already exists
        if not col_in_preds and sketch_id in self.sketches:
            if self.sparse:
                return self.sketches[sketch_id].to_dense(), 0
            else:
                return self.sketches[sketch_id].detach().clone(), 0

        # measure sketcching time
        t0 = perf_counter_ns()
        
        # sketch infrequent items
        sketch = torch.zeros((self.depth, self.width * (2 if separate_negatives else 1)), dtype=torch.float)
        if len(self.distincts) > 0:
            signs = 1
            negatives = 1
            bins = 0
            for key, join_indices in keys.items():
                bins += self.bins[key][components[key]]
                if separate_negatives:
                    temp = self.signs[key][0]
                    signs *= temp
                    negatives *= temp * (temp < 0)
                else:
                    for join_idx in join_indices:
                        signs *= self.signs[key][join_idx]
            assert bins.shape == signs.shape == (self.depth, max(1, len(self.distincts))), \
                f"{bins.shape} == {signs.shape} == {(self.depth, len(self.distincts))}"
            bins %= self.width
            signs *= self.distincts['_count'].values[None, :]

            sketch.view(self.depth, -1).scatter_add_(1, bins.long(), signs.float())
            
            if separate_negatives:
                # keep separate counters to track negative updates
                bins += self.width
                sketch.view(self.depth, -1).scatter_add_(1, bins.long(), negatives.float())

        t1 = perf_counter_ns()
        sketch_time = (t1 - t0)

        if not col_in_preds:
            if self.sparse:
                self.sketches[sketch_id] = sketch.to_sparse()
            else:
                self.sketches[sketch_id] = sketch.detach().clone()
        return sketch, sketch_time
    
class BoundSketch(Sketch):
    def __init__(self, data:pd.DataFrame, depth:int, width:int, bin_hashes:list, exact_preds=False, sparse=False, **kwargs):
        self.depth = depth
        self.width = width
        self.nrows = len(data)
        self.bin_hashes = bin_hashes
        self.sparse = sparse

        self.columns = [data.name,] if isinstance(data, pd.Series) else list(data.columns)

        # creates a dataframe with only distinct rows and their counts
        self.distincts = data.value_counts(dropna=False).sort_values(ascending=False).reset_index(name='_count')
        assert self.distincts['_count'].sum() == self.nrows

        self.vhash = np.vectorize(hash)
        values = self.vhash(self.distincts[self.columns].values) + 1 # [N, col]
        bins = [bin_hash(values) for bin_hash in bin_hashes]
        # print(f"values {values.shape} mask {mask.shape} signs {signs[0].shape} bins {bins[0].shape}")
        # self.bins = {col: [bins_all[:, :, i] for bins_all in bins] for i, col in enumerate(self.columns)}

        self.columns = set(self.columns)

        # save computed sketches
        self.sketches = dict()

        self.memory = self.distincts.memory_usage().sum()

        # memory usage of pushdown (exact) sketches
        self.pushdown = dict()

        # Count-Min for predicate selectivity
        self.countmins = {}
        if not exact_preds:
            for col in self.columns:
                values = self.vhash(self.distincts[col].values) + 1 # N
                mask = self.distincts[col].notnull().values[None, :] # 1, N
                bins = bin_hashes[0](values) % self.width
                counts = torch.tensor(self.distincts['_count'].values)[None, :].expand_as(bins)
                counts *= mask # don't count nulls
                # assert bins.shape == counts.shape == (self.depth * len(bin_hashes), len(self.distincts)), \
                #     f"{bins.shape} == {counts.shape} == {self.depth * len(bin_hashes), len(self.distincts)}"
                assert bins.shape == counts.shape == (self.depth, len(self.distincts)), \
                    f"{bins.shape} == {counts.shape} == {self.depth, len(self.distincts)}"
                # self.countmins[col] = torch.zeros((self.depth * len(bin_hashes), self.width), dtype=torch.long).scatter_add_(1, bins, counts)
                self.countmins[col] = torch.zeros((self.depth, self.width), dtype=torch.long).scatter_add_(1, bins, counts)
            
    def memory_usage(self):
        nbytes = sum(self.pushdown.values())
        for sketch in self.sketches.values():
            if sketch.is_sparse:
                indices = sketch.indices()
                nbytes += indices.nelement() * indices.element_size()
                values = sketch.values()
                nbytes += values.nelement() * values.element_size()
            else:
                nbytes += sketch.numel() * sketch.element_size()
        return nbytes
    
    def __call__(self, predicates:dict, keys:dict, components:dict, count: bool = True, cuda: bool = False, **kwargs):
        """
        returns:
            the selectivity of the predicates (float) or the sketch of the keys (Estimator)
        """
        reduce_mode = 'sum' if count else 'amax'
        col_in_preds = self.columns.intersection(predicates.keys())
        col_in_keys = self.columns.intersection(keys.keys())


        preds = []
        if not col_in_keys and not col_in_preds:
            # if no selection is needed and not a join key attribute, return 1
            return 1, 0
        elif col_in_preds:
            # otherwise, filter selection is needed
            for col in col_in_preds:
                for op, val in predicates[col].items():
                    if op == '=':
                        op = '==' # pandas uses '==' for equality
                    if not pd.api.types.is_numeric_dtype(self.distincts[col]):
                        val = f"'{val}'"
                    preds.append(f"`{col}`{op}{val}")
            q = " & ".join(preds)
            selection = self.distincts.query(q)
            # print(f"{q} --> {len(sel_lo)}/{len(self.distincts)} {len(sel_hi)}/{len(self.distincts_hi)}")
            if col_in_keys:
                # return a pushdown sketch
                t0 = perf_counter_ns()
                sketch = torch.zeros((self.depth, self.width), dtype=torch.float)
                if len(selection) > 0:
                    selection = selection.groupby(list(keys.keys())).sum('_count').reset_index()
                    bins = 0
                    for key, _ in keys.items():
                        values = self.vhash(selection[key].values) + 1
                        bins += self.bin_hashes[components[key]](values)
                    bins %= self.width
                    counts = torch.tensor(selection['_count'].values)
                    counts = counts[None, :].expand_as(bins) 
                    assert bins.shape == counts.shape == (self.depth, max(1, len(selection))), \
                        f"{bins.shape} == {counts.shape} == {(self.depth, len(selection))}"

                    sketch.view(self.depth, -1).scatter_reduce_(1, bins.long(), counts.float(), reduce_mode)
                t1 = perf_counter_ns()
                sketch_time = (t1 - t0)
                # record memory usage of pushdown sketches
                # assumes sketch of selection is only ever computed once
                pushdown_id = frozenset(keys.keys()).union(components.items()).union(preds).union({('count', count)})
                self.pushdown[pushdown_id] = sketch.numel() * sketch.element_size()
                return sketch, sketch_time
            else:
                # return probability if not a join key attribute
                if not self.countmins:
                    prob = (selection['_count'].sum()) / self.nrows
                else:
                    # convert to count-min probability
                    prob = 1
                    freq = torch.zeros((self.depth,), dtype=torch.long)
                    for col in col_in_preds:
                        if len(selection) > 0:
                            values = selection[col].map(hash).values + 1
                            bins = self.bin_hashes[0](values) # depth, N
                            freq += self.countmins[col].gather(1, bins).sum(dim=1).min().item()
                    prob *= freq.min().item() / self.nrows
                return prob, 0
        else:
            # otherwise no filters are applied and proceed to sketching
            selection = self.distincts

        # check if sketch already exists
        sketch_id = frozenset(keys.keys()).union(components.items()).union({('count', count)})
        if not col_in_preds and sketch_id in self.sketches:
            if self.sparse:
                return self.sketches[sketch_id].to_dense(), 0
            else:
                return self.sketches[sketch_id].detach().clone(), 0
        
        # measure sketcching time
        t0 = perf_counter_ns()

        # group by join keys
        selection = selection.groupby(list(keys.keys())).sum('_count').reset_index()

        sketch = torch.zeros((self.depth, self.width), dtype=torch.float)
        if len(selection) > 0:
            bins = 0
            for key, _ in keys.items():
                values = selection[key].map(hash).values + 1
                bins += self.bin_hashes[components[key]](values)
            bins %= self.width
            counts = torch.tensor(selection['_count'].values)
            counts = counts[None, :].expand_as(bins)
            assert bins.shape == counts.shape == (self.depth, max(1, len(selection))), \
                f"{bins.shape} == {counts.shape} == {(self.depth, len(selection))}"

            sketch.view(self.depth, -1).scatter_reduce_(1, bins.long(), counts.float(), reduce_mode)

        t1 = perf_counter_ns()
        sketch_time = (t1 - t0)
        
        # save sketch for reuse, if there were no predicates
        if not col_in_preds:
            if self.sparse:
                self.sketches[sketch_id] = sketch.to_sparse()
            else:
                self.sketches[sketch_id] = sketch.detach().clone()
        else:
            # record memory usage of pushdown sketches
            # assumes pushdown sketch is only ever computed once in a workload
            pushdown_id = sketch_id.union(preds)
            self.pushdown[pushdown_id] = sketch.numel() * sketch.element_size()
        return sketch, sketch_time
    
# sketches for selectivity estimation

class ExactSelectivity(Sketch):
    def __init__(self, data:pd.DataFrame, **kwargs):
        self.nrows = len(data)
        self.columns = [data.name,] if isinstance(data, pd.Series) else list(data.columns)
        self.distincts = data.value_counts(dropna=False).sort_values(ascending=False).reset_index(name='_count')
        self.columns = set(self.columns)
        self.memory = self.distincts.memory_usage().sum()
        
        # cache the selectivity of the predicates
        self.saved = dict()

    def sql_like_to_regex(self, sql_like: str) -> str:
        """Convert SQL LIKE pattern to regex."""
        # Escape special regex characters except % and ?
        escaped = re.escape(sql_like).replace(r'\%', '%').replace(r'\?', '?')
        
        # Convert SQL LIKE wildcards to regex wildcards
        regex_pattern = escaped.replace('%', '.*').replace('?', '.')

        return f"^{regex_pattern}$"  # Ensure full-string matching like SQL LIKE

    def __call__(self, predicates:dict, *args, **kwargs):
        """
        returns:
            the selectivity of the predicates (float) or the sketch of the keys (Estimator)
        """
        col_in_preds = self.columns.intersection(predicates.keys())
        if not col_in_preds:
            return 1, 0
        # otherwise, filter selection is needed

        pred_id = frozenset({f'{col}{op}{val}' for col in col_in_preds for op, val in predicates[col].items()})
        
        # check if selecitivity is in cache
        if pred_id in self.saved:
            return self.saved[pred_id], 0
        
        preds = []
        for col in col_in_preds:
            use_string = not pd.api.types.is_numeric_dtype(self.distincts[col])
            for op, val in predicates[col].items():
                if str.upper(op) == 'LIKE':
                    # convert SQL LIKE to regex
                    val = self.sql_like_to_regex(val)
                    preds.append(f"`{col}`.str.contains(r'{val}', case=False, regex=True)")
                else:
                    if op == '=':
                        op = '=='
                    if use_string:
                        val = f"'{val}'"
                    preds.append(f"`{col}`{op}{val}")
        
        q = " & ".join(preds)
        selection = self.distincts.query(q)

        prob = (selection['_count'].sum()) / self.nrows
        self.saved[pred_id] = prob
        return prob, 0

def calculate_intervals(left:int, right:int, intervals:list):
    """
    Calculate the minimal set of available intervals that cover the range [left, right].
    The cover is calculated starting with the largest interval size and then iteratively
    refining the left and right-most intervals with smaller ones.
    The resulting set of intervals covers the entire range [left, right] with minimal overlap.

    Returns:
        A dictionary with the interval size as the key and a numpy array of intervals as the value.
        The intervals are represented as integers, where each integer represents an interval
        of the form [i * interval_size, (i + 1) * interval_size).
    """
    assert left <= right, f"left {left} must be less than or equal to right {right}"

    # sort intervals in descending order
    sorted_intervals = sorted(intervals, reverse=True)
    smallest = sorted_intervals[-1]
    covers = dict()

    # track range of cover
    current_left = None # included
    current_right = None # excluded

    # extend cover with all but the smallest intervals
    # do not exceed range
    for interval_size in sorted_intervals[:-1]:
        # print(f'current cover {current_left} {current_right}')
        # print(f'checking size {interval_size}')

        # check if cover has been initialized
        if current_left is None:
            # attempt to cover
            left_interval = left // interval_size
            right_interval = (right + smallest) // interval_size
            if left_interval * interval_size < left:
                left_interval += 1
            if (right_interval + 1) * interval_size > (right + smallest):
                right_interval -= 1
            
            # print(f'\tpossible bounds: {left_interval} {right_interval}')
            # update if intervals are valid
            if left_interval <= right_interval:
                current_left = left_interval * interval_size
                current_right = (right_interval + 1) * interval_size
                covers[interval_size] = [[left_interval, right_interval]]
            continue
        
        covers_level = []

        if current_left > left:
            # extend cover on the left
            left_interval = left // interval_size
            right_interval = current_left // interval_size - 1
            if left_interval * interval_size < left:
                left_interval += 1
            # print(f'\tpossible left bound: {left_interval}')
        
            # update if intervals are valid
            if left_interval <= right_interval:
                current_left = left_interval * interval_size
                covers_level.append([left_interval, right_interval])

        if current_right <= (right + smallest):
            # extend cover on the right
            left_interval = current_right // interval_size
            right_interval = (right + smallest) // interval_size
            if (right_interval + 1) * interval_size > (right + smallest):
                right_interval -= 1
            # print(f'\tpossible right bound: {right_interval}')
            
            # update if intervals are valid
            if left_interval <= right_interval:
                current_right = (right_interval + 1) * interval_size
                covers_level.append([left_interval, right_interval])
        
        # check if nothing was added
        # print(f'\passed covers: {covers_level}')
        if covers_level:
            covers[interval_size] = covers_level

    # print(f'current cover {current_left} {current_right}')

    # finally, cover rest of range with smallest interval size
    # allowed to exceed range
    interval_size = sorted_intervals[-1]
    covers_level = []
    if current_left is None and current_right is None:
        # cover entire range
        interval_size = sorted_intervals[-1]
        # print(f'covering range with smallest interval size ({interval_size})')

        left_interval = left // interval_size
        right_interval = right // interval_size

        covers[interval_size] = [[left_interval, right_interval]]
    else:
        if current_left > left:
            # extend cover on left
            left_interval = left // interval_size
            right_interval = current_left // interval_size - 1
            covers_level.append([left_interval, right_interval])
        if current_right is not None and current_right <= right:
            # extend cover on right
            left_interval = current_right // interval_size
            right_interval = right // interval_size
            covers_level.append([left_interval, right_interval])
    
    # print(f'checking size {interval_size}\n\thas covers: {covers_level}')
    if covers_level:
        covers[interval_size] = covers_level

    return covers

class CountSketch(Sketch):
    def __init__(self, data:pd.Series, depth:int, width:int, sign_hash:object, bin_hash:object, intervals:list = None, **kwargs):
        self.depth = depth
        self.width = width
        self.nrows = len(data)
        self.bin_hash = bin_hash
        self.sign_hash = sign_hash
        self.sorted_intervals = tuple(sorted(intervals, reverse=True)) if intervals is not None else (1,)

        # save type of data elements
        self.type = data.dtype.type
        self.is_datetime = pd.api.types.is_datetime64_any_dtype(data)

        # check if type is a pandas datetime
        if self.is_datetime:
            # convert to int (expected to be nanoseconds since epoch)
            data = data.view('int64')

        # require that datatype is numeric
        assert pd.api.types.is_numeric_dtype(data), f"CountSketch only supports numeric data types, not {self.type}"

        # save bounds of data (excludes NaN values)
        self.min = data.min()
        self.max = data.max()

        self.column = data.name

        # creates a dataframe with only distinct rows and their counts
        distincts = data.value_counts(dropna=False).sort_values(ascending=False).reset_index(name='_count')
        assert distincts['_count'].sum() == self.nrows

        # vectorized function to convert each element to an int
        self.vhash = np.vectorize(hash)

        # create sketches for each interval
        self.sketches = dict()
        mask = distincts[self.column].notnull().values[None, :]
        for interval in self.sorted_intervals:
            # create a sketch for the interval
            # values = (distincts[self.column] // interval).map(hash).values + 1
            values = self.vhash(distincts[self.column] // interval) + 1
            bins = self.bin_hash(values) % self.width
            signs = self.sign_hash(values) * mask

            # scale update by frequency of each value
            signs *= torch.tensor(distincts['_count'].values)[None, :].expand_as(bins)

            # print(f"values {values.shape} mask {mask.shape} signs {signs[0].shape} bins {bins[0].shape}")
            self.sketches[interval] = torch.zeros((self.depth, self.width), dtype=torch.long).scatter_add_(1, bins, signs)

        self.memory = self.memory_usage()

    def memory_usage(self):
        nbytes = 0
        for sketch in self.sketches.values():
            if sketch.is_sparse:
                indices = sketch.indices()
                nbytes += indices.nelement() * indices.element_size()
                values = sketch.values()
                nbytes += values.nelement() * values.element_size()
            else:
                nbytes += sketch.numel() * sketch.element_size()
        return nbytes
    
    def __call__(self, predicates:dict, keys:dict, *args, **kwargs):
        """
        returns:
            the Count Sketch selectivity estimate of the predicates (float)
        """
        col_in_preds = self.column in predicates.keys()

        if not col_in_preds:
            # if no selection is needed and not a join key attribute, return 1
            return 1, 0
        
        # otherwise, filter selection is needed
        
        # find left and right bounds (inclusive) of the predicates
        left = self.min
        right = self.max

        # if both left and right are NaN, return 0
        if pd.isna(left) and pd.isna(right):
            return 0, 0

        epsilon = 1e-6 # some small value
        for op, val in predicates[self.column].items():
            # print(f'left {left} right {right}')
            # print(f'applying {op} {val} ({type(val)})')
            # check if type is a pandas datetime
            # if self.is_datetime and isinstance(val, (str, pd.Timestamp)):
                # convert val (object) to Timestamp to int (nanoseconds since epoch)
                # val = pd.to_datetime(val).value
            # else:
                # try to convert val to the same type as the data
                # val = self.type(val)
            # print(f'\tval {val} ({type(val)})')
            # require that datatype is numeric
            assert pd.api.types.is_numeric_dtype(type(val)), f"CountSketch only supports numeric data types, not {val.dtype.type}"
            
            # determine the left and right bounds of the predicates
            if op in ('==', '='):
                if val < left or val > right:
                    return 0, 0
                left = right = val
            elif op in ('!=', '<>'):
                if val == left == right:
                    return 0, 0
            elif op == '>=':
                # check if potential left bound is invalid
                if val > right:
                    return 0, 0
                left = max(left, val)
            elif op == '<=':
                # check if potential right bound is invalid
                if val < left:
                    return 0, 0
                right = min(right, val)
            elif op == '>':
                # check if potential left bound is invalid
                if val >= right:
                    return 0, 0
                # shift by epsilon to include left bound
                left = max(left, val+epsilon)
            elif op == '<':
                # check if potential right bound is invalid
                if val <= left:
                    return 0, 0
                # shift by epsilon to include right bound
                right = min(right, val-epsilon)
            else:
                raise NotImplementedError(f"Predicate operator {op} not supported for CountSketch")
        # print(f'left {left} right {right}')

        # calculate the intervals to check
        covers = calculate_intervals(left, right, self.sorted_intervals)

        # determine size of interval array
        total_intervals = 0
        intervals_per_level = dict()
        for interval_size, ranges in covers.items():
            intervals_per_level[interval_size] = int(sum(right-left+1 for left, right in ranges))
            total_intervals += intervals_per_level[interval_size]

        # initialize array
        intervals = np.zeros(total_intervals, dtype=np.int64)

        # populate array with intervals
        current_idx = 0
        for interval_size, ranges in covers.items():
            for left, right in ranges:
                next_idx = int(current_idx + right - left + 1)
                intervals[current_idx:next_idx] = np.arange(left, right+1)
                current_idx = next_idx
        
        # hash the intervals
        # intervals = list(map(lambda x: hash(x) + 1, intervals))
        intervals = self.vhash(intervals) + 1

        # sketch the query intervals
        bins = self.bin_hash(intervals) % self.width
        signs = self.sign_hash(intervals)

        # for each interval size, compute the cardinality
        est = 0
        bins_idx = 0
        for interval_size, num_intervals in intervals_per_level.items():
            # get view of the bins and signs corresponding to the current interval size
            bins_intervals = bins[:, bins_idx:bins_idx + num_intervals]
            signs_intervals = signs[:, bins_idx:bins_idx + num_intervals]
            bins_idx += num_intervals

            # gather the count estimates from the sketch
            counts = self.sketches[interval_size].gather(1, bins_intervals) * signs_intervals

            # aggregate the median sketch estimate
            est += counts.sum(dim=1).median().item()
        
        assert bins_idx == total_intervals, \
            f'Number of intervals ({bins_idx}) counted mismatch with expected size of the cover ({total_intervals})'    
        # return the selectivity estimate (within [0, 1])
        prob = min(max(0, est / self.nrows), 1)
        return prob, 0

class CountMin(Sketch):
    def __init__(self, data:pd.Series, depth:int, width:int, bin_hash:object, intervals:list = None, **kwargs):
        self.depth = depth
        self.width = width
        self.nrows = len(data)
        self.bin_hash = bin_hash
        self.sorted_intervals = tuple(sorted(intervals, reverse=True)) if intervals is not None else (1,)

        # save type of data elements
        self.type = data.dtype.type
        self.is_datetime = pd.api.types.is_datetime64_any_dtype(data)

        # check if type is a pandas datetime
        if self.is_datetime:
            # convert to int (expected to be nanoseconds since epoch)
            data = data.view('int64')

        # require that datatype is numeric
        assert pd.api.types.is_numeric_dtype(data), f"CountSketch only supports numeric data types, not {self.type}"

        # save bounds of data (excludes NaN values)
        self.min = data.min()
        self.max = data.max()

        self.column = data.name

        # creates a dataframe with only distinct rows and their counts
        distincts = data.value_counts(dropna=False).sort_values(ascending=False).reset_index(name='_count')
        assert distincts['_count'].sum() == self.nrows

        # vectorized function to convert each element to an int
        self.vhash = np.vectorize(hash)

        # create sketches for each interval
        self.sketches = dict()
        mask = distincts[self.column].notnull().values
        for interval in self.sorted_intervals:
            # create a sketch for the interval
            values = self.vhash(distincts[self.column] // interval) + 1
            bins = self.bin_hash(values) % self.width

            # scale update by frequency of each non-null value
            counts = torch.tensor(distincts['_count'].values * mask)[None, :].expand_as(bins)

            # print(f"values {values.shape} mask {mask.shape} signs {signs[0].shape} bins {bins[0].shape}")
            self.sketches[interval] = torch.zeros((self.depth, self.width), dtype=torch.long).scatter_add_(1, bins, counts)

        self.memory = self.memory_usage()

    def memory_usage(self):
        nbytes = 0
        for sketch in self.sketches.values():
            if sketch.is_sparse:
                indices = sketch.indices()
                nbytes += indices.nelement() * indices.element_size()
                values = sketch.values()
                nbytes += values.nelement() * values.element_size()
            else:
                nbytes += sketch.numel() * sketch.element_size()
        return nbytes
    
    def __call__(self, predicates:dict, keys:dict, *args, **kwargs):
        """
        returns:
            the Count Min selectivity estimate of the predicates (float)
        """
        col_in_preds = self.column in predicates.keys()

        if not col_in_preds:
            # if no selection is needed and not a join key attribute, return 1
            return 1, 0
        
        # otherwise, filter selection is needed
        
        # find left and right bounds (inclusive) of the predicates
        left = self.min
        right = self.max

        # if both left and right are NaN, return 0
        if pd.isna(left) and pd.isna(right):
            return 0, 0

        epsilon = 1e-6 # some small value
        for op, val in predicates[self.column].items():
            # print(f'left {left} right {right}')
            # print(f'applying {op} {val} ({type(val)})')
            # check if type is a pandas datetime
            # if self.is_datetime:
                # convert val (object) to Timestamp to int (nanoseconds since epoch)
                # val = pd.to_datetime(val).value
            # else:
                # try to convert val to the same type as the data
                # val = self.type(val)
            # print(f'\tval {val} ({type(val)})')
            # require that datatype is numeric
            assert pd.api.types.is_numeric_dtype(type(val)), f"CountMin only supports numeric data types, not {type(val)}"
            
            # determine the left and right bounds of the predicates
            if op in ('==', '='):
                if val < left or val > right:
                    return 0, 0
                left = right = val
            elif op in ('!=', '<>'):
                if val == left == right:
                    return 0, 0
            elif op == '>=':
                # check if potential left bound is invalid
                if val > right:
                    return 0, 0
                left = max(left, val)
            elif op == '<=':
                # check if potential right bound is invalid
                if val < left:
                    return 0, 0
                right = min(right, val)
            elif op == '>':
                # check if potential left bound is invalid
                if val >= right:
                    return 0, 0
                # shift bound by epsilon to include left bound
                left = max(left, val+epsilon)
            elif op == '<':
                # check if potential right bound is invalid
                if val <= left:
                    return 0, 0
                # shift bound by epsilon to include right bound
                right = min(right, val-epsilon)
            else:
                raise NotImplementedError(f"Operator {op} not supported for CountMin")
        # print(f'left {left} right {right}')

        # calculate the query intervals
        covers = calculate_intervals(left, right, self.sorted_intervals)

        # determine size of interval array
        total_intervals = 0
        intervals_per_level = dict()
        for interval_size, ranges in covers.items():
            intervals_per_level[interval_size] = int(sum(right-left+1 for left, right in ranges))
            total_intervals += intervals_per_level[interval_size]

        # initialize array
        intervals = np.zeros(total_intervals, dtype=np.int64)

        # populate array with intervals
        current_idx = 0
        for interval_size, ranges in covers.items():
            for left, right in ranges:
                next_idx = int(current_idx + right - left + 1)
                intervals[current_idx:next_idx] = np.arange(left, right+1)
                current_idx = next_idx
        
        # hash the intervals
        # intervals = list(map(lambda x: hash(x) + 1, intervals))
        intervals = self.vhash(intervals) + 1

        # sketch the query intervals
        bins = self.bin_hash(intervals) % self.width

        # for each interval size, compute the cardinality
        est = 0
        bins_idx = 0
        for interval_size, num_intervals in intervals_per_level.items():
            # get view of the bins and signs corresponding to the current interval size
            bins_intervals = bins[:, bins_idx:bins_idx + num_intervals]
            bins_idx += num_intervals

            # gather frequencies from the count-min sketch
            counts = self.sketches[interval_size].gather(1, bins_intervals)

            # aggregate the minimum sketch estimate
            est += counts.sum(dim=1).min().item()
            
        # return the selectivity estimate (within [0, 1])
        prob = min(max(0, est / self.nrows), 1)
        return prob, 0