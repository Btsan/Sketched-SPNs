from copy import deepcopy
from time import perf_counter_ns

import numpy as np
import pandas as pd
import torch

from Estimators import CountEstimator, DegreeEstimator

class AMS(object):
    def __init__(self, data:pd.DataFrame, depth:int, sign_hashes:list, bifocal: int = 10, **kwargs):
        self.sketch_method = 'ams'
        self.depth = depth
        self.nrows = len(data)
        self.sign_hashes = sign_hashes
        self.bifocal = max(0, bifocal)
        self.saved = dict()

        self.columns = {data.name,} if isinstance(data, pd.Series) else set(data.columns)

        # creates a dataframe with only distinct rows and their counts
        distincts = data.value_counts(dropna=False).sort_values(ascending=False).reset_index(name='_count')

        self.distincts_hi = distincts[:bifocal]
        self.signs_hi = dict()
        if len(self.distincts_hi):
            for col in self.columns:
                values = self.distincts_hi[col].map(hash).values + 1 # [N]
                mask = self.distincts_hi[col].notnull().values[None, :] # [1, N]
                self.signs_hi[col] = [sign_hash(values) * mask for sign_hash in sign_hashes]
        else:
            for col in self.columns:
                self.signs_hi[col] = [torch.zeros(depth, 1) for _ in sign_hashes]
                
        self.distincts_lo = distincts[bifocal:]
        
        self.signs_lo = dict()
        if len(self.distincts_lo):
            for col in self.columns:
                values = self.distincts_lo[col].map(hash).values + 1 # [N]
                mask = self.distincts_lo[col].notnull().values[None, :] # [1, N]
                self.signs_lo[col] = [sign_hash(values) * mask for sign_hash in sign_hashes]
        else:
            for col in self.columns:
                self.signs_lo[col] = [torch.zeros(depth, 1) for _ in sign_hashes]

        self.memory = self.distincts_hi.memory_usage().sum() + self.distincts_lo.memory_usage().sum()
        for col in self.columns:
            for hashes in self.signs_hi[col]:
                self.memory += hashes.numel() * hashes.element_size()
            for hashes in self.signs_lo[col]:
                self.memory += hashes.numel() * hashes.element_size()

    def memory_usage(self):
        nbytes = 0
        for estimator in self.saved.values():
            nbytes += estimator.memory_usage()
        return nbytes

    def __call__(self, predicates:dict, keys:dict, **kwargs):
        """
        returns:
            the selectivity of the predicates (float) or the sketch of the keys (Estimator)
        """
        col_in_preds = self.columns.intersection(predicates.keys())
        col_in_keys = self.columns.intersection(keys.keys())

        sel_lo = self.distincts_lo
        sel_hi = self.distincts_hi
        if col_in_preds:
            preds = []
            for col in col_in_preds:
                for op, val in predicates[col].items():
                    if op == '=':
                        op = '==' # pandas uses '==' for equality
                    # if sel[col].dtype not in (int, float):
                    if not pd.api.types.is_numeric_dtype(sel_lo[col]):
                        val = f"'{val}'"
                    preds.append(f"`{col}`{op}{val}")
            q = " & ".join(preds)
            sel_lo = sel_lo.query(q)
            sel_hi = sel_hi.query(q)

            if col_in_keys:
                signs_lo = 1
                for col, join_indices in keys.items():
                    values = sel_lo[col].map(hash).values + 1
                    for join_idx in join_indices:
                        signs_lo *= self.sign_hashes[join_idx](values)
                    mask = sel_lo[col].notnull().values[None, :]
                    signs_lo *= mask
                assert signs_lo.shape == (self.depth, max(1, len(sel_lo))), f"{signs_lo.shape} == {(self.depth, len(sel_lo))}"
                signs_lo *= sel_lo['_count'].values[None, :]
                sketch_lo = signs_lo.sum(dim=-1, keepdim=True).float()
                
                if self.bifocal > 0:
                    exact_hi = sel_hi.set_index(sorted(col_in_keys))['_count']
                    signs_hi = 1
                    for col, join_indices in keys.items():
                        values = sel_hi[col].map(hash).values + 1
                        for join_idx in join_indices:
                            signs_hi *= self.sign_hashes[join_idx](values)
                        mask = sel_hi[col].notnull().values[None, :]
                        signs_hi *= mask
                    assert signs_hi.shape == (self.depth, len(sel_hi)), f"{signs_hi.shape} == {(self.depth, len(sel_hi))}"
                    signs_hi *= sel_hi['_count'].values[None, :]
                    sketch_hi = signs_hi.sum(dim=-1, keepdim=True).float()
                    return CountEstimator(sketch_lo, sketch_hi, exact_hi)
                return CountEstimator(sketch_lo)
            
            prob = (sel_lo['_count'].sum() + sel_hi['_count'].sum()) / self.nrows
            return prob, 0
        elif not col_in_keys:
            return 1, 0

        # check if sketch already exists
        sketch_id = frozenset(keys.items())
        if not col_in_preds and sketch_id in self.saved:
            return deepcopy(self.saved[sketch_id]), 0
        
        # measure sketcching time
        t0 = perf_counter_ns()
        
        # sketch infrequent items
        signs_lo = 1
        for key, join_indices in keys.items():
            for join_idx in join_indices:
                signs_lo *= self.signs_lo[key][join_idx]
        assert signs_lo.shape == (self.depth, max(1, len(self.distincts_lo))), f"{signs_lo.shape} == {(self.depth, len(self.distincts_lo))}"
        signs_lo *= self.distincts_lo['_count'].values[None, :]
        sketch_lo = signs_lo.sum(dim=-1, keepdim=True).float()

        # sketch frequent items
        if self.bifocal > 0:
            exact_hi = self.distincts_hi.set_index(sorted(col_in_keys))['_count']
            signs_hi = 1
            for key, join_indices in keys.items():
                for join_idx in join_indices:
                    signs_hi *= self.signs_hi[key][join_idx]
            assert signs_hi.shape == (self.depth, len(self.distincts_hi)), f"{signs_hi.shape} == {(self.depth, len(self.distincts_hi))}"
            signs_hi *= self.distincts_hi['_count'].values[None, :]
            sketch_hi = signs_hi.sum(dim=-1, keepdim=True).float()
            estimator =  CountEstimator(sketch_lo, sketch_hi, exact_hi)
        else:
            estimator = CountEstimator(sketch_lo)
        
        t1 = perf_counter_ns()
        sketch_time = (t1 - t0)

        # save sketch for reuse, if there were no predicates
        if not col_in_preds:
            self.saved[sketch_id] = estimator
        return deepcopy(estimator), sketch_time

class CountSketch(object):
    def __init__(self, data:pd.DataFrame, depth:int, width:int, sign_hashes:list, bin_hashes:list, bifocal: int = 10, **kwargs):
        self.sketch_method = 'count-sketch'
        self.depth = depth
        self.width = width
        self.nrows = len(data)
        self.sign_hashes = sign_hashes
        self.bin_hashes = bin_hashes
        self.bifocal = max(0, bifocal)

        self.columns = {data.name,} if isinstance(data, pd.Series) else set(data.columns)

        # creates a dataframe with only distinct rows and their counts
        distincts = data.value_counts(dropna=False).sort_values(ascending=False).reset_index(name='_count')

        self.distincts_hi = distincts[:bifocal]
        self.signs_hi = {}
        self.bins_hi = {}
        if len(self.distincts_hi):
            for col in self.columns:
                values = self.distincts_hi[col].map(hash).values + 1 # [N]
                mask = self.distincts_hi[col].notnull().values[None, :] # [1, N]
                self.signs_hi[col] = [sign_hash(values) * mask for sign_hash in sign_hashes]
                self.bins_hi[col] = [bin_hash(values) for bin_hash in bin_hashes]
        else:
            for col in self.columns:
                self.signs_hi[col] = [torch.zeros(depth, 1) for _ in sign_hashes]
                self.bins_hi[col] = [torch.zeros(depth, 1) for _ in bin_hashes]
                
        self.distincts_lo = distincts[bifocal:]
        
        self.signs_lo = {}
        self.bins_lo = {}
        if len(self.distincts_lo):
            for col in self.columns:
                values = self.distincts_lo[col].map(hash).values + 1 # [N]
                mask = self.distincts_lo[col].notnull().values[None, :] # [1, N]
                self.signs_lo[col] = [sign_hash(values) * mask for sign_hash in sign_hashes]
                self.bins_lo[col] = [bin_hash(values) for bin_hash in bin_hashes]
        else:
            for col in self.columns:
                self.signs_lo[col] = [torch.zeros(depth, 1) for _ in sign_hashes]
                self.bins_lo[col] = [torch.zeros(depth, 1) for _ in bin_hashes]

        self.saved = dict()
        self.memory = self.distincts_hi.memory_usage().sum() + self.distincts_lo.memory_usage().sum()
        for col in self.columns:
            for hashes in self.signs_hi[col]:
                self.memory += hashes.numel() * hashes.element_size()
            for hashes in self.signs_lo[col]:
                self.memory += hashes.numel() * hashes.element_size()
            for hashes in self.bins_hi[col]:
                self.memory += hashes.numel() * hashes.element_size()
            for hashes in self.bins_lo[col]:
                self.memory += hashes.numel() * hashes.element_size()
        
        # memory usage of pushdown (exact) sketches
        self.pushdown = dict()

        # Count-Min for predicate selectivity # just one hash suffices
        self.countmins = {}
        for col in self.columns:
            values = distincts[col].map(hash).values + 1 # N
            mask = distincts[col].notnull().values[None, :] # 1, N
            # bins = torch.concatenate([bin_hash(values) for bin_hash in bin_hashes], dim=0)
            bins = bin_hashes[0](values) % self.width
            counts = torch.tensor(distincts['_count'].values)[None, :].expand_as(bins)
            counts *= mask # don't count nulls
            # assert bins.shape == counts.shape == (self.depth * len(bin_hashes), len(distincts)), \
            #     f"{bins.shape} == {counts.shape} == {self.depth * len(bin_hashes), len(distincts)}"
            assert bins.shape == counts.shape == (self.depth, len(distincts)), \
                f"{bins.shape} == {counts.shape} == {self.depth, len(distincts)}"
            # print(f"\n{col} {distincts['_count']}  counts {counts}")
            # print(f"\nvalues {values} bins {bins}")
            # self.countmins[col] = torch.zeros((self.depth * len(bin_hashes), self.width), dtype=torch.long).scatter_add_(1, bins, counts)
            self.countmins[col] = torch.zeros((self.depth, self.width), dtype=torch.long).scatter_add_(1, bins, counts)

    def memory_usage(self):
        nbytes = sum(self.pushdown.values())
        for estimator in self.saved.values():
            nbytes += estimator.memory_usage()
        return nbytes
    
    def __call__(self, predicates:dict, keys:dict, components:dict, cuda : bool = False, exact_prob : bool = False, **kwargs):
        """
        returns:
            the selectivity of the predicates (float) or the sketch of the keys (Estimator)
        """
        col_in_preds = self.columns.intersection(predicates.keys())
        col_in_keys = self.columns.intersection(keys.keys())

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
                    if not pd.api.types.is_numeric_dtype(self.distincts_lo[col]):
                        val = f"'{val}'"
                    preds.append(f"`{col}`{op}{val}")
            q = " & ".join(preds)
            sel_lo = self.distincts_lo.query(q)
            sel_hi = self.distincts_hi.query(q)
            # print(f"{q} --> {len(sel_lo)}/{len(self.distincts_lo)} {len(sel_hi)}/{len(self.distincts_hi)}")

            if col_in_keys:
                # pushdown sketch of infrequent items
                t0 = perf_counter_ns()
                sketch_lo = torch.zeros((self.depth, self.width), dtype=torch.float)
                if len(sel_lo) > 0:
                    sel_lo = sel_lo.groupby(list(keys.keys())).sum('_count').reset_index()
                    signs_lo = 1
                    bins_lo = 0
                    for key, join_indices in keys.items():
                        values = sel_lo[key].map(hash).values + 1
                        bins_lo += self.bin_hashes[components[key]](values)
                        for join_idx in join_indices:
                            signs_lo *= self.sign_hashes[join_idx](values)
                        # mask = sel_lo[key].notnull().values[None, :] # [1, N]
                        # signs_lo *= mask
                    assert bins_lo.shape == signs_lo.shape == (self.depth, max(1, len(sel_lo))), f"{bins_lo.shape} == {signs_lo.shape} == {(self.depth, len(sel_lo))}"
                    bins_lo %= self.width
                    signs_lo *= sel_lo['_count'].values[None, :]

                    # assert bins_lo.dtype == torch.int64, f"bins_lo {bins_lo.dtype} {bins_lo.shape} {bins_lo}\nsigns_lo {signs_lo.dtype} {signs_lo.shape} {signs_lo}\ndistincts_lo {self.distincts_lo}"
                    sketch_lo.view(self.depth, -1).scatter_add_(1, bins_lo.long(), signs_lo.float())

                if self.bifocal > 0:
                    exact_hi = sel_hi.groupby(list(keys.keys())).sum('_count')['_count']
                    sel_hi = exact_hi.reset_index()
                    sketch_hi = torch.zeros((self.depth, self.width), dtype=torch.long)
                    if len(sel_hi) > 0:
                        signs_hi = 1
                        bins_hi = 0
                        for key, join_indices in keys.items():
                            values = sel_hi[key].map(hash).values + 1 # [N]
                            bins_hi += self.bin_hashes[components[key]](values) # [depth, N]
                            for join_idx in join_indices:
                                signs_hi *= self.sign_hashes[join_idx](values) # [depth, N]
                            # mask = sel_hi[key].notnull().values[None, :] # [1, N]
                            # signs_hi *= mask
                        assert bins_hi.shape == signs_hi.shape == (self.depth, len(sel_hi)), f"{bins_hi.shape} == {signs_hi.shape} == {(self.depth, len(sel_hi))}"
                        bins_hi %= self.width
                        signs_hi *= sel_hi['_count'].values[None, :]
                        sketch_hi = sketch_hi.scatter_add_(1, bins_hi, signs_hi).float()
                    estimator = CountEstimator(sketch_lo, sketch_hi, exact_hi)
                else:
                    estimator = CountEstimator(sketch_lo)
                t1 = perf_counter_ns()
                sketch_time = (t1 - t0)
                if cuda:
                    return estimator.cuda(), sketch_time
                pushdown_id = sketch_id.union(preds)
                self.pushdown[pushdown_id] += estimator.memory_usage()
                return estimator, sketch_time
            else:
                # return probability if not a join key attribute
                if exact_prob:
                    prob = (sel_lo['_count'].sum() + sel_hi['_count'].sum()) / self.nrows
                else:
                    # convert to count-min probability
                    # freq = torch.zeros((self.depth * len(self.bin_hashes), self.width), dtype=torch.long)
                    freq = torch.zeros((self.depth, self.width), dtype=torch.long)
                    for col in col_in_preds:
                        # cm = torch.zeros((self.depth * len(self.bin_hashes), self.width), dtype=torch.long)
                        cm = torch.zeros((self.depth, self.width), dtype=torch.long)
                        if len(sel_lo) > 0:
                            values = sel_lo[col].map(hash).values + 1
                            # bins = torch.concatenate([bin_hash(values) for bin_hash in self.bin_hashes], dim=0) # depth * b, N
                            bins = self.bin_hashes[0](values) # depth, N
                            cm = cm.scatter_add(1, bins, torch.ones(1, dtype=torch.long).expand_as(bins))
                        if len(sel_hi) > 0:
                            values = sel_lo[col].map(hash).values + 1
                            # bins = torch.concatenate([bin_hash(values) for bin_hash in self.bin_hashes], dim=0) # depth * b, N
                            bins = self.bin_hashes[0](values) # depth, N
                            cm = cm.scatter_add(1, bins, torch.ones(1, dtype=torch.long).expand_as(bins))
                        freq += self.countmins[col] * (cm > 0)
                    prob = freq.sum(dim=-1).min().item() / self.nrows
                return prob, 0

        # check if sketch already exists
        if not col_in_preds and sketch_id in self.saved:
            if cuda:
                return self.saved[sketch_id].cuda(), 0
            return deepcopy(self.saved[sketch_id]), 0
        
        # measure sketcching time
        t0 = perf_counter_ns()
        
        # sketch infrequent items
        sketch_lo = torch.zeros((self.depth, self.width), dtype=torch.float)
        if len(self.distincts_lo) > 0:
            signs_lo = 1
            bins_lo = 0
            for key, join_indices in keys.items():
                bins_lo += self.bins_lo[key][components[key]]
                for join_idx in join_indices:
                    signs_lo *= self.signs_lo[key][join_idx]
            assert bins_lo.shape == signs_lo.shape == (self.depth, max(1, len(self.distincts_lo))), f"{bins_lo.shape} == {signs_lo.shape} == {(self.depth, len(self.distincts_lo))}"
            bins_lo %= self.width
            signs_lo *= self.distincts_lo['_count'].values[None, :]

            sketch_lo.view(self.depth, -1).scatter_add_(1, bins_lo.long(), signs_lo.float())

        # sketch frequent items
        if self.bifocal > 0:
            exact_hi = self.distincts_hi.set_index(sorted(col_in_keys))['_count']
            signs_hi = 1
            bins_hi = 0
            for key, join_indices in keys.items():
                bins_hi += self.bins_hi[key][components[key]]
                for join_idx in join_indices:
                    signs_hi *= self.signs_hi[key][join_idx]
            assert bins_hi.shape == signs_hi.shape == (self.depth, len(self.distincts_hi)), f"{bins_hi.shape} == {signs_hi.shape} == {(self.depth, len(self.distincts_hi))}"
            bins_hi %= self.width
            signs_hi *= self.distincts_hi['_count'].values[None, :]
            sketch_hi = torch.zeros((self.depth, self.width), dtype=torch.long)
            sketch_hi = sketch_hi.scatter_add_(1, bins_hi, signs_hi).float()
            estimator = CountEstimator(sketch_lo, sketch_hi, exact_hi)
        else:
            estimator =  CountEstimator(sketch_lo)

        t1 = perf_counter_ns()
        sketch_time = (t1 - t0)

        if not col_in_preds:
            self.saved[sketch_id] = estimator
        if cuda:
            return estimator.cuda(), sketch_time
        return deepcopy(estimator), sketch_time
    
class BoundSketch(object):
    def __init__(self, data:pd.DataFrame, depth:int, width:int, bin_hashes:list, bifocal: int = 10, **kwargs):
        self.sketch_method = 'bound-sketch'
        self.depth = depth
        self.width = width
        self.nrows = len(data)
        self.bin_hashes = bin_hashes
        self.bifocal = max(0, bifocal)

        self.columns = {data.name,} if isinstance(data, pd.Series) else set(data.columns)

        # creates a dataframe with only distinct rows and their counts
        distincts = data.value_counts(dropna=False).sort_values(ascending=False).reset_index(name='_count')
        assert distincts['_count'].sum() == self.nrows

        self.distincts_hi = distincts[:bifocal]
        self.distincts_lo = distincts[bifocal:]
        self.saved = dict()

        self.memory = self.distincts_hi.memory_usage().sum() + self.distincts_lo.memory_usage().sum()

        self.pushdown = dict()

        # Count-Min for predicate selectivity
        # Count-Min for predicate selectivity # just one hash suffices
        self.countmins = {}
        for col in self.columns:
            values = distincts[col].map(hash).values + 1 # N
            mask = distincts[col].notnull().values[None, :] # 1, N
            # bins = torch.concatenate([bin_hash(values) for bin_hash in bin_hashes], dim=0)
            bins = bin_hashes[0](values) % self.width
            counts = torch.tensor(distincts['_count'].values)[None, :].expand_as(bins)
            counts *= mask # don't count nulls
            # assert bins.shape == counts.shape == (self.depth * len(bin_hashes), len(distincts)), \
            #     f"{bins.shape} == {counts.shape} == {self.depth * len(bin_hashes), len(distincts)}"
            assert bins.shape == counts.shape == (self.depth, len(distincts)), \
                f"{bins.shape} == {counts.shape} == {self.depth, len(distincts)}"
            # self.countmins[col] = torch.zeros((self.depth * len(bin_hashes), self.width), dtype=torch.long).scatter_add_(1, bins, counts)
            self.countmins[col] = torch.zeros((self.depth, self.width), dtype=torch.long).scatter_add_(1, bins, counts)
            
    def memory_usage(self):
        nbytes = sum(self.pushdown.values())
        for estimator in self.saved.values():
            nbytes += estimator.memory_usage()
        return nbytes
    
    def __call__(self, predicates:dict, keys:dict, components:dict, count: bool = True, cuda: bool = False, exact_prob=False, **kwargs):
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
                    if not pd.api.types.is_numeric_dtype(self.distincts_lo[col]):
                        val = f"'{val}'"
                    preds.append(f"`{col}`{op}{val}")
            q = " & ".join(preds)
            sel_lo = self.distincts_lo.query(q)
            sel_hi = self.distincts_hi.query(q)
            # print(f"{q} --> {len(sel_lo)}/{len(self.distincts_lo)} {len(sel_hi)}/{len(self.distincts_hi)}")
            # return probability if not a join key attribute
            if not col_in_keys:
                if exact_prob:
                    prob = (sel_lo['_count'].sum() + sel_hi['_count'].sum()) / self.nrows
                else:
                    # convert to count-min probability
                    # freq = torch.zeros((self.depth * len(self.bin_hashes), self.width), dtype=torch.long)
                    freq = torch.zeros((self.depth, self.width), dtype=torch.long)
                    for col in col_in_preds:
                        # cm = torch.zeros((self.depth * len(self.bin_hashes), self.width), dtype=torch.long)
                        cm = torch.zeros((self.depth, self.width), dtype=torch.long)
                        if len(sel_lo) > 0:
                            values = sel_lo[col].map(hash).values + 1
                            # bins = torch.concatenate([bin_hash(values) for bin_hash in self.bin_hashes], dim=0) # depth * b, N
                            bins = self.bin_hashes[0](values) # depth, N
                            cm = cm.scatter_add(1, bins, torch.ones(1, dtype=torch.long).expand_as(bins))
                        if len(sel_hi) > 0:
                            values = sel_lo[col].map(hash).values + 1
                            # bins = torch.concatenate([bin_hash(values) for bin_hash in self.bin_hashes], dim=0) # depth * b, N
                            bins = self.bin_hashes[0](values) # depth, N
                            cm = cm.scatter_add(1, bins, torch.ones(1, dtype=torch.long).expand_as(bins))
                        freq += self.countmins[col] * (cm > 0)
                    prob = freq.sum(dim=-1).min().item() / self.nrows
                return prob, 0
        else:
            # otherwise no filters are applied and proceed to sketching
            sel_lo = self.distincts_lo
            sel_hi = self.distincts_hi

        # check if sketch already exists
        sketch_id = frozenset(keys.keys()).union(components.items()).union({('count', count)})
        if not col_in_preds and sketch_id in self.saved:
            estimator = self.saved[sketch_id]
            if cuda:
                return estimator.cuda(), 0
            return deepcopy(estimator), 0
        
        # measure sketcching time
        t0 = perf_counter_ns()

        # group by join keys
        sel_lo = sel_lo.groupby(list(keys.keys())).sum('_count').reset_index()
        sel_hi = sel_hi.groupby(list(keys.keys())).sum('_count').reset_index()

        sketch_lo = torch.zeros((self.depth, self.width), dtype=torch.float)
        if len(sel_lo) > 0:
            bins_lo = 0
            for key, _ in keys.items():
                values = sel_lo[key].map(hash).values + 1
                bins_lo += self.bin_hashes[components[key]](values)
            bins_lo %= self.width
            counts_lo = torch.tensor(sel_lo['_count'].values)
            counts_lo = counts_lo[None, :].expand_as(bins_lo)
            assert bins_lo.shape == counts_lo.shape == (self.depth, max(1, len(sel_lo))), f"{bins_lo.shape} == {counts_lo.shape} == {(self.depth, len(sel_lo))}"

            sketch_lo.view(self.depth, -1).scatter_reduce_(1, bins_lo.long(), counts_lo.float(), reduce_mode)
        
        if self.bifocal > 0:
            exact_hi = sel_hi.set_index(sorted(col_in_keys))['_count']
            sketch_hi = torch.zeros((self.depth, self.width), dtype=torch.long)
            if len(sel_hi) > 0:
                bins_hi = 0
                for col, _ in keys.items():
                    values = sel_hi[col].map(hash).values + 1
                    bins_hi += self.bin_hashes[components[col]](values)
                counts_hi = torch.tensor(sel_hi['_count'].values)
                counts_hi = counts_hi[None, :].expand_as(bins_hi)
                assert bins_hi.shape == counts_hi.shape == (self.depth, len(sel_hi)), f"{bins_hi.shape} == {counts_hi.shape} == {(self.depth, len(sel_hi))}"
                bins_hi %= self.width
                sketch_hi = sketch_hi.scatter_reduce_(1, bins_hi, counts_hi, reduce_mode).float()
            estimator = CountEstimator(sketch_lo, sketch_hi, exact_hi) if count else DegreeEstimator(sketch_lo, sketch_hi, exact_hi)
        else:
            estimator = CountEstimator(sketch_lo) if count else DegreeEstimator(sketch_lo)

        t1 = perf_counter_ns()
        sketch_time = (t1 - t0)
        
        # save sketch for reuse, if there were no predicates
        if not col_in_preds:
            self.saved[sketch_id] = estimator
        else:
            pushdown_id = sketch_id.union(preds)
            self.pushdown[pushdown_id] += estimator.memory_usage()
        if cuda:
            return estimator.cuda(), sketch_time
        return deepcopy(estimator), sketch_time