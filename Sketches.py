from time import perf_counter_ns
import re
from typing import List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch

from sqlglot import exp

# from Estimators import CountEstimator, DegreeEstimator

class Sketch(object):
    """Base class for sketches."""

    def memory_usage(self):
        return 0

    def __call__(self, predicates: exp.Expression, keys:dict, **kwargs):
        """
        returns:
            the selectivity of the predicates (float) or the sketch of the keys (Tensor)
        """
        raise NotImplementedError("Subclasses should implement this method.")

    # ========================================================================
    # HELPER METHODS FOR SQLGLOT EXPRESSIONS
    # ========================================================================
    
    def _get_columns_from_expr(self, expr: exp.Expression) -> set:
        """Extract all column names from sqlglot expression."""
        columns = set()
        if expr is not None:
            for col in expr.find_all(exp.Column):
                columns.add(col.name)
        return columns
    
    def _filter_with_expression(self, expr: exp.Expression) -> pd.DataFrame:
        """
        Filter distincts DataFrame using sqlglot expression.
        
        Converts sqlglot expression to pandas query string.
        Handles AND, OR, and comparison operators.
        
        Args:
            expr: sqlglot expression
        
        Returns:
            Filtered DataFrame
        """
        if expr is None:
            return self.distincts
        
        # Convert sqlglot expression to pandas query string
        query_str = self._expr_to_pandas_query(expr)
        
        if query_str:
            print(query_str)
            return self.distincts.query(query_str)
        else:
            return self.distincts
    
    def _expr_to_pandas_query(self, expr: exp.Expression) -> str:
        """
        Convert sqlglot expression to pandas query string.
        
        Recursively processes expression tree to build query string.
        
        Args:
            expr: sqlglot expression
        
        Returns:
            Pandas query string (e.g., "`salary` > 50000 & `age` < 40")
        """
        if isinstance(expr, exp.And):
            # Recursively process AND children
            left = self._expr_to_pandas_query(expr.this)
            right = self._expr_to_pandas_query(expr.expression)
            return f"({left}) & ({right})"
        
        elif isinstance(expr, exp.Or):
            # Recursively process OR children
            left = self._expr_to_pandas_query(expr.this)
            right = self._expr_to_pandas_query(expr.expression)
            return f"({left}) | ({right})"
        
        elif isinstance(expr, exp.EQ):
            return self._comparison_to_query(expr, '==')
        
        elif isinstance(expr, exp.GT):
            return self._comparison_to_query(expr, '>')
        
        elif isinstance(expr, exp.GTE):
            return self._comparison_to_query(expr, '>=')
        
        elif isinstance(expr, exp.LT):
            return self._comparison_to_query(expr, '<')
        
        elif isinstance(expr, exp.LTE):
            return self._comparison_to_query(expr, '<=')
        
        elif isinstance(expr, exp.NEQ):
            return self._comparison_to_query(expr, '!=')
        
        elif isinstance(expr, exp.In):
            # Handle IN predicate
            col = expr.this
            values = expr.expressions
            
            if isinstance(col, exp.Column):
                col_name = col.name
                # Convert to OR of equalities
                or_clauses = []
                for val in values:
                    if isinstance(val, exp.Literal):
                        val_str = self._format_value(col_name, val.this)
                        or_clauses.append(f"`{col_name}` == {val_str}")
                return " | ".join(f"({clause})" for clause in or_clauses)
        
        elif isinstance(expr, exp.Like):
            # Handle LIKE predicate
            col = expr.this
            pattern = expr.expression
            
            if isinstance(col, exp.Column) and isinstance(pattern, exp.Literal):
                col_name = col.name
                pattern_str = pattern.this
                # Convert SQL LIKE to pandas string contains
                # This is a simplified conversion
                if pattern_str.startswith('%') and pattern_str.endswith('%'):
                    # %pattern% -> contains
                    substr = pattern_str[1:-1]
                    return f"`{col_name}`.str.contains('{substr}', na=False)"
                elif pattern_str.startswith('%'):
                    # %pattern -> endswith
                    suffix = pattern_str[1:]
                    return f"`{col_name}`.str.endswith('{suffix}', na=False)"
                elif pattern_str.endswith('%'):
                    # pattern% -> startswith
                    prefix = pattern_str[:-1]
                    return f"`{col_name}`.str.startswith('{prefix}', na=False)"
                else:
                    # Exact match
                    return f"`{col_name}` == '{pattern_str}'"
        
        elif isinstance(expr, exp.Between):
            # Handle BETWEEN predicate
            col = expr.this
            low = expr.args.get('low')
            high = expr.args.get('high')
            
            if isinstance(col, exp.Column):
                col_name = col.name
                low_val = self._format_value(col_name, low.this) if isinstance(low, exp.Literal) else str(low)
                high_val = self._format_value(col_name, high.this) if isinstance(high, exp.Literal) else str(high)
                return f"(`{col_name}` >= {low_val}) & (`{col_name}` <= {high_val})"
        
        # Unknown expression type - return empty string
        return ""
    
    def _comparison_to_query(self, expr: exp.Expression, op: str) -> str:
        """Convert comparison expression to pandas query string."""
        left = expr.left
        right = expr.right
        
        print(left, type(left), right, type(right))
        if isinstance(left, exp.Column):
            col_name = left.name
            
            # Handle different right-hand side types
            if isinstance(right, exp.Literal):
                # Simple literal value
                value = right.this
                value_str = self._format_value(col_name, value)
            
            elif isinstance(right, exp.Cast):
                # ✅ Cast expression (e.g., timestamp)
                value_str = self._handle_cast_value(col_name, right)
            
            else:
                # Unknown type
                value_str = str(right)
            
            return f"`{col_name}` {op} {value_str}"
        
        return ""
    
    def _handle_cast_value(self, col_name: str, cast_expr: exp.Cast) -> str:
        """
        Handle Cast expressions, particularly for timestamps.
        
        Converts timestamps to nanoseconds since epoch to match
        the application's internal representation.
        """
        cast_to = cast_expr.to
        
        # Check if it's a timestamp cast
        if cast_to and 'TIMESTAMP' in str(cast_to).upper():
            inner_expr = cast_expr.this
            
            if isinstance(inner_expr, exp.Literal):
                timestamp_str = inner_expr.this.strip("'\"")
                
                try:
                    # Parse with pandas
                    dt = pd.to_datetime(timestamp_str)
                    
                    # Convert to nanoseconds since epoch
                    nanoseconds = int(dt.value)
                    
                    return str(inner_expr)
                
                except Exception:
                    return f"'{timestamp_str}'"
        
        # For non-timestamp casts, extract inner value
        if isinstance(cast_expr.this, exp.Literal):
            value = cast_expr.this.this
            return self._format_value(col_name, value)
        
        return cast_expr.sql()
    
    def _format_value(self, col_name: str, value) -> str:
        """Format value for pandas query based on column dtype."""
        # Check if column is numeric
        if col_name in self.distincts.columns:
            if not pd.api.types.is_numeric_dtype(self.distincts[col_name]):
                # String column - add quotes
                return f"'{value}'"
        
        return str(value)

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
            return self.saved[sketch_id].clone(), 0
        
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
            self.saved[sketch_id] = sketch.detach().clone()
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
    
    def __call__(self, predicates: exp.Expression, keys:dict, components:dict, cuda : bool = False, separate_negatives : bool = False, **kwargs):
        """
        returns:
            the selectivity of the predicates (float) or the sketch of the keys (Estimator)
        """
        if predicates is not None:
            col_in_preds = self._get_columns_from_expr(predicates)
            col_in_preds = self.columns.intersection(col_in_preds)
        else:
            col_in_preds = set()

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
            # otherwise, filter selection is needed - convert to pandas query
            selection = self._filter_with_expression(predicates)

            if col_in_keys:
                # return pushdown sketch
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

        # check if sketch already exists (no predicates)
        if not col_in_preds and sketch_id in self.sketches:
            if self.sparse:
                return self.sketches[sketch_id].to_dense(), 0
            else:
                return self.sketches[sketch_id].clone(), 0

        # create sketch for keys without predicates
        t0 = perf_counter_ns()
        sketch = torch.zeros((self.depth, self.width * (2 if separate_negatives else 1)), dtype=torch.float)

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
                return self.sketches[sketch_id].clone(), 0
        
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
                    preds.append(f"`{col}`.notna() & `{col}`.str.contains(r'{val}', case=False, regex=True)")
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

class UnivariateEstimator(Sketch):
    """ Base class for univariate selectivity estimators. """
    def __init__(self, **kwargs):
        pass
    
    # ========================================================================
    # RANGE EXTRACTION (Handles AND/OR/IN/BETWEEN)
    # ========================================================================
    
    def _extract_ranges(self, expr: exp.Expression) -> List[Tuple[float, float]]:
        """
        Extract all ranges from expression.
        
        Handles:
        - AND: Intersection of child ranges
        - OR: Union of child ranges
        - IN: OR of equalities
        - BETWEEN: Single range
        - Comparisons: Single range
        
        Returns:
            List of disjoint (left, right) ranges
        
        Example:
            age > 50 OR age < 20
            → [(50.0001, max), (min, 19.9999)]
        """
        if isinstance(expr, exp.And):
            # AND: Intersect ranges from children
            left_ranges = self._extract_ranges(expr.this)
            right_ranges = self._extract_ranges(expr.expression)
            
            # check if either side is empty
            if not left_ranges and not right_ranges:
                return [(self.min, self.max)]
            elif not left_ranges:
                return right_ranges # Left doesn't apply
            elif not right_ranges:
                return left_ranges # Right doesn't apply

            # Intersect all pairs
            result = []
            for l1, r1 in left_ranges:
                for l2, r2 in right_ranges:
                    intersection = self._intersect_ranges((l1, r1), (l2, r2))
                    if intersection:
                        result.append(intersection)
            
            return self._merge_ranges(result)
        
        elif isinstance(expr, exp.Or):
            # OR: Union ranges from children
            left_ranges = self._extract_ranges(expr.this)
            right_ranges = self._extract_ranges(expr.expression)
            
            # Merge all ranges
            return self._merge_ranges(left_ranges + right_ranges)
        
        elif isinstance(expr, exp.In):
            # IN: Convert to OR of equalities
            return self._handle_in(expr)
        
        elif isinstance(expr, exp.Between):
            # BETWEEN: Convert to single range
            return self._handle_between(expr)
        
        else:
            # Leaf comparison: extract single range
            range_bounds = self._comparison_to_range(expr)
            return range_bounds
    
    def _comparison_to_range(self, comp: exp.Expression) -> Optional[Tuple[float, float]]:
        """
        Convert comparison to single range.
        
        Args:
            comp: Comparison expression
        
        Returns:
            (left, right) bounds or None if not our column
        """
        # Check if this is a comparison on our column
        if not isinstance(comp.left, exp.Column):
            return []
        
        if comp.left.name != self.column:
            return []
        
        # Extract value
        val = self._extract_value(comp.right)
        
        # Convert to range based on operator
        epsilon = 1e-6
        
        if isinstance(comp, exp.EQ):
            # Equality: single point
            return [(val, val)]
        
        elif isinstance(comp, exp.NEQ):
            # Not equal: two ranges
            # For simplicity, return full range (conservative)
            # Proper handling requires OR: (min, val-ε) OR (val+ε, max)
            return [(self.min, val - epsilon), (val + epsilon, self.max)]
        
        elif isinstance(comp, exp.GT):
            # Strictly greater
            return [(val + epsilon, self.max)]
        
        elif isinstance(comp, exp.GTE):
            # Greater or equal
            return [(val, self.max)]
        
        elif isinstance(comp, exp.LT):
            # Strictly less
            return [(self.min, val - epsilon)]
        
        elif isinstance(comp, exp.LTE):
            # Less or equal
            return [(self.min, val)]
        
        else:
            # Unknown comparison type
            return []
    
    def _handle_between(self, node: exp.Between) -> List[Tuple[float, float]]:
        """
        Handle BETWEEN predicate.
        
        Args:
            node: Between expression
        
        Returns:
            Single range [low, high]
        
        Example:
            age BETWEEN 30 AND 50 → [(30, 50)]
        """
        # Check if it's for our column
        col = node.this
        if not isinstance(col, exp.Column) or col.name != self.column:
            return [(self.min, self.max)]  # Not our column
        
        # Extract bounds
        low = node.args.get('low')
        high = node.args.get('high')
        
        low_val = self._extract_value(low)
        high_val = self._extract_value(high)
        
        # BETWEEN is inclusive on both ends
        return [(low_val, high_val)]
    
    def _handle_in(self, node: exp.In) -> List[Tuple[float, float]]:
        """
        Handle IN predicate.
        
        Converts to OR of equalities.
        
        Args:
            node: In expression
        
        Returns:
            List of single-point ranges
        
        Example:
            age IN (25, 30, 35) → [(25, 25), (30, 30), (35, 35)]
        """
        # Check if it's for our column
        col = node.this
        if not isinstance(col, exp.Column) or col.name != self.column:
            return [(self.min, self.max)]  # Not our column
        
        # Extract values
        values = []
        for val_node in node.expressions:
            val = self._extract_value(val_node)
            values.append(val)
        
        # Convert to list of single-point ranges
        # Each value is a point: (val, val)
        ranges = [(v, v) for v in values]
        
        # Merge any adjacent points
        return self._merge_ranges(ranges)
    
    # ========================================================================
    # RANGE ALGEBRA
    # ========================================================================
    
    def _intersect_ranges(self, r1: Tuple[float, float], 
                         r2: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Compute intersection of two ranges.
        
        Args:
            r1: First range (left, right)
            r2: Second range (left, right)
        
        Returns:
            Intersection range or None if empty
        
        Example:
            (10, 30) ∩ (25, 50) = (25, 30)
            (10, 20) ∩ (30, 40) = None (disjoint)
        """
        left = max(r1[0], r2[0])
        right = min(r1[1], r2[1])
        
        if left <= right:
            return (left, right)
        else:
            return None  # Empty intersection
    
    def _merge_ranges(self, ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Merge overlapping ranges.
        
        Args:
            ranges: List of ranges
        
        Returns:
            List of disjoint ranges
        
        Example:
            [(10, 30), (25, 50), (70, 90)]
            → [(10, 50), (70, 90)]
        """
        if not ranges:
            return []
        
        # Sort by left endpoint
        sorted_ranges = sorted(ranges)
        
        merged = [sorted_ranges[0]]
        
        epsilon = 1e-6
        
        for left, right in sorted_ranges[1:]:
            prev_left, prev_right = merged[-1]
            
            # Check if overlapping or adjacent
            if left <= prev_right + epsilon:
                # Merge: extend previous range
                merged[-1] = (prev_left, max(prev_right, right))
            else:
                # Disjoint: add new range
                merged.append((left, right))
        
        return merged
    
    # ========================================================================
    # ESTIMATION
    # ========================================================================
    
    def _estimate_ranges(self, ranges: List[Tuple[float, float]]) -> float:
        """
        Estimate cardinality for union of ranges.
        
        Args:
            ranges: List of disjoint ranges
        
        Returns:
            Total cardinality estimate
        """
        if not ranges:
            return 0
        
        # Skip empty ranges
        valid_ranges = [(l, r) for l, r in ranges if l <= r]
        
        if not valid_ranges:
            return 0
        
        # Calculate covers for all ranges
        all_covers = []
        
        for left, right in valid_ranges:
            covers = calculate_intervals(left, right, self.sorted_intervals)
            all_covers.append(covers)
        
        # Merge all covers before querying sketch
        merged_cover = self._merge_covers(all_covers)
        
        # Query sketch once for merged covers
        estimates = self._estimate_from_covers(merged_cover)
        
        return estimates
    
    def _merge_covers(self, covers: List[dict]) -> dict:
        """
        Merge multiple cover dictionaries into one.
        
        Combines covers from multiple ranges, eliminating overlaps
        to avoid double-counting intervals.
        
        Args:
            covers: List of cover dictionaries
                   Each cover: {interval_size: [[left, right], ...], ...}
        
        Returns:
            Merged cover dictionary
        
        Example:
            cover1 = {10: [[0, 2]], 1: [[30, 40]]}
            cover2 = {10: [[7, 9]], 1: [[35, 50]]}
            
            merged = {
                10: [[0, 2], [7, 9]],  # Disjoint ranges
                1:  [[30, 50]]          # Merged [30,40] ∪ [35,50]
            }
        """
        if not covers:
            return {}
        
        if len(covers) == 1:
            return covers[0]
        
        # Collect all interval sizes from all covers
        all_sizes = set()
        for cover in covers:
            all_sizes.update(cover.keys())
        
        merged = {}
        
        # Merge ranges at each interval size level
        for size in all_sizes:
            # Collect all ranges for this interval size
            all_ranges = []
            for cover in covers:
                if size in cover:
                    all_ranges.extend(cover[size])
            
            # Merge overlapping/adjacent ranges
            merged_ranges = self._merge_interval_ranges(all_ranges)
            
            if merged_ranges:
                merged[size] = merged_ranges
        
        return merged
    
    def _merge_interval_ranges(self, ranges: List[List[int]]) -> List[List[int]]:
        """
        Merge overlapping interval index ranges.
        
        Args:
            ranges: List of [left, right] interval index ranges
        
        Returns:
            List of disjoint [left, right] ranges
        
        Example:
            Input:  [[10, 20], [15, 30], [50, 60]]
            Output: [[10, 30], [50, 60]]
            
            Input:  [[10, 20], [21, 30]]  # Adjacent
            Output: [[10, 30]]             # Merged
        """
        if not ranges:
            return []
        
        # Sort by left endpoint
        sorted_ranges = sorted(ranges, key=lambda r: r[0])
        
        merged = [sorted_ranges[0]]
        
        for left, right in sorted_ranges[1:]:
            prev_left, prev_right = merged[-1]
            
            # Check if overlapping or adjacent
            # Adjacent intervals should merge: [10,20] + [21,30] → [10,30]
            if left <= prev_right + 1:
                # Merge: extend previous range
                merged[-1] = [prev_left, max(prev_right, right)]
            else:
                # Disjoint: add new range
                merged.append([left, right])
        
        return merged
    
    def _estimate_from_covers(self, covers: dict, agg: Callable = min) -> float:
        """
        Estimate cardinality from interval covers.
        
        (Implementation unchanged from original CountMin)
        """
        # Determine total intervals
        total_intervals = 0
        intervals_per_level = {}
        
        for interval_size, ranges in covers.items():
            num = int(sum(right - left + 1 for left, right in ranges))
            intervals_per_level[interval_size] = num
            total_intervals += num
        
        # Build interval array
        intervals = np.zeros(total_intervals, dtype=np.int64)
        
        current_idx = 0
        for interval_size, ranges in covers.items():
            for left, right in ranges:
                next_idx = int(current_idx + right - left + 1)
                intervals[current_idx:next_idx] = np.arange(left, right + 1)
                current_idx = next_idx
        
        # Hash intervals
        vhash = np.vectorize(hash)
        intervals = vhash(intervals) + 1
        
        # Sketch query
        bins = self.bin_hash(intervals) % self.width
        
        # Estimate for each interval level
        estimates = []
        bins_idx = 0
        
        for interval_size, num_intervals in intervals_per_level.items():
            bins_intervals = bins[:, bins_idx:bins_idx + num_intervals]
            bins_idx += num_intervals
            
            # Gather from sketch
            counts = self.sketches[interval_size].gather(1, bins_intervals)
            
            # independent estimates
            estimates.append(counts.sum(dim=1))

        return estimates

    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _extract_value(self, node: exp.Expression) -> float:
        """Extract numeric value from expression node."""
        if isinstance(node, exp.Literal):
            return float(node.this)
        
        elif isinstance(node, exp.Cast):
            cast_to = node.to
            
            if cast_to and 'TIMESTAMP' in str(cast_to).upper():
                inner = node.this
                if isinstance(inner, exp.Literal):
                    timestamp_str = inner.this.strip("'\"")
                    dt = pd.to_datetime(timestamp_str)
                    return float(dt.value)
            
            if isinstance(node.this, exp.Literal):
                return float(node.this.this)
        
        try:
            return float(node.sql())
        except:
            raise ValueError(f"Cannot extract value from {node.sql()}")

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
    
    def __call__(self, predicates: exp.Expression, keys:dict, *args, **kwargs):
        """
        returns:
            the Count Sketch selectivity estimate of the predicates (float)
        """
        if predicates is not None:
            col_in_preds = self.column in self._get_columns_from_expr(predicates)

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

class CountMin(UnivariateEstimator):
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
    
    def __call__(self, predicates: exp.Expression, keys:dict, *args, **kwargs):
        """
        returns:
            the Count Min selectivity estimate of the predicates (float)
        """
        col_in_preds = self.column in self._get_columns_from_expr(predicates)

        if not col_in_preds:
            # if no selection is needed and not a join key attribute, return 1
            return 1, 0
        
        # find left and right bounds (inclusive) of the predicates
        ranges = self._extract_ranges(predicates)

        # no valid ranges
        if not ranges:
            return 0, 0

        # take the minimum estimate for each range
        estimates = self._estimate_ranges(ranges)
        est = sum(counts.min().item() for counts in estimates)
            
        # return the selectivity estimate (within [0, 1])
        prob = min(max(0, est / self.nrows), 1)
        return prob, 0

class StringCountMin:
    """
    CountMin sketch for string columns with LIKE predicate support.
    
    Uses three 3-gram based sketches:
    - Prefix sketch: hash(string[0:3])
    - Suffix sketch: hash(string[-3:])
    - Infix sketch: hash(all unique 3-grams)
    """
    
    CHUNK_SIZE = 3
    PAD_CHAR = '\x00'  # Null byte for padding short patterns
    
    def __init__(self, data: pd.Series, depth: int, width: int, 
                 bin_hash: object, **kwargs):
        """
        Initialize StringCountMin sketch.
        
        Args:
            data: Series of strings
            depth: Sketch depth
            width: Sketch width
            bin_hash: Hash function for binning
        """
        self.column = data.name
        self.nrows = len(data)
        self.depth = depth
        self.width = width
        self.bin_hash = bin_hash
        
        # Check if string column
        self.is_string = data.dtype == object
        
        if not self.is_string:
            raise ValueError(f"StringCountMin requires string column, got {data.dtype}")
        
        # Vectorized hash
        self.vhash = np.vectorize(hash)
        
        # Build 3-gram sketches
        self._build_string_sketches(data)
        
        # Calculate memory usage
        self.memory = self._calculate_memory()
    
    # ========================================================================
    # SKETCH CONSTRUCTION
    # ========================================================================
    
    def _build_string_sketches(self, data: pd.Series):
        """Build prefix, suffix, and infix 3-gram sketches."""
        
        # Prefix sketch: first 3 characters
        print(f"Building prefix sketch for {self.column}...")
        prefix_chunks = data.str[:self.CHUNK_SIZE].dropna()
        self.prefix_sketch = self._build_chunk_sketch(prefix_chunks)
        
        # Suffix sketch: last 3 characters
        print(f"Building suffix sketch for {self.column}...")
        suffix_chunks = data.str[-self.CHUNK_SIZE:].dropna()
        self.suffix_sketch = self._build_chunk_sketch(suffix_chunks)
        
        # Infix sketch: all unique 3-grams (presence-based)
        print(f"Building infix sketch for {self.column}...")
        self.infix_sketch = self._build_infix_sketch(data)
        
        print(f"✓ String sketches built for {self.column}")
    
    def _build_chunk_sketch(self, chunks: pd.Series) -> torch.Tensor:
        """
        Build CountMin sketch from chunk series.
        
        Args:
            chunks: Series of k-grams
        
        Returns:
            Sketch tensor [depth × width]
        """
        # Get chunk counts
        chunk_counts = chunks.value_counts()
        
        # Initialize sketch
        sketch = torch.zeros((self.depth, self.width), dtype=torch.long)
        
        # Add chunks to sketch
        for chunk, count in chunk_counts.items():
            chunk_hash = hash(chunk) + 1
            bins = self.bin_hash(np.array([chunk_hash])) % self.width
            
            # Update sketch
            for bin_idx in bins:
                sketch[:, bin_idx] += count
        
        return sketch
    
    def _build_infix_sketch(self, data: pd.Series) -> torch.Tensor:
        """
        Build presence-based infix sketch.
        
        CRITICAL: Uses set() to ensure each row is counted at most once
        per unique 3-gram. This gives correct semantics for LIKE '%pattern%'.
        
        Args:
            data: Series of strings
        
        Returns:
            Sketch tensor [depth × width]
        """
        sketch = torch.zeros((self.depth, self.width), dtype=torch.long)
        
        for string in data:
            if pd.notna(string) and len(string) >= self.CHUNK_SIZE:
                # Extract all 3-grams from this string
                # Use set() to get UNIQUE 3-grams (critical!)
                unique_kgrams = set()
                for i in range(len(string) - self.CHUNK_SIZE + 1):
                    kgram = string[i:i + self.CHUNK_SIZE]
                    unique_kgrams.add(kgram)
                
                # Add this row to each unique 3-gram's count
                for kgram in unique_kgrams:
                    kgram_hash = hash(kgram) + 1
                    bins = self.bin_hash(np.array([kgram_hash])) % self.width
                    
                    for bin_idx in bins:
                        sketch[:, bin_idx] += 1
        
        return sketch
    
    # ========================================================================
    # QUERY INTERFACE
    # ========================================================================
    
    def estimate_like(self, pattern: str) -> float:
        """
        Estimate selectivity for LIKE pattern.
        
        Args:
            pattern: LIKE pattern (e.g., 'John%', '%son', '%middle%')
        
        Returns:
            Selectivity estimate [0, 1]
        
        Strategy:
        - Extract all 3-grams from pattern
        - Query appropriate sketch(s) for each 3-gram
        - Take MINIMUM estimate (preserves Count-Min overestimation property)
        """
        # Determine pattern type
        if pattern.endswith('%') and not pattern.startswith('%'):
            # Prefix pattern: 'John%'
            prefix = pattern[:-1]
            return self._estimate_prefix(prefix)
        
        elif pattern.startswith('%') and not pattern.endswith('%'):
            # Suffix pattern: '%son'
            suffix = pattern[1:]
            return self._estimate_suffix(suffix)
        
        elif pattern.startswith('%') and pattern.endswith('%'):
            # Substring pattern: '%middle%'
            substring = pattern[1:-1]
            return self._estimate_substring(substring)
        
        else:
            # Complex pattern: 'John%son' or exact match 'John'
            return self._estimate_complex(pattern)
    
    def _estimate_prefix(self, prefix: str) -> float:
        """
        Estimate selectivity for prefix pattern.
        
        Strategy:
        - Extract all 3-grams from prefix: 'Johnson' → ['Joh', 'ohn', 'hns', 'nso', 'son']
        - Query prefix sketch for each
        - Take MINIMUM (most restrictive estimate)
        
        For short prefixes (< 3 chars), pad with null bytes.
        """
        if not prefix:
            return 1.0  # Empty prefix matches everything
        
        # Extract 3-grams
        kgrams = self._extract_kgrams(prefix, pad_right=True)
        
        if not kgrams:
            return 1.0  # No valid 3-grams
        
        # Query prefix sketch for each 3-gram
        estimates = []
        for kgram in kgrams:
            count = self._query_sketch(self.prefix_sketch, kgram)
            estimates.append(count)
        
        # Take minimum (most restrictive)
        min_count = min(estimates)
        
        return min(1.0, max(0.0, min_count / self.nrows))
    
    def _estimate_suffix(self, suffix: str) -> float:
        """
        Estimate selectivity for suffix pattern.
        
        Similar to prefix, but queries suffix sketch.
        """
        if not suffix:
            return 1.0
        
        # Extract 3-grams (pad left for suffixes)
        kgrams = self._extract_kgrams(suffix, pad_left=True)
        
        if not kgrams:
            return 1.0
        
        # Query suffix sketch
        estimates = []
        for kgram in kgrams:
            count = self._query_sketch(self.suffix_sketch, kgram)
            estimates.append(count)
        
        # Take minimum
        min_count = min(estimates)
        
        return min(1.0, max(0.0, min_count / self.nrows))
    
    def _estimate_substring(self, substring: str) -> float:
        """
        Estimate selectivity for substring pattern.
        
        Queries infix sketch for all 3-grams in substring.
        """
        if not substring:
            return 1.0
        
        # Extract 3-grams
        kgrams = self._extract_kgrams(substring, pad_right=True)
        
        if not kgrams:
            return 1.0
        
        # Query infix sketch
        estimates = []
        for kgram in kgrams:
            count = self._query_sketch(self.infix_sketch, kgram)
            estimates.append(count)
        
        # Take minimum
        min_count = min(estimates)
        
        return min(1.0, max(0.0, min_count / self.nrows))
    
    def _estimate_complex(self, pattern: str) -> float:
        """
        Estimate selectivity for complex pattern like 'John%son' or 'John'.
        
        Strategy:
        - Split on '%'
        - Estimate each part (prefix, suffix, substrings)
        - Take MINIMUM across all parts
        """
        parts = pattern.split('%')
        
        if len(parts) == 1:
            # No wildcards - exact match
            # Use prefix sketch for first 3 chars
            return self._estimate_prefix(pattern)
        
        estimates = []
        
        # First part: prefix
        if parts[0]:
            sel = self._estimate_prefix(parts[0])
            estimates.append(sel * self.nrows)  # Convert back to count
        
        # Last part: suffix
        if parts[-1]:
            sel = self._estimate_suffix(parts[-1])
            estimates.append(sel * self.nrows)
        
        # Middle parts: substrings
        for part in parts[1:-1]:
            if part:
                sel = self._estimate_substring(part)
                estimates.append(sel * self.nrows)
        
        if not estimates:
            return 1.0
        
        # Take minimum across all parts
        min_count = min(estimates)
        
        return min(1.0, max(0.0, min_count / self.nrows))
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _extract_kgrams(self, s: str, pad_left: bool = False, 
                        pad_right: bool = False) -> List[str]:
        """
        Extract all 3-grams from string.
        
        Args:
            s: String to extract from
            pad_left: Pad on left for suffix patterns
            pad_right: Pad on right for prefix patterns
        
        Returns:
            List of 3-grams
        
        Examples:
            'Johnson' → ['Joh', 'ohn', 'hns', 'nso', 'son']
            'Jo' (pad_right) → ['Jo\x00']
            'on' (pad_left) → ['\x00on']
        """
        if len(s) < self.CHUNK_SIZE:
            # Short string - need padding
            if pad_left:
                s = self.PAD_CHAR * (self.CHUNK_SIZE - len(s)) + s
            elif pad_right:
                s = s + self.PAD_CHAR * (self.CHUNK_SIZE - len(s))
            else:
                # No padding - can't extract 3-grams
                return []
            
            return [s]
        
        # Extract all 3-grams
        kgrams = []
        for i in range(len(s) - self.CHUNK_SIZE + 1):
            kgram = s[i:i + self.CHUNK_SIZE]
            kgrams.append(kgram)
        
        return kgrams
    
    def _query_sketch(self, sketch: torch.Tensor, kgram: str) -> float:
        """
        Query sketch for 3-gram count.
        
        Returns minimum count across all hash functions (Count-Min property).
        """
        kgram_hash = hash(kgram) + 1
        bins = self.bin_hash(np.array([kgram_hash])) % self.width
        
        # Get counts from sketch
        counts = []
        for bin_idx in bins:
            counts.append(sketch[:, bin_idx].min().item())
        
        # Return minimum across all bins (most conservative estimate)
        return min(counts) if counts else 0
    
    def _calculate_memory(self) -> int:
        """Calculate total memory usage in bytes."""
        nbytes = 0
        
        for sketch in [self.prefix_sketch, self.suffix_sketch, self.infix_sketch]:
            if sketch.is_sparse:
                indices = sketch.indices()
                nbytes += indices.nelement() * indices.element_size()
                values = sketch.values()
                nbytes += values.nelement() * values.element_size()
            else:
                nbytes += sketch.numel() * sketch.element_size()
        
        return nbytes
    
    # ========================================================================
    # MAIN ENTRY POINT (for integration with CountMin)
    # ========================================================================
    
    def __call__(self, predicate: Optional[exp.Expression], 
                 keys: dict) -> Tuple[float, int]:
        """
        Estimate selectivity for LIKE predicate.
        
        Args:
            predicate: sqlglot LIKE expression
            keys: Unused (for compatibility)
        
        Returns:
            (selectivity, time_ns)
        """
        if predicate is None:
            return 1.0, 0
        
        # Handle LIKE predicates
        if isinstance(predicate, exp.Like):
            col = predicate.this
            
            # Check if it's for our column
            if not isinstance(col, exp.Column) or col.name != self.column:
                return 1.0, 0  # Not our column
            
            # Extract pattern
            pattern_node = predicate.expression
            if isinstance(pattern_node, exp.Literal):
                pattern = pattern_node.this.strip("'\"")
                
                # Estimate selectivity
                selectivity = self.estimate_like(pattern)
                
                return selectivity, 0
        
        # Not a LIKE predicate
        return 1.0, 0