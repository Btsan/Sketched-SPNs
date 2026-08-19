from collections import deque
from time import perf_counter_ns
import re
from typing import List, Tuple, Dict, Optional, Callable, Set

import numpy as np
import pandas as pd
import torch

from sqlglot import exp

import predicate_to_string as _pts
from predicate_to_string import predicate_to_canonical_string

# from Estimators import CountEstimator, DegreeEstimator

def _to_hash_values(data) -> np.ndarray:
    """Convert column values to int64 for KWiseHash sign/bin functions.

    Dispatches by dtype to avoid the ~265 ms/call overhead of np.vectorize(hash)
    for large integer columns (all JOB join keys are int64, yielding 80× speedup).
    Accepts both pd.Series and np.ndarray.
    """
    arr = data.to_numpy() if isinstance(data, pd.Series) else np.asarray(data)
    kind = arr.dtype.kind
    if kind in ('i', 'u'):                          # signed/unsigned integer
        return arr.astype(np.int64, copy=False) + 1
    elif kind == 'f':                               # float (may contain NaN)
        return np.where(np.isfinite(arr), arr.astype(np.int64), 0) + 1
    else:                                           # object/string — CPython hash cache is fast
        return np.vectorize(hash)(arr) + 1

class _ExpiryCache:
    """Dict-backed sketch cache with query-count TTL eviction."""

    def __init__(self, ttl: int = 100000):
        self.ttl = ttl
        self._data:   dict  = {}
        self._nbytes: dict  = {}   # key → stored byte count
        self._access: dict  = {}   # key → last query_id (authoritative)
        self._queue:  deque = deque()  # (query_id, key) — append-only, may have stale entries
        self._bytes:  int   = 0
        self._last_sweep: int = -1

    def __contains__(self, key) -> bool:
        return key in self._data

    def get(self, key):
        return self._data.get(key)

    def put(self, key, value, nbytes: int, query_id: int) -> None:
        self._data[key]   = value
        self._nbytes[key] = nbytes
        self._access[key] = query_id
        self._queue.append((query_id, key))
        self._bytes += nbytes

    def touch(self, key, query_id: int) -> None:
        """Extend TTL of an existing entry."""
        self._access[key] = query_id
        self._queue.append((query_id, key))

    def sweep(self, current_qid: int) -> None:
        """Evict entries whose last access is older than `ttl` queries. O(evicted) amortised."""
        if current_qid <= self._last_sweep:
            return
        self._last_sweep = current_qid
        threshold = current_qid - self.ttl
        while self._queue and self._queue[0][0] <= threshold:
            qid, key = self._queue.popleft()
            if self._access.get(key) == qid and key in self._data:
                del self._data[key]
                self._bytes -= self._nbytes.pop(key, 0)
                del self._access[key]

    def byte_usage(self) -> int:
        return self._bytes


class Sketch(object):
    """Base class for sketches."""

    def memory_usage(self):
        return 0

    def __call__(self, predicates: exp.Expression, keys: dict, **kwargs):
        """
        returns:
            the selectivity of the predicates (float) or the sketch of the keys (Tensor)
        """
        raise NotImplementedError("Subclasses should implement this method.")

    # ========================================================================
    # COLUMN-AWARE PREDICATE FILTERING
    # ========================================================================
    
    def _filter_predicates_by_columns(self, expr: exp.Expression, 
                                     relevant_columns: Set[str]) -> Optional[exp.Expression]:
        """
        Filter expression to only include predicates on relevant columns.
        
        Handles:
        - Leaf predicates on irrelevant columns → remove (treat as True)
        - AND with irrelevant parts → simplify
        - OR with irrelevant parts → handle carefully
        
        Args:
            expr: sqlglot Expression
            relevant_columns: Set of column names to keep
        
        Returns:
            Filtered expression, or None if entire predicate is irrelevant
        
        Examples:
            expr = "x=1 AND y=2", relevant = {y}
            → Returns: "y=2"
            
            expr = "x=1 OR y=2", relevant = {y}
            → Returns: None (conservative - can't filter safely)
            
            expr = "x=1", relevant = {y}
            → Returns: None
        """
        if expr is None:
            return None
        
        # Parentheses - unwrap
        if isinstance(expr, exp.Paren):
            filtered = self._filter_predicates_by_columns(expr.this, relevant_columns)
            return exp.Paren(this=filtered) if filtered else None
        
        # NOT - check inner
        if isinstance(expr, exp.Not):
            filtered = self._filter_predicates_by_columns(expr.this, relevant_columns)
            return exp.Not(this=filtered) if filtered else None
        
        # AND - keep only relevant sides
        if isinstance(expr, exp.And):
            left = self._filter_predicates_by_columns(expr.this, relevant_columns)
            right = self._filter_predicates_by_columns(expr.expression, relevant_columns)
            
            # Both sides relevant
            if left and right:
                return exp.And(this=left, expression=right)
            # Only left relevant
            elif left:
                return left
            # Only right relevant
            elif right:
                return right
            # Neither relevant
            else:
                return None
        
        # OR - irrelevant terms treated as FALSE
        if isinstance(expr, exp.Or):
            left = self._filter_predicates_by_columns(expr.this, relevant_columns)
            right = self._filter_predicates_by_columns(expr.expression, relevant_columns)
            
            # Irrelevant conditions are FALSE
            # FALSE OR relevant → relevant
            # relevant OR FALSE → relevant
            # FALSE OR FALSE → FALSE (None)
            
            # Both sides relevant
            if left and right:
                return exp.Or(this=left, expression=right)
            # Only left relevant (right is FALSE)
            elif left:
                return left
            # Only right relevant (left is FALSE)
            elif right:
                return right
            # Both irrelevant (both FALSE)
            else:
                return None
        
        # Leaf predicate - check if columns are relevant
        expr_columns = self._get_columns_from_expr(expr)
        
        if expr_columns.issubset(relevant_columns):
            # All columns relevant - keep predicate
            return expr
        else:
            # Contains irrelevant columns - remove
            return None

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
        
        MODIFIED: Only includes predicates on relevant columns (self.columns).
        
        Args:
            expr: sqlglot expression
        
        Returns:
            Filtered DataFrame
        """
        if expr is None:
            return self.distincts
        
        # Filter to only relevant columns
        if hasattr(self, 'columns'):
            filtered_expr = self._filter_predicates_by_columns(expr, self.columns)
        else:
            filtered_expr = expr
        
        # If no relevant predicates remain, return all data
        if filtered_expr is None:
            return self.distincts
        
        # Convert sqlglot expression to pandas query string
        query_str = self._expr_to_pandas_query(filtered_expr)
        
        if query_str:
            try:
                selection = self.distincts.query(query_str)
                # print(f"Filtering with query: {query_str} --> {len(selection)}/{len(self.distincts)} rows")
                return selection
            except Exception as e:
                raise ValueError(f"Failed to parse expression to pandas query: {query_str}\nError: {e}")
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
        # Handle parenthesized expressions FIRST
        if isinstance(expr, exp.Paren):
            return self._expr_to_pandas_query(expr.this)
        
        elif isinstance(expr, exp.And):
            # Recursively process AND children
            left = self._expr_to_pandas_query(expr.this)
            right = self._expr_to_pandas_query(expr.expression)
            return f"({left}) & ({right})"
        
        elif isinstance(expr, exp.Or):
            # Recursively process OR children
            left = self._expr_to_pandas_query(expr.this)
            right = self._expr_to_pandas_query(expr.expression)
            return f"({left}) | ({right})"
        
        elif isinstance(expr, exp.Not):
            # Handle NOT expressions
            # Check if this is "IS NOT NULL" (NOT wrapping IS NULL)
            inner = expr.this
            if isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null):
                # IS NOT NULL
                if isinstance(inner.this, exp.Column):
                    col_name = inner.this.name
                    return f"`{col_name}`.notna()"
            
            # General NOT handling (for other cases)
            inner_query = self._expr_to_pandas_query(expr.this)
            if inner_query:
                return f"~({inner_query})"
            return ""
        
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
        
        elif isinstance(expr, exp.Is):
            # Handle IS expressions (IS NULL, IS TRUE, IS FALSE, etc.)
            col = expr.this
            value = expr.expression
            
            if isinstance(col, exp.Column):
                col_name = col.name
                
                if isinstance(value, exp.Null):
                    # IS NULL
                    return f"`{col_name}`.isna()"
                
                elif isinstance(value, exp.Boolean):
                    # IS TRUE / IS FALSE
                    bool_val = str(value.this).lower()
                    return f"`{col_name}` == {bool_val}"
                
                elif isinstance(value, exp.Literal):
                    # IS <literal value>
                    val_str = self._format_value(col_name, value.this)
                    return f"`{col_name}` == {val_str}"
        
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
            regex_pattern = self._like_to_regex(pattern.this)
            return f"`{col.name}`.str.contains('{regex_pattern}', na=False, regex=True)"
        
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
        else:
            raise NotImplementedError(f"Expression type {type(expr)} not supported in pandas query conversion.")
        return ""
    
    def _like_to_regex(self, pattern: str) -> str:
        """Convert SQL LIKE pattern to regex pattern."""
        # Escape backslashes
        escaped = pattern.replace('\\', '\\\\')
        
        # Escape regex special characters
        for char in '.^$*+?{}[]|()':
            escaped = escaped.replace(char, '\\' + char)
        
        # Convert SQL wildcards: % → .*, _ → .
        regex = escaped.replace('%', '.*').replace('_', '.')
        
        # Anchor for exact matching
        return '^' + regex + '$'
    
    def _comparison_to_query(self, expr: exp.Expression, op: str) -> str:
        """Convert comparison expression to pandas query string."""
        left = expr.left
        right = expr.right
        
        if isinstance(left, exp.Column):
            col_name = left.name
            
            # Handle different right-hand side types
            if isinstance(right, exp.Literal):
                # Simple literal value
                value = right.this
                value_str = self._format_value(col_name, value)
            
            elif isinstance(right, exp.Cast):
                # Cast expression (e.g., timestamp)
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
        if (cast_to and 'TIMESTAMP' in str(cast_to).upper()) or \
            (pd.api.types.is_datetime64_any_dtype(self.distincts[col_name])):
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
        if hasattr(self, 'distincts') and col_name in self.distincts.columns:
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
            values = _to_hash_values(self.distincts[col])
            mask = self.distincts[col].notnull().to_numpy()[None, :] # [1, N]
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
                    values = _to_hash_values(selection[col])
                    for join_idx in join_indices:
                        signs *= self.sign_hashes[join_idx](values)
                    mask = selection[col].notnull().to_numpy()[None, :]
                    signs *= mask
                assert signs.shape == (self.depth, max(1, len(selection))), f"{signs.shape} == {(self.depth, len(selection))}"
                signs *= selection['_count'].to_numpy()[None, :]
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
        signs *= self.distincts['_count'].to_numpy()[None, :]
        sketch = signs.sum(dim=-1, keepdim=True).float()
        
        t1 = perf_counter_ns()
        sketch_time = (t1 - t0)

        # save sketch for reuse, if there were no predicates
        if not col_in_preds:
            self.saved[sketch_id] = sketch.detach().clone()
        return sketch, sketch_time

class FastAGMS(Sketch):
    def __init__(self, data:pd.DataFrame, depth:int, width:int, sign_hashes:list, bin_hashes:list, exact_preds=False, sparse=False, sample_sketch=None, method='count-sketch', **kwargs):
        self.depth = depth
        self.width = width
        self.nrows = len(data)
        self.sign_hashes = sign_hashes
        self.bin_hashes = bin_hashes
        self.sparse = sparse
        self.method = method.lower()

        self.columns = [data.name,] if isinstance(data, pd.Series) else list(data.columns)

        # creates a dataframe with only distinct rows and their counts
        if isinstance(data, pd.Series):
            dtypes = {data.name: data.dtype}
        else:
            dtypes = data.dtypes.to_dict()
        self.distincts = data.value_counts(dropna=False).sort_values(ascending=False).reset_index(name='_count')

        # truncate data if sample size is given
        self.scale_factor=1
        if sample_sketch is not None:
            if 0 < sample_sketch < 1:
                # Treat as percentage
                sample_size = int(len(self.distincts) * sample_sketch)
                sample_size = max(sample_size, 1000)
            elif self.nrows > sample_sketch >= 1:
                # Treat as absolute number
                sample_size = int(sample_sketch)
            else:
                # Ignore
                print(f"IGNORING SAMPLING IN {type(self)}")
                sample_size = len(self.distincts)
            sample_size = min(len(self.distincts), sample_size)
            self.distincts = self.distincts.iloc[:sample_size]
            self.scale_factor = self.nrows / (self.distincts['_count'].sum())
            print(f" Scale Factor {self.scale_factor:.2f} ", end='')
            assert self.scale_factor >= 1, self.scale_factor


        # cast back to original dtypes, just in case
        for col, dtype in dtypes.items():
            if col in self.distincts.columns:
                self.distincts[col] = self.distincts[col].astype(dtype)
        self.columns = set(self.columns)

        self._cache = _ExpiryCache(ttl=kwargs.get('sketch_ttl', 100000))
        self.memory = self.distincts.memory_usage().sum()

        # memory usage of pushdown (exact) sketches
        self.pushdown = dict()

        # Count-Min for predicate selectivity
        self.countmins = {}
        if not exact_preds:
            for col in self.columns:
                values = _to_hash_values(self.distincts[col])
                mask = self.distincts[col].notnull().to_numpy()[None, :] # 1, N
                # bins = torch.concatenate([bin_hash(values) for bin_hash in bin_hashes], dim=0)
                bins = bin_hashes[0](values) % self.width
                counts = torch.tensor(self.distincts['_count'].to_numpy())[None, :].expand_as(bins)
                counts *= mask # don't count nulls
                # assert bins.shape == counts.shape == (self.depth * len(bin_hashes), len(distincts)), \
                #     f"{bins.shape} == {counts.shape} == {self.depth * len(bin_hashes), len(distincts)}"
                assert bins.shape == counts.shape == (self.depth, len(self.distincts)), \
                    f"{bins.shape} == {counts.shape} == {self.depth, len(self.distincts)}"
                # print(f"\n{col} {distincts['_count']}  counts {counts}")
                # print(f"\nvalues {values} bins {bins}")
                # self.countmins[col] = torch.zeros((self.depth * len(bin_hashes), self.width), dtype=torch.long).scatter_add_(1, bins, counts)
                self.countmins[col] = torch.zeros((self.depth, self.width), dtype=torch.long).scatter_add_(1, bins, counts)

    def compute_sketch(self, distincts:pd.DataFrame, keys:dict, components:dict):
        if self.method == 'count-sketch':
            sketch = torch.zeros((self.depth, self.width), dtype=torch.float)
            signs = 1
            bins = 0
            global_mask = np.ones((1, len(distincts)), dtype=bool)
            for key, join_indices in keys.items():
                values = _to_hash_values(distincts[key])
                global_mask &= ~pd.isna(distincts[key].to_numpy())[None, :]
                try:
                    bins += self.bin_hashes[components[key]](values)
                except Exception as e:
                    print(f"Error computing bins for {key}: {e}")
                    if components[key] >= len(self.bin_hashes):
                        raise ValueError(f"Component index {components[key]} out of range for bin_hashes of length {len(self.bin_hashes)}")
                for join_idx in join_indices:
                    try:
                        signs *= self.sign_hashes[join_idx](values)
                    except Exception as e:
                        print(f"Error computing signs for {key}, join_idx {join_idx}: {e}")
                        if join_idx >= len(self.sign_hashes):
                            raise ValueError(f"Join index {join_idx} out of range for sign_hashes of length {len(self.sign_hashes)}")
            assert bins.shape == signs.shape == (self.depth, max(1, len(distincts))), f"{bins.shape} == {signs.shape} == {(self.depth, len(distincts))}"
            bins %= self.width
            signs *= global_mask
            signs *= distincts['_count'].to_numpy()[None, :]
            sketch.view(self.depth, -1).scatter_add_(1, bins.long(), signs.float())
        elif self.method in ('bound-sketch', 'factorjoin'):
            sketch = torch.zeros((self.depth, self.width, 2), dtype=torch.float)
            bins = 0
            mask = None
            for key, join_indices in keys.items():
                values = _to_hash_values(distincts[key])
                mask = ~pd.isna(distincts[key].to_numpy())[None, :]
                try:
                    bins += self.bin_hashes[components[key]](values)
                except Exception as e:
                    print(f"Error computing bins for {key}: {e}")
                    if components[key] >= len(self.bin_hashes):
                        raise ValueError(f"Component index {components[key]} out of range for bin_hashes of length {len(self.bin_hashes)}")
            assert bins.shape == (self.depth, max(1, len(distincts))), f"{bins.shape} == {(self.depth, len(distincts))}"
            bins %= self.width
            counts = torch.tensor(distincts['_count'])[None, :].expand_as(bins) * mask
            sketch[:,:,0].view(self.depth, -1).scatter_reduce_(1, bins.long(), counts.float(), 'amax')
            sketch[:,:,1].view(self.depth, -1).scatter_reduce_(1, bins.long(), counts.float(), 'sum')
        else:
            raise NotImplementedError(f"Method {self.method} not impelmented")
        return sketch


    def memory_usage(self):
        nbytes = self._cache.byte_usage()
        for t in self.countmins.values():
            nbytes += t.numel() * t.element_size()
        nbytes += self.memory  # self.distincts DataFrame
        return nbytes

    def __call__(self, predicates: exp.Expression, keys:dict, components:dict, cuda : bool = False, **kwargs):
        """
        returns:
            the selectivity of the predicates (float) or the sketch of the keys (Estimator)
        """
        current_qid = _pts._query_id
        self._cache.sweep(current_qid)

        if predicates is not None:
            col_in_preds = self._get_columns_from_expr(predicates)
            col_in_preds = self.columns.intersection(col_in_preds)
        else:
            col_in_preds = set()

        col_in_keys = self.columns.intersection(keys.keys())

        sketch_id = frozenset(keys.items()).union(components.items())
        if not col_in_keys and not col_in_preds:
            # if no selection is needed and not a join key attribute, return 1
            return 1, 0
        elif col_in_preds:
            sketch_id = sketch_id.union({predicate_to_canonical_string(predicates)})
            if sketch_id in self._cache:
                _s = self._cache.get(sketch_id)
                self._cache.touch(sketch_id, current_qid)
                return (_s.to_dense() if _s.is_sparse else _s.clone()), 0
            t0 = perf_counter_ns()
            # otherwise, filter selection is needed - convert to pandas query
            selection = self._filter_with_expression(predicates)

            if col_in_keys:
                # return pushdown sketch
                sketch = torch.zeros((self.depth, self.width), dtype=torch.float)
                
                if len(selection) > 0:
                    selection = selection.groupby(list(keys.keys())).sum('_count').reset_index()
                    sketch = self.compute_sketch(selection, keys, components)
                    sketch *= self.scale_factor

                sketch_time = (perf_counter_ns() - t0)


                # print(f"caching pushdown sketch ({self.method}[{sketch.shape}]) for id {sketch_id}")
                # record memory usage of pushdown sketches
                # assumes sketch of selection is only ever computed once
                # self.pushdown[sketch_id] = sketch.numel() * sketch.element_size()
                nonzero_count = (torch.count_nonzero(sketch)).item()
                sparse_size = nonzero_count * (sketch.element_size() + 8) # assumes 8 bytes per index
                dense_size = sketch.numel() * sketch.element_size()
                # if the sketching time is less than 0.25s, don't save and return directly
                if sketch_time < 2.5e8:
                    # save memory usage to simulate caching
                    if sketch_id not in self.pushdown:
                        self.pushdown[sketch_id] = min(sparse_size, dense_size)
                        return sketch, sketch_time
                    else:
                        return sketch, 0
                elif self.sparse and 2 * sparse_size < dense_size:
                    _s = sketch.to_sparse()
                    _idx = _s.indices()
                    nbytes = _idx.nelement() * _idx.element_size() + _s.values().nelement() * _s.values().element_size()
                    self._cache.put(sketch_id, _s, nbytes, current_qid)
                else:
                    _s = sketch.detach().clone()
                    self._cache.put(sketch_id, _s, _s.numel() * _s.element_size(), current_qid)
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
                            values = _to_hash_values(selection[col])
                            bins = self.bin_hashes[0](values) # depth, N
                            freq += self.countmins[col].gather(1, bins).sum(dim=1).min().item()
                    prob = (self.scale_factor * freq.sum(dim=-1).min().item()) / self.nrows
                return prob, 0

        # print(f"returning sketch ({self.method}) for id {sketch_id}")
        # check if sketch already exists (no predicates)
        if sketch_id in self._cache:
            _s = self._cache.get(sketch_id)
            self._cache.touch(sketch_id, current_qid)
            return (_s.to_dense() if _s.is_sparse else _s.clone()), 0

        # create sketch for keys without predicates
        t0 = perf_counter_ns()
        sketch = self.compute_sketch(self.distincts, keys, components)
        sketch *= self.scale_factor

        t1 = perf_counter_ns()
        sketch_time = (t1 - t0)

        if self.sparse:
            nonzero_count = (torch.count_nonzero(sketch)).item()
            sparse_size = nonzero_count * (sketch.element_size() + 8)
            dense_size = sketch.numel() * sketch.element_size()
            if 2 * sparse_size < dense_size:
                _s = sketch.to_sparse()
                _idx = _s.indices()
                nbytes = _idx.nelement() * _idx.element_size() + _s.values().nelement() * _s.values().element_size()
            else:
                _s = sketch.detach().clone()
                nbytes = _s.numel() * _s.element_size()
        else:
            _s = sketch.detach().clone()
            nbytes = _s.numel() * _s.element_size()
        self._cache.put(sketch_id, _s, nbytes, current_qid)
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

        self.columns = set(self.columns)

        self._cache = _ExpiryCache(ttl=kwargs.get('sketch_ttl', 100000))
        self.memory = self.distincts.memory_usage().sum()

        # memory usage of pushdown (exact) sketches
        self.pushdown = dict()

        # Count-Min for predicate selectivity
        self.countmins = {}
        if not exact_preds:
            for col in self.columns:
                values = _to_hash_values(self.distincts[col])
                mask = self.distincts[col].notnull().to_numpy()[None, :] # 1, N
                bins = bin_hashes[0](values) % self.width
                counts = torch.tensor(self.distincts['_count'].to_numpy())[None, :].expand_as(bins)
                counts *= mask # don't count nulls
                assert bins.shape == counts.shape == (self.depth, len(self.distincts)), \
                    f"{bins.shape} == {counts.shape} == {self.depth, len(self.distincts)}"
                self.countmins[col] = torch.zeros((self.depth, self.width), dtype=torch.long).scatter_add_(1, bins, counts)

    def memory_usage(self):
        nbytes = self._cache.byte_usage()
        for t in self.countmins.values():
            nbytes += t.numel() * t.element_size()
        nbytes += self.memory  # self.distincts DataFrame
        return nbytes

    def __call__(self, predicates:dict, keys:dict, components:dict, count: bool = True, cuda: bool = False, **kwargs):
        """
        returns:
            the selectivity of the predicates (float) or the sketch of the keys (Estimator)
        """
        current_qid = _pts._query_id
        self._cache.sweep(current_qid)

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
                        values = _to_hash_values(selection[key])
                        bins += self.bin_hashes[components[key]](values)
                    bins %= self.width
                    counts = torch.tensor(selection['_count'].to_numpy())
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
                            values = _to_hash_values(selection[col])
                            bins = self.bin_hashes[0](values) # depth, N
                            freq += self.countmins[col].gather(1, bins).sum(dim=1).min().item()
                    prob *= freq.min().item() / self.nrows
                return prob, 0
        else:
            # otherwise no filters are applied and proceed to sketching
            selection = self.distincts

        # check if sketch already exists
        sketch_id = frozenset(keys.keys()).union(components.items()).union({('count', count)})
        if not col_in_preds and sketch_id in self._cache:
            _s = self._cache.get(sketch_id)
            self._cache.touch(sketch_id, current_qid)
            return _s.clone(), 0
        
        # measure sketcching time
        t0 = perf_counter_ns()

        # group by join keys
        selection = selection.groupby(list(keys.keys())).sum('_count').reset_index()

        sketch = torch.zeros((self.depth, self.width), dtype=torch.float)
        if len(selection) > 0:
            bins = 0
            for key, _ in keys.items():
                values = _to_hash_values(selection[key])
                bins += self.bin_hashes[components[key]](values)
            bins %= self.width
            counts = torch.tensor(selection['_count'].to_numpy())
            counts = counts[None, :].expand_as(bins)
            assert bins.shape == counts.shape == (self.depth, max(1, len(selection))), \
                f"{bins.shape} == {counts.shape} == {(self.depth, len(selection))}"

            sketch.view(self.depth, -1).scatter_reduce_(1, bins.long(), counts.float(), reduce_mode)

        t1 = perf_counter_ns()
        sketch_time = (t1 - t0)
        
        # save sketch for reuse, if there were no predicates
        if not col_in_preds:
            if self.sparse:
                _s = sketch.to_sparse()
                _idx = _s.indices()
                nbytes = _idx.nelement() * _idx.element_size() + _s.values().nelement() * _s.values().element_size()
            else:
                _s = sketch.detach().clone()
                nbytes = _s.numel() * _s.element_size()
            self._cache.put(sketch_id, _s, nbytes, current_qid)
        else:
            # record memory usage of pushdown sketches
            # assumes pushdown sketch is only ever computed once in a workload
            pushdown_id = sketch_id.union(preds)
            self.pushdown[pushdown_id] = sketch.numel() * sketch.element_size()
        return sketch, sketch_time
    
# sketches for selectivity estimation

class ExactSelectivity(Sketch):
    def __init__(self, data:pd.DataFrame, sample_selectivity=None, **kwargs):
        self.nrows = len(data)
        self.columns = [data.name,] if isinstance(data, pd.Series) else list(data.columns)

        if isinstance(data, pd.Series):
            dtypes = {data.name: data.dtype}
        else:
            dtypes = data.dtypes.to_dict()
        self.distincts = data.value_counts(dropna=False).sort_values(ascending=False).reset_index(name='_count')
        # cast back to original dtypes, just in case
        for col, dtype in dtypes.items():
            if col in self.distincts.columns:
                self.distincts[col] = self.distincts[col].astype(dtype)

        self.columns = set(self.columns)

        # truncate data if sample size is given
        self.scale_factor=1
        self.residual_count = 0
        if sample_selectivity is not None:
            if 0 < sample_selectivity < 1:
                # Treat as percentage
                sample_size = int(len(self.distincts) * sample_selectivity)
                sample_size = max(sample_size, 100)
            elif self.nrows > sample_selectivity >= 1:
                # Treat as absolute number
                sample_size = int(sample_selectivity)
            else:
                # Ignore
                print(f"IGNORING SAMPLING IN {type(self)}")
                sample_size = len(self.distincts)
            sample_size = min(len(self.distincts), sample_size)
            self.residual_count = self.distincts.iloc[sample_size:]['_count'].mean() if sample_size < len(self.distincts) else 0
            self.distincts = self.distincts.iloc[:sample_size]
            self.scale_factor = self.nrows / (self.distincts['_count'].sum())
            print(f" (Scale Factor {self.scale_factor:.2f}, Residual {self.residual_count:.2f}) ", end='')
            assert self.scale_factor >= 1, self.scale_factor

        # record memory usage
        self.memory = self.distincts.memory_usage().sum()
        
        # cache the selectivity of the predicates
        self.saved = dict()

    def __call__(self, predicates: exp.Expression, keys:dict, components:dict, cuda : bool = False, **kwargs):
        """
        returns:
            the selectivity of the predicates (float) or the sketch of the keys (Estimator)
        """
        t0 = perf_counter_ns()
        if predicates is not None:
            col_in_preds = self.columns.intersection(self._get_columns_from_expr(predicates))
        else:
            col_in_preds = set()

        if not col_in_preds:
            return 1, t0 - perf_counter_ns()

        sel_id = predicate_to_canonical_string(predicates)
        prob = self.saved.get(sel_id)
        if prob is not None:
            return prob, t0 - perf_counter_ns()

        # filter selection is needed - convert to pandas query
        selection = self._filter_with_expression(predicates)
        if not len(selection):
            return self.residual_count / self.nrows, t0 - perf_counter_ns()

        prob = (self.scale_factor * selection['_count'].sum() + self.residual_count) / self.nrows
        self.saved[sel_id] = prob
        return prob, t0 - perf_counter_ns()

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
        self.is_datetime = False
        pass
    
    # ========================================================================
    # RANGE EXTRACTION (Extended for NOT, IS NULL, IS TRUE/FALSE)
    # ========================================================================
    
    def _extract_ranges(self, expr: exp.Expression) -> List[Tuple[float, float]]:
        """
        Extract all ranges from expression.
        
        Handles:
        - AND: Intersection of child ranges
        - OR: Union of child ranges
        - NOT: Complement of child ranges
        - IN: OR of equalities
        - BETWEEN: Single range
        - Comparisons: Single range
        - IS TRUE/FALSE: Specific values
        - IS NOT NULL: Full range
        - Parentheses: Unwrap
        
        Returns:
            List of disjoint (left, right) ranges over non-null values
        
        Example:
            NOT (age > 50)
            → [(min, 50)]  (complement of (50, max])
        """
        # Handle Parentheses - unwrap
        if isinstance(expr, exp.Paren):
            return self._extract_ranges(expr.this)
        
        # Handle NOT - complement inner ranges
        if isinstance(expr, exp.Not):
            inner_ranges = self._extract_ranges(expr.this)
            return self._complement_ranges(inner_ranges)
        
        # Handle IS predicates
        if isinstance(expr, exp.Is):
            return self._handle_is_predicate(expr)
        
        # Handle AND: Intersect ranges from children
        if isinstance(expr, exp.And):
            left_ranges = self._extract_ranges(expr.this)
            right_ranges = self._extract_ranges(expr.expression)
            
            # Check if either side is empty
            if not left_ranges and not right_ranges:
                return [(self.min, self.max)]
            elif not left_ranges:
                return right_ranges  # Left doesn't apply
            elif not right_ranges:
                return left_ranges  # Right doesn't apply
            
            # Intersect all pairs
            result = []
            for l1, r1 in left_ranges:
                for l2, r2 in right_ranges:
                    intersection = self._intersect_ranges((l1, r1), (l2, r2))
                    if intersection:
                        result.append(intersection)
            
            return self._merge_ranges(result)
        
        # Handle OR: Union ranges from children
        elif isinstance(expr, exp.Or):
            left_ranges = self._extract_ranges(expr.this)
            right_ranges = self._extract_ranges(expr.expression)
            
            # Merge all ranges
            return self._merge_ranges(left_ranges + right_ranges)
        
        # Handle IN: Convert to OR of equalities
        elif isinstance(expr, exp.In):
            return self._handle_in(expr)
        
        # Handle BETWEEN: Convert to single range
        elif isinstance(expr, exp.Between):
            return self._handle_between(expr)
        
        else:
            # Leaf comparison: extract single range
            return self._comparison_to_range(expr)
    
    def _handle_is_predicate(self, expr: exp.Is) -> List[Tuple[float, float]]:
        """
        Handle IS predicates.
        
        Args:
            expr: IS expression
        
        Returns:
            Ranges for the predicate
        
        Cases:
            IS TRUE -> [(1, 1)]
            IS FALSE -> [(0, 0)]
            IS NOT NULL -> [(min, max)]
            IS NULL -> [] (handled separately in __call__)
        """
        # Check if it's for our column
        col = expr.this
        if not isinstance(col, exp.Column) or col.name != self.column:
            return [(self.min, self.max)]  # Not our column
        
        if isinstance(expr.expression, exp.Boolean):
            if expr.expression.this:  # True
                return [(1.0, 1.0)]
            else:  # False
                return [(0.0, 0.0)]
        
        # IS NOT NULL
        elif isinstance(expr.expression, exp.Not):
            inner = expr.expression.this
            if isinstance(inner, exp.Null):
                # IS NOT NULL -> all non-null values
                return [(self.min, self.max)]
        
        # IS NULL
        elif isinstance(expr.expression, exp.Null):
            # IS NULL -> return empty range (handled separately)
            return []
        
        # Unknown IS predicate
        return [(self.min, self.max)]
    
    def _complement_ranges(self, ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Complement a set of ranges (within valid domain).
        
        Args:
            ranges: List of disjoint ranges
        
        Returns:
            Complemented ranges covering gaps
        
        Example:
            Input:  [(20, 30), (50, 60)]
            Output: [(min, 19.999), (30.001, 49.999), (60.001, max)]
            
            Input:  [] (empty)
            Output: [(min, max)] (full range)
            
            Input:  [(min, max)]
            Output: [] (empty complement)
        """
        if not ranges:
            # Complement of empty set is full range
            return [(self.min, self.max)]
        
        epsilon = 1e-6
        result = []
        
        # Sort ranges
        sorted_ranges = sorted(ranges)
        
        # Check if full range - complement is empty
        if len(sorted_ranges) == 1:
            left, right = sorted_ranges[0]
            if abs(left - self.min) < epsilon and abs(right - self.max) < epsilon:
                # Full range -> empty complement
                return []
        
        # Add gap before first range
        first_left = sorted_ranges[0][0]
        if first_left > self.min + epsilon:
            result.append((self.min, first_left - epsilon))
        
        # Add gaps between ranges
        for i in range(len(sorted_ranges) - 1):
            gap_start = sorted_ranges[i][1] + epsilon
            gap_end = sorted_ranges[i + 1][0] - epsilon
            
            if gap_start <= gap_end:
                result.append((gap_start, gap_end))
        
        # Add gap after last range
        last_right = sorted_ranges[-1][1]
        if last_right < self.max - epsilon:
            result.append((last_right + epsilon, self.max))
        
        return result
    
    def _comparison_to_range(self, comp: exp.Expression) -> List[Tuple[float, float]]:
        """
        Convert comparison to single range.
        
        Args:
            comp: Comparison expression
        
        Returns:
            List containing (left, right) bounds or empty if not our column
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
            # Not equal: complement of single point
            # (min, val-ε) ∪ (val+ε, max)
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
    # NULL HANDLING
    # ========================================================================
    
    def _check_is_null(self, expr: exp.Expression) -> Optional[bool]:
        """
        Check if expression is IS NULL or IS NOT NULL for our column.
        
        Args:
            expr: Expression to check
        
        Returns:
            True if IS NULL, False if IS NOT NULL, None otherwise
        """
        
        # NOT IS NULL
        if isinstance(expr, exp.Not):
            inner = expr.this
            if isinstance(inner, exp.Is):
                col = inner.this
                if (isinstance(col, exp.Column) and col.name == self.column and
                    isinstance(inner.expression, exp.Null)):
                    return False

        # IS NULL / IS NOT NULL
        if not isinstance(expr, exp.Is):
            return None
        
        # Check if it's for our column
        col = expr.this
        if not isinstance(col, exp.Column) or col.name != self.column:
            return None
        
        # IS NULL
        if isinstance(expr.expression, exp.Null):
            return True
        
        # IS NOT NULL
        if isinstance(expr.expression, exp.Not):
            inner = expr.expression.this
            if isinstance(inner, exp.Null):
                return False
        
        return None
    
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
        # Skip empty ranges
        valid_ranges = [(l, r) for l, r in ranges if l <= r]
        
        if not valid_ranges:
            return None
        
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
        
        # Hash intervals (int64 array — direct shift is equivalent to CPython hash for these values)
        intervals = _to_hash_values(intervals)
        
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
            
            # Independent estimates
            estimates.append(counts.sum(dim=1))

        return estimates

    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _extract_value(self, node: exp.Expression) -> float:
        """Extract numeric value from expression node."""
        if isinstance(node, exp.Literal):
            return pd.to_datetime(node.this).value if self.is_datetime else float(node.this)
        
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
        # create sketches for each interval
        self.sketches = dict()
        mask = distincts[self.column].notnull().to_numpy()[None, :]
        for interval in self.sorted_intervals:
            # create a sketch for the interval
            values = _to_hash_values(distincts[self.column] // interval)
            bins = self.bin_hash(values) % self.width
            signs = self.sign_hash(values) * mask

            # scale update by frequency of each value
            signs *= torch.tensor(distincts['_count'].to_numpy())[None, :].expand_as(bins)

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
        intervals = _to_hash_values(intervals)

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
    """
    CountMin sketch with complete predicate support.
    
    Extensions:
    - Tracks null_count for IS NULL predicates
    - Handles NOT by complementing ranges
    - Handles IS NOT NULL as full range
    - Handles IS TRUE/FALSE as specific values
    - Handles Parentheses by unwrapping
    """
    
    def __init__(self, data: pd.Series, depth: int, width: int, bin_hash: object, 
                 intervals: list = None, sample_selectivity=None, **kwargs):
        self.depth = depth
        self.width = width
        self.nrows = len(data)
        self.bin_hash = bin_hash
        self.sorted_intervals = tuple(sorted(intervals, reverse=True)) if intervals is not None else (1,)

        # Save type of data elements
        self.type = data.dtype.type
        self.is_datetime = pd.api.types.is_datetime64_any_dtype(data)

        # Check if type is a pandas datetime
        if self.is_datetime:
            # Convert to int (expected to be nanoseconds since epoch)
            data = data.view('int64')

        # Require that datatype is numeric
        assert pd.api.types.is_numeric_dtype(data), \
            f"CountMin only supports numeric data types, not {self.type}"

        # Track null count for IS NULL predicates
        self.null_count = data.isna().sum()

        # Save bounds of data (excludes NaN values)
        self.min = data.min()
        self.max = data.max()

        self.column = data.name

        # Creates a dataframe with only distinct rows and their counts
        distincts = data.value_counts(dropna=False).sort_values(ascending=False).reset_index(name='_count')
        assert distincts['_count'].sum() == self.nrows

        
        # truncate data if sample size is given
        self.scale_factor=1
        # if sample_selectivity is not None:
        #     if 0 < sample_selectivity < 1:
        #         # Treat as percentage
        #         sample_size = int(len(distincts) * sample_selectivity)
        #         sample_size = max(sample_size, 10_000)
        #     elif self.nrows > sample_selectivity >= 1:
        #         # Treat as absolute number
        #         sample_size = int(sample_selectivity)
        #     else:
        #         # Ignore
        #         print(f"IGNORING SAMPLING IN {type(self)}")
        #         sample_size = len(distincts)
        #     sample_size = min(len(distincts), sample_size)
        #     distincts = distincts.iloc[:sample_size]
        #     self.scale_factor = self.nrows / (distincts['_count'].sum())
        #     print(f" Scale Factor {self.scale_factor:.2f} ", end='')
        #     assert self.scale_factor >= 1

        # Vectorized function to convert each element to an int
        # Create sketches for each interval
        self.sketches = dict()
        mask = distincts[self.column].notnull().to_numpy()
        for interval in self.sorted_intervals:
            # Create a sketch for the interval
            values = _to_hash_values(distincts[self.column] // interval)
            bins = self.bin_hash(values) % self.width

            # Scale update by frequency of each non-null value
            counts = torch.tensor(distincts['_count'].to_numpy() * mask)[None, :].expand_as(bins)

            self.sketches[interval] = torch.zeros((self.depth, self.width), dtype=torch.long).scatter_add_(1, bins, counts)

        self.memory = sum(
            sketch.indices().nelement() * sketch.indices().element_size() +
            sketch.values().nelement() * sketch.values().element_size()
            if sketch.is_sparse
            else sketch.numel() * sketch.element_size()
            for sketch in self.sketches.values()
        )
        self.saved = dict()

    def memory_usage(self):
        return self.memory
    
    def __call__(self, predicates: exp.Expression, keys: dict, *args, **kwargs):
        if predicates is None:
            return 1, 0

        sel_id = predicate_to_canonical_string(predicates)
        cached = self.saved.get(sel_id)
        if cached is not None:
            return cached, 0

        if self.column not in self._get_columns_from_expr(predicates):
            self.saved[sel_id] = 1
            return 1, 0

        is_null = self._check_is_null(predicates)
        if is_null is not None:
            prob = self.null_count / self.nrows if is_null else 1.0 - (self.null_count / self.nrows)
            self.saved[sel_id] = prob
            return prob, 0

        ranges = self._extract_ranges(predicates)
        if not ranges:
            self.saved[sel_id] = 0
            return 0, 0

        estimates = self._estimate_ranges(ranges)
        est = sum(counts.min().item() for counts in estimates) if estimates else 0
        prob = min(max(0, (self.scale_factor * est) / self.nrows), 1)
        self.saved[sel_id] = prob
        return prob, 0


















































class StringCountMin(UnivariateEstimator):
    """
    CountMin sketch for string columns with complete predicate support.
    
    Extends UnivariateEstimator to inherit range algebra while adding:
    - Lexicographic encoding for range queries
    - N-gram sketches for LIKE patterns
    - Hybrid estimation routing
    
    Sketch Types:
    1. proxy_sketches: {interval: sketch} for range queries (>, <, =, etc.)
    2. prefix_sketch: For LIKE "abc%" patterns
    3. suffix_sketch: For LIKE "%xyz" patterns
    4. infix_sketch: For LIKE "%abc%" patterns
    """
    
    def __init__(self, data: pd.Series, depth: int, width: int, bin_hash: Callable,
                 ngram_size: int = 3, encoding_length: int = 4, 
                 intervals: List[int] = None, **kwargs):
        """
        Initialize StringCountMin.
        
        Args:
            data: String series
            depth: Number of hash functions
            width: Number of bins per hash
            bin_hash: Hash function
            ngram_size: N-gram size for LIKE (default 3)
            encoding_length: Chars to encode for ordering (default 8)
            intervals: Intervals for proxy sketches (default [1])
        """
        # Initialize parent (sets up basic attributes)
        super().__init__(**kwargs)
        
        self.depth = depth
        self.width = width
        self.nrows = len(data)
        self.bin_hash = bin_hash
        self.ngram_size = ngram_size
        self.encoding_length = encoding_length
        self.column = data.name
        
        # Null handling
        self.null_count = data.isna().sum()
        non_null_data = data.dropna()
        
        # ====================================================================
        # NUMERICAL PROXY (for range queries)
        # ====================================================================
        
        # Encode strings as sortable integers
        self.encoded_data = self._encode_strings(non_null_data)
        
        # Set min/max for range extraction (used by parent's methods)
        self.min = self.encoded_data.min() if len(self.encoded_data) > 0 else 0
        self.max = self.encoded_data.max() if len(self.encoded_data) > 0 else 0
        
        # Build proxy sketches
        self.sorted_intervals = tuple(sorted(intervals or [1024, 2048, 4196], reverse=True))
        self.sketches = self._build_proxy_sketches(non_null_data)  # Use 'sketches' like CountMin
        
        # ====================================================================
        # N-GRAM SKETCHES (for LIKE patterns)
        # ====================================================================
        
        self.prefix_sketch = self._build_prefix_sketch(non_null_data)
        self.suffix_sketch = self._build_suffix_sketch(non_null_data)
        self.infix_sketch = self._build_infix_sketch(non_null_data)
        
        self.memory = self._compute_memory()
        self.saved = dict()

    # ========================================================================
    # STRING ENCODING (for lexicographic ordering)
    # ========================================================================
    
    def _encode_strings(self, data: pd.Series) -> pd.Series:
        """Encode strings as sortable integers preserving lexicographic order."""
        def encode(s):
            if pd.isna(s):
                return np.nan
            s_trunc = str(s)[:self.encoding_length]
            result = 0
            for char in s_trunc:
                result = result * 256 + ord(char)
            for _ in range(len(s_trunc), self.encoding_length):
                result = result * 256
            return result
        return data.apply(encode)
    
    def _encode_single(self, s: str) -> int:
        """Encode single string to integer."""
        s_trunc = s[:self.encoding_length]
        result = 0
        for char in s_trunc:
            result = result * 256 + ord(char)
        for _ in range(len(s_trunc), self.encoding_length):
            result = result * 256
        return result
    
    # ========================================================================
    # PROXY SKETCHES (for range queries)
    # ========================================================================
    
    def _build_proxy_sketches(self, data: pd.Series) -> Dict[int, torch.Tensor]:
        """Build proxy sketches using encoded values."""
        value_counts = self.encoded_data.value_counts()
        
        if len(value_counts) == 0:
            return {i: torch.zeros((self.depth, self.width), dtype=torch.long) 
                    for i in self.sorted_intervals}
        
        sketches = {}
        for interval in self.sorted_intervals:
            values = _to_hash_values(value_counts.index.to_numpy() // interval)
            bins = self.bin_hash(values) % self.width
            counts = torch.tensor(value_counts.to_numpy(), dtype=torch.long)[None, :].expand_as(bins)
            sketches[interval] = torch.zeros((self.depth, self.width), dtype=torch.long).scatter_add_(1, bins, counts)
        
        return sketches
    
    # ========================================================================
    # N-GRAM SKETCHES (for LIKE patterns)
    # ========================================================================
    
    def _build_prefix_sketch(self, data: pd.Series) -> torch.Tensor:
        """Build prefix sketch (first n-gram)."""
        vc = data.value_counts()
        ngrams, freqs = [], []
        for s, c in vc.items():
            if len(s) >= self.ngram_size:
                ngrams.append(s[:self.ngram_size])
                freqs.append(c)
        return self._build_ngram_sketch(ngrams, freqs)
    
    def _build_suffix_sketch(self, data: pd.Series) -> torch.Tensor:
        """Build suffix sketch (last n-gram)."""
        vc = data.value_counts()
        ngrams, freqs = [], []
        for s, c in vc.items():
            if len(s) >= self.ngram_size:
                ngrams.append(s[-self.ngram_size:])
                freqs.append(c)
        return self._build_ngram_sketch(ngrams, freqs)
    
    def _build_infix_sketch(self, data: pd.Series) -> torch.Tensor:
        """
        Build infix sketch for all n-grams.
        
        FIXED: Counts each n-gram once per string (not per occurrence).
        
        Args:
            data: Series of string values
        
        Returns:
            Sketch tensor (depth × width)
        
        Example:
            String 'cha-cha-cha' appears 10 times
            
            Unique n-grams in string: {'cha', 'ha-', 'a-c', '-ch'}
            
            For 'cha':
            - Appears in 1 string (with 10 occurrences)
            - Added to sketch with count: 10 (once)
            
            Selectivity for LIKE '%cha%': 10 / 10 = 1.0 ✓
        """
        vc = data.value_counts()
        
        # Build n-gram → total count mapping
        ngram_counts = {}
        
        for s, c in vc.items():
            # Extract UNIQUE n-grams from this string
            seen_ngrams = set()
            for i in range(len(s) - self.ngram_size + 1):
                ng = s[i:i + self.ngram_size]
                seen_ngrams.add(ng)
            
            # Add count ONCE per unique n-gram
            for ng in seen_ngrams:
                ngram_counts[ng] = ngram_counts.get(ng, 0) + c
        
        # Convert to lists for sketch building
        if not ngram_counts:
            return torch.zeros((self.depth, self.width), dtype=torch.long)
        
        ngrams = list(ngram_counts.keys())
        freqs = list(ngram_counts.values())
        
        # Build sketch using existing method
        return self._build_ngram_sketch(ngrams, freqs)
    
    def _build_ngram_sketch(self, ngrams: List[str], freqs: List[int]) -> torch.Tensor:
        """Build sketch from n-grams using scatter_add."""
        if not ngrams:
            return torch.zeros((self.depth, self.width), dtype=torch.long)
        
        hashes = np.array([hash(ng) for ng in ngrams], dtype=np.int64)
        bins = self.bin_hash(hashes) % self.width
        freq_tensor = torch.tensor(freqs, dtype=torch.long)[None, :].expand_as(bins)
        sketch = torch.zeros((self.depth, self.width), dtype=torch.long)
        return sketch.scatter_add_(1, bins, freq_tensor)
    
    # ========================================================================
    # MAIN ENTRY POINT (routes to appropriate estimation method)
    # ========================================================================
    
    def __call__(self, predicates: exp.Expression, keys: dict, *args, **kwargs):
        if predicates is None:
            return 1, 0

        sel_id = predicate_to_canonical_string(predicates)
        cached = self.saved.get(sel_id)
        if cached is not None:
            return cached, 0

        if self.column not in self._get_columns_from_expr(predicates):
            self.saved[sel_id] = 1
            return 1, 0

        is_null = self._check_is_null(predicates)
        if is_null is not None:
            sel = self.null_count / self.nrows if is_null else 1.0 - (self.null_count / self.nrows)
            self.saved[sel_id] = sel
            return sel, 0

        if self._contains_like(predicates):
            sel = self._estimate_ngram(predicates)
        else:
            sel = self._estimate_range(predicates)

        sel = min(max(0, sel or 1.0), 1)
        self.saved[sel_id] = sel
        return sel, 0
    
    # ========================================================================
    # RANGE ESTIMATION (inherits _extract_ranges from parent)
    # ========================================================================
    
    def _estimate_range(self, predicates: exp.Expression) -> float:
        """Estimate using proxy sketches and inherited range extraction."""
        is_null_check = self._check_is_null(predicates)
        if is_null_check is not None:
            if is_null_check:
                return self.null_count / self.nrows if self.nrows > 0 else 0.0
            else:
                return (self.nrows - self.null_count) / self.nrows if self.nrows > 0 else 0.0
        

        # Use parent's _extract_ranges (inherits all AND/OR/NOT logic)
        ranges = self._extract_ranges(predicates)
        
        if not ranges:
            return 0.0
        
        # Estimate from ranges using parent's _estimate_ranges
        estimates = self._estimate_ranges(ranges)
        if not estimates:
            return 0.0
        
        est = sum(c.min().item() for c in estimates)
        return est / self.nrows if self.nrows > 0 else 0.0
    
    # ========================================================================
    # OVERRIDE: Comparison to Range (String-specific)
    # ========================================================================
    
    def _comparison_to_range(self, comp: exp.Expression) -> List[Tuple[float, float]]:
        """
        Override parent's method to handle string encoding.
        
        Converts string comparisons to ranges using encoded values.
        """
        if not isinstance(comp.left, exp.Column) or comp.left.name != self.column:
            return []
        
        # Extract string value
        val = self._extract_string_value(comp.right)
        if not val:
            return []
        
        # Encode for comparison
        enc = self._encode_single(val)
        epsilon = 1
        
        if isinstance(comp, exp.EQ):
            return [(enc, enc)]
        elif isinstance(comp, exp.NEQ):
            return [(self.min, enc - epsilon), (enc + epsilon, self.max)]
        elif isinstance(comp, exp.GT):
            return [(enc + epsilon, self.max)]
        elif isinstance(comp, exp.GTE):
            return [(enc, self.max)]
        elif isinstance(comp, exp.LT):
            return [(self.min, enc - epsilon)]
        elif isinstance(comp, exp.LTE):
            return [(self.min, enc)]
        else:
            return []
    
    # ========================================================================
    # OVERRIDE: Handle BETWEEN (String-specific)
    # ========================================================================
    
    def _handle_between(self, node: exp.Between) -> List[Tuple[float, float]]:
        """Override to use string encoding."""
        col = node.this
        if not isinstance(col, exp.Column) or col.name != self.column:
            return [(self.min, self.max)]
        
        low_str = self._extract_string_value(node.args.get('low'))
        high_str = self._extract_string_value(node.args.get('high'))
        
        if low_str is None or high_str is None:
            return [(self.min, self.max)]
        
        low_val = self._encode_single(low_str)
        high_val = self._encode_single(high_str)
        
        return [(low_val, high_val)]
    
    # ========================================================================
    # OVERRIDE: Handle IN (String-specific)
    # ========================================================================
    
    def _handle_in(self, node: exp.In) -> List[Tuple[float, float]]:
        """Override to use string encoding."""
        col = node.this
        if not isinstance(col, exp.Column) or col.name != self.column:
            return [(self.min, self.max)]
        
        ranges = []
        for val_node in node.expressions:
            string_val = self._extract_string_value(val_node)
            if string_val is not None:
                encoded = self._encode_single(string_val)
                ranges.append((encoded, encoded))
        
        # Use parent's _merge_ranges
        return self._merge_ranges(ranges)
    
    # ========================================================================
    # N-GRAM ESTIMATION (String-specific, not in parent)
    # ========================================================================
    
    def _estimate_ngram(self, expr: exp.Expression) -> float:
        """Estimate using n-gram sketches for LIKE patterns."""
        if isinstance(expr, exp.Paren):
            return self._estimate_ngram(expr.this)
        
        if isinstance(expr, exp.Not):
            inner = self._estimate_ngram(expr.this)
            return 1.0 - inner if inner is not None else None
        
        if isinstance(expr, exp.And):
            left, right = self._estimate_ngram(expr.this), self._estimate_ngram(expr.expression)
            if left is None or right is None:
                return left or right
            return left * right
        
        if isinstance(expr, exp.Or):
            left, right = self._estimate_ngram(expr.this), self._estimate_ngram(expr.expression)
            if left is None or right is None:
                return left or right
            return left + right - (left * right)
        
        if isinstance(expr, exp.Like):
            if not isinstance(expr.this, exp.Column) or expr.this.name != self.column:
                return None
            pattern = self._extract_string_value(expr.expression)
            return self._estimate_like(pattern) if pattern else None
        
        # Non-LIKE in mixed query - use range estimation
        return self._estimate_range(expr)
    
    def _estimate_like(self, pattern: str) -> float:
        """Estimate LIKE pattern using n-gram sketches."""
        constraints = self._decompose_pattern(pattern)
        if not constraints:
            return 1.0
        
        estimates = []
        for ctype, text in constraints:
            if len(text) < self.ngram_size:
                continue

            if ctype == 'prefix':
                counts = self._query_sketch(self.prefix_sketch, text[:self.ngram_size] if len(text) >= self.ngram_size else None)
            elif ctype == 'suffix':
                counts = self._query_sketch(self.suffix_sketch, text[-self.ngram_size:] if len(text) >= self.ngram_size else None)
            elif ctype == 'infix':
                counts = self._query_sketch(self.infix_sketch, text[:self.ngram_size] if len(text) >= self.ngram_size else None)
            else:
                continue
            
            if counts is not None:
                estimates.append(counts.min())
        
        if not estimates:
            return 1.0
        
        return min(estimates) / self.nrows if self.nrows > 0 else 0.0
    
    def _decompose_pattern(self, pattern: str) -> List[Tuple[str, str]]:
        """Decompose LIKE pattern into constraints."""
        if pattern == '%':
            return []
        
        parts = pattern.split('%')
        constraints = []
        
        if parts[0] and not pattern.startswith('%'):
            constraints.append(('prefix', parts[0]))
        if parts[-1] and not pattern.endswith('%'):
            constraints.append(('suffix', parts[-1]))
        for i in range(1, len(parts) - 1):
            if parts[i]:
                constraints.append(('infix', parts[i]))
        
        return constraints
    
    def _query_sketch(self, sketch: torch.Tensor, ngram: Optional[str]) -> Optional[np.ndarray]:
        """Query sketch for n-gram."""
        if ngram is None or len(ngram) < self.ngram_size:
            return np.array([self.nrows] * self.depth)
        
        h = np.array([hash(ngram)], dtype=np.int64)
        bins = self.bin_hash(h) % self.width
        return sketch.gather(1, bins).squeeze(1).numpy()
    
    # ========================================================================
    # HELPER METHODS (String-specific)
    # ========================================================================
    
    def _contains_like(self, expr: exp.Expression) -> bool:
        """Check if expression contains LIKE."""
        return isinstance(expr, exp.Like) or any(isinstance(c, exp.Like) for c in expr.iter_expressions())
    
    def _extract_string_value(self, node: exp.Expression) -> Optional[str]:
        """Extract string value from expression node."""
        if isinstance(node, exp.Literal):
            v = node.this
            return v[1:-1] if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')) else v
        try:
            return str(node.sql())
        except:
            return None
    
    def _compute_memory(self) -> int:
        """Compute total memory usage."""
        total = sum(s.numel() * s.element_size() for s in self.sketches.values())
        total += sum(s.numel() * s.element_size() for s in [self.prefix_sketch, self.suffix_sketch, self.infix_sketch])
        return total
    
    def memory_usage(self) -> int:
        """Return memory usage."""
        return self.memory