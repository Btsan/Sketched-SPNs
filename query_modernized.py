"""
Query class using sqlglot primitives for predicate representation.

Key changes:
- selection_predicates: Dict[str, sqlglot.expressions.Expression]
- Direct use of sqlglot AST for predicates
- No custom PredicateTree classes needed
"""

from typing import Generator, Tuple, Dict, List, Set, Optional
from itertools import permutations
from collections import defaultdict
import random
import warnings
import re

import sqlglot
from sqlglot import expressions as exp


selection_ops_re = re.compile(r"(\>\=?|\<\=?|\<\>|\=|BETWEEN|IN|LIKE|NOT LIKE)")
attribute_re = re.compile(r"(_|[a-zA-Z])(_|\d|[a-zA-Z])*.(_|[a-zA-Z])+")


def text_between(input: str, start: str, end: str):
    """Extract text between two substrings."""
    if start is None:
        idx_start = 0
    else:
        idx_start = input.index(start)
    if end is None:
        idx_end = len(input)
    else:
        idx_end = input.index(end)
    return input[idx_start + len(start) + 1 : idx_end]

class Query(object):
    """
    Query representation using sqlglot primitives.
    
    Key attributes:
    - sql: Original SQL string
    - joins: List of join predicates (left, op, right)
    - selection_predicates: Dict[str, exp.Expression]
        Maps table alias to sqlglot expression tree
        Expression tree preserves AND/OR structure for proper
        inclusion-exclusion in cardinality estimation
    """
    
    sql: str
    joins: List[Tuple[str, str, str]]
    selection_predicates: Dict[str, exp.Expression]  # table → sqlglot expr
    node2component: Dict[str, int]
    num_components: int
    alias2joined_attrs: Dict[str, Dict[str, Tuple[int]]]

    def __init__(self, sql: str, history: Optional['Query'] = None):
        """
        Initialize Query with sqlglot-based predicates.
        
        Args:
            sql: SQL query string
            history: Optional previous Query instance for component ID reuse.
                    When provided, component IDs are assigned to maximize
                    reuse of existing sketch assignments across workload.
        """
        self.sql = sql
        
        # Parse with sqlglot
        try:
            self.parsed = sqlglot.parse_one(sql, read='postgres')
        except Exception as e:
            # Fallback: try without dialect
            self.parsed = sqlglot.parse_one(sql)
        
        # Extract joins from both ON clauses and WHERE clause
        self.joins = []
        
        # CRITICAL: Extract joins from ON clauses FIRST (explicit JOINs)
        # This works even when there's no WHERE clause
        self._extract_joins_from_on_clauses()
        
        # Then extract selections (and additional joins from WHERE)
        self.selection_predicates = self._extract_selection_predicates()
        
        # CRITICAL: Remove cyclic joins for Fast-AGMS correctness
        # Cyclic joins cause hash functions to not cancel properly
        removed_joins = self.remove_cyclic_joins()
        if removed_joins:
            print(f"Warning: Removed {len(removed_joins)} redundant join(s) to eliminate cycles:")
            for left, op, right in removed_joins:
                print(f"  {left} {op} {right}")
        
        # Label each transitive join component with history awareness
        if history:
            self.node2component, self.num_components, self.alias2joined_attrs, self.sketch_history = \
                self.component_labeling(self.joins, history.sketch_history)
        else:
            self.node2component, self.num_components, self.alias2joined_attrs, self.sketch_history = \
                self.component_labeling(self.joins)
    
    # ========================================================================
    # SQLGLOT-BASED PREDICATE EXTRACTION
    # ========================================================================
    
    def _extract_joins_from_on_clauses(self):
        """
        Extract join predicates from ON clauses in explicit JOIN syntax.
        
        This handles queries like:
            FROM t1 JOIN t2 ON t1.id = t2.id JOIN t3 ON t2.x = t3.x
        
        Works even when there's no WHERE clause (which was the bug).
        """
        # Find all JOIN nodes
        for join_node in self.parsed.find_all(exp.Join):
            # Get the ON clause
            on_clause = join_node.args.get('on')
            
            if on_clause:
                # Process ON clause to extract join predicates
                self._process_join_predicate(on_clause)
    
    def _process_join_predicate(self, node: exp.Expression):
        """
        Process ON clause predicate to extract joins.
        
        Handles:
        - Simple equi-joins: t1.id = t2.id
        - Multiple conditions: t1.id = t2.id AND t1.x = t2.x
        
        Args:
            node: sqlglot expression from ON clause
        """
        if isinstance(node, exp.And):
            # Multiple conditions joined by AND
            if node.this:
                self._process_join_predicate(node.this)
            if node.expression:
                self._process_join_predicate(node.expression)
        
        elif self._is_join_predicate(node):
            # Equi-join between tables
            left, op, right = self._extract_join_info(node)
            self.joins.append((left, op, right))
        
        # Note: We ignore non-equi-join conditions in ON clauses
        # These would need special handling
    
    def _extract_selection_predicates(self) -> Dict[str, exp.Expression]:
        """
        Extract selection predicates using sqlglot.
        
        Separates selections from joins and groups by table.
        Preserves AND/OR structure in sqlglot expression tree.
        
        Returns:
            Dict mapping table alias to sqlglot expression
        """
        # Get WHERE clause
        where = self.parsed.find(exp.Where)
        if not where:
            return {}
        
        predicate = where.this
        
        # Separate joins from selections
        selections_by_table: Dict[str, List[exp.Expression]] = {}
        self._process_predicate(predicate, selections_by_table)
        
        # Combine selections for each table
        result = {}
        for table, preds in selections_by_table.items():
            result[table] = self._combine_selections(preds)
        
        return result
    
    def _process_predicate(self, node: exp.Expression, selections: Dict[str, List[exp.Expression]]):
        """
        Recursively process predicate tree.
        
        Separates join predicates (equi-joins between tables)
        from selection predicates (filters on single table).
        
        Args:
            node: sqlglot expression node
            selections: Dict accumulating selection predicates by table
        """
        if isinstance(node, exp.And):
            # Process each child of AND
            # In sqlglot, And has 'this' and 'expression' args
            if node.this:
                self._process_predicate(node.this, selections)
            if node.expression:
                self._process_predicate(node.expression, selections)
        
        elif isinstance(node, exp.Or):
            # OR node - need to check if all children are on same table
            tables = self._get_tables_in_predicate(node)
            
            if len(tables) == 1:
                # All predicates in OR are on same table - it's a selection
                table = list(tables)[0]
                if table not in selections:
                    selections[table] = []
                selections[table].append(node)
            else:
                # OR spans multiple tables - process children separately
                if node.this:
                    self._process_predicate(node.this, selections)
                if node.expression:
                    self._process_predicate(node.expression, selections)
        
        elif self._is_join_predicate(node):
            # Equi-join between tables
            left, op, right = self._extract_join_info(node)
            self.joins.append((left, op, right))
        
        else:
            # Selection predicate on single table
            tables = self._get_tables_in_predicate(node)
            if len(tables) == 1:
                table = list(tables)[0]
                if table not in selections:
                    selections[table] = []
                selections[table].append(node)
            elif len(tables) == 0:
                # No tables (e.g., constant predicate) - skip
                pass
            else:
                # Multiple tables but not a join (complex predicate)
                # For now, skip or handle specially
                pass
    
    def _is_join_predicate(self, node: exp.Expression) -> bool:
        """Check if predicate is an equi-join between tables."""
        if not isinstance(node, exp.EQ):
            return False
        
        left = node.left
        right = node.right
        
        # Both sides must be columns
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            return False
        
        # Get table aliases
        left_table = self._get_table_from_column(left)
        right_table = self._get_table_from_column(right)
        
        # Must be from different tables
        return left_table != right_table and left_table and right_table
    
    def _extract_join_info(self, node: exp.EQ) -> Tuple[str, str, str]:
        """Extract (left, op, right) from join predicate."""
        left_col = node.left
        right_col = node.right
        
        left_table = self._get_table_from_column(left_col)
        left_name = left_col.name
        left_str = f"{left_table}.{left_name}"
        
        right_table = self._get_table_from_column(right_col)
        right_name = right_col.name
        right_str = f"{right_table}.{right_name}"
        
        return (left_str, '=', right_str)
    
    def _get_table_from_column(self, col: exp.Column) -> Optional[str]:
        """Get table alias from column expression."""
        if col.table:
            return col.table
        
        # Try to infer from context
        # This is a simplification - may need more sophisticated logic
        return None
    
    def _get_tables_in_predicate(self, node: exp.Expression) -> Set[str]:
        """Get all table aliases referenced in predicate."""
        tables = set()
        
        for col in node.find_all(exp.Column):
            table = self._get_table_from_column(col)
            if table:
                tables.add(table)
        
        return tables
    
    def _combine_selections(self, predicates: List[exp.Expression]) -> exp.Expression:
        """Combine multiple predicates with AND."""
        if len(predicates) == 0:
            return None
        elif len(predicates) == 1:
            return predicates[0]
        else:
            # Combine with AND
            result = predicates[0]
            for pred in predicates[1:]:
                result = exp.And(this=result, expression=pred)
            return result
    
    # ========================================================================
    # BACKWARD COMPATIBILITY (LEGACY DICT FORMAT)
    # ========================================================================
    
    @property
    def selects(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Legacy property for backward compatibility.
        
        Converts sqlglot expressions to nested dict format.
        WARNING: Loses OR structure!
        """
        result = {}
        
        for table, expr in self.selection_predicates.items():
            if expr is None:
                continue
            
            # Flatten expression to dict (loses structure!)
            result[table] = self._expr_to_dict(expr)
        
        return result
    
    def _expr_to_dict(self, expr: exp.Expression) -> Dict[str, Dict[str, str]]:
        """Convert sqlglot expression to legacy dict format."""
        result = {}
        
        # Find all comparison predicates
        for pred in self._flatten_predicates(expr):
            if isinstance(pred, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ)):
                left = pred.left
                right = pred.right
                
                if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                    col = left.name
                    op = self._get_operator_string(pred)
                    val = right.this
                    
                    if col not in result:
                        result[col] = {}
                    result[col][op] = str(val)
        
        return result
    
    def _flatten_predicates(self, expr: exp.Expression) -> List[exp.Expression]:
        """Flatten AND structure to list of predicates."""
        if isinstance(expr, exp.And):
            result = []
            if expr.this:
                result.extend(self._flatten_predicates(expr.this))
            if expr.expression:
                result.extend(self._flatten_predicates(expr.expression))
            return result
        else:
            return [expr]
    
    def _get_operator_string(self, expr: exp.Expression) -> str:
        """Get operator string from expression."""
        if isinstance(expr, exp.EQ):
            return '='
        elif isinstance(expr, exp.GT):
            return '>'
        elif isinstance(expr, exp.GTE):
            return '>='
        elif isinstance(expr, exp.LT):
            return '<'
        elif isinstance(expr, exp.LTE):
            return '<='
        elif isinstance(expr, exp.NEQ):
            return '!='
        else:
            return str(type(expr).__name__)
    
    # ========================================================================
    # ORIGINAL METHODS (UNCHANGED)
    # ========================================================================

    def __repr__(self) -> str:
        return self.sql

    def table_mapping_iter(self) -> Generator[Tuple[str, str], None, None]:
        """
        Iterate over (alias, table_name) pairs.
        
        Handles:
        - Implicit: FROM t1, t2, t3
        - Explicit: FROM t1 JOIN t2 ON ... JOIN t3 ON ...
        - Mixed: FROM t1, t2 JOIN t3 ON ...
        
        Yields:
            (alias, table_name) tuples
        """
        # Get FROM clause
        from_clause = self.parsed.find(exp.From)
        
        if from_clause:
            # First table in FROM
            table_expr = from_clause.this
            alias, name = self._extract_table_info(table_expr)
            if alias and name:
                yield alias, name
            
            # Additional tables in FROM (for implicit joins)
            # In sqlglot, multiple tables in FROM are stored in from_clause.expressions
            if hasattr(from_clause, 'expressions') and from_clause.expressions:
                for table_expr in from_clause.expressions:
                    alias, name = self._extract_table_info(table_expr)
                    if alias and name:
                        yield alias, name
        
        # Get all JOIN clauses (for explicit joins)
        for join in self.parsed.find_all(exp.Join):
            table_expr = join.this
            alias, name = self._extract_table_info(table_expr)
            if alias and name:
                yield alias, name


    def _extract_table_info(self, table_expr: exp.Expression) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract (alias, table_name) from table expression.
        
        Args:
            table_expr: sqlglot table expression
        
        Returns:
            (alias, table_name) tuple or (None, None)
        """
        # Handle Table expressions
        if isinstance(table_expr, exp.Table):
            table_name = table_expr.name
            
            # Get alias if present
            alias = table_expr.alias
            if not alias:
                alias = table_name  # Use table name as alias
            
            # Question: Include schema?
            # Currently returns just table name, not schema.table_name
            # Can be extended if needed
            
            return alias, table_name
        
        # Handle Alias expressions (subqueries, CTEs)
        elif isinstance(table_expr, exp.Alias):
            alias = table_expr.alias
            inner = table_expr.this
            
            if isinstance(inner, exp.Table):
                return alias, inner.name
            elif isinstance(inner, exp.Subquery):
                # For subqueries, use alias as both alias and "table name"
                # This is debatable - what should the table name be for a subquery?
                return alias, f"<subquery:{alias}>"
            else:
                return None, None
        
        else:
            return None, None

    def condition_iter(self) -> Generator[Tuple[str, str, str, bool], None, None]:
        """
        Iterate over all conditions (joins and selections).
        
        NOTE: This is kept for backward compatibility but is NOT used
        in the new sqlglot-based approach.
        
        Yields:
            (left, operator, right, is_selection)
        """
        # Remove closing semicolon if present
        if self.sql.endswith(";"):
            sql_query = self.sql[:-1]
        else:
            sql_query = self.sql

        selections = re.split(r"\sWHERE\s", sql_query)[1]

        if " BETWEEN " in selections:
            raise NotImplementedError("BETWEEN keyword not allowed")

        selections = re.split(r"\sAND\s", selections)

        for i, selection in enumerate(selections):
            left, op, right = selection_ops_re.split(selection)
            left = left.strip()
            right = right.strip()

            # With BETWEEN the next AND is part of BETWEEN
            if op == "BETWEEN":
                right += " AND " + selections[i + 1].strip()
                selections.pop(i + 1)

            is_selection = attribute_re.match(right) == None

            if attribute_re.match(left) == None:
                raise NotImplementedError(
                    "Selection values on the left are not supported"
                )

            if not is_selection and op != "=":
                raise ValueError(f"Must be equi-join but got: {op}")
            
            if right.endswith("::timestamp"):
                right = right[:-len("::timestamp")]

            if right[0] == right[-1] == "'":
                right = right[1:-1]

            yield left, op, right, is_selection

    def component_labeling(
        self, 
        joins: List[Tuple[str, str, str]], 
        sketch_history: Optional[Dict[str, Set[Tuple[int, Tuple[int, ...]]]]] = None
    ) -> Tuple[Dict[str, int], int, Dict[str, Dict[str, Tuple[int, ...]]], Dict[str, Set[Tuple[int, Tuple[int, ...]]]]]:
        """
        Label connected components and assign edge IDs with sketch history awareness.
        
        Minimizes unique sketches across workload by:
        - Reusing component IDs when nodes have been in same component before
        - Reusing edge ID assignments when nodes have same edge tuple signature
        - Tracking full (component_id, edge_tuple) signatures per node
        
        Args:
            joins: List of (left_node, op, right_node) tuples
            sketch_history: Historical (component_id, edge_tuple) signatures per node
                           Example: {'A.x': {(0, (0,)), (0, (0, 1))}}
        
        Returns:
            Tuple of:
            - node2component: Dict mapping nodes to component IDs
            - num_components: Number of distinct component IDs used
            - alias2joined_attrs: Dict[alias → Dict[attr → Tuple[edge_ids]]]
            - updated_history: Updated sketch history with new signatures
        
        Example:
            # Q1: A.x = B.x (no history)
            node2component = {'A.x': 0, 'B.x': 0}
            alias2joined_attrs = {'A': {'x': (0,)}, 'B': {'x': (0,)}}
            history = {'A.x': {(0, (0,))}, 'B.x': {(0, (0,))}}
            
            # Q2: B.x = C.x (with Q1 history)
            # B.x has signature (0, (0,)) → reuse component 0, edge 0!
            node2component = {'B.x': 0, 'C.x': 0}
            alias2joined_attrs = {'B': {'x': (0,)}, 'C': {'x': (0,)}}
            # B.x signature matches! Sketch reused!
        """
        from collections import Counter, defaultdict
        
        # Step 1: Standard DFS to identify connected components
        component_groups = self._find_connected_components(joins)
        
        # Step 2: Assign component IDs with history awareness
        node2component, used_component_ids = self._assign_component_ids_from_sketch_history(
            component_groups,
            sketch_history
        )
        
        num_components = len(used_component_ids)
        
        # Step 3: Assign edge IDs to maximize sketch reuse
        join2edge = self._assign_edge_ids_with_history(
            joins,
            node2component,
            sketch_history
        )
        
        # Step 4: Build alias2joined_attrs from join→edge mapping
        alias2joined_attrs = self._build_joined_attrs(joins, join2edge)
        
        # Step 5: Update sketch history with new signatures
        updated_history = self._update_sketch_history(
            node2component,
            alias2joined_attrs,
            sketch_history
        )
        
        return node2component, num_components, alias2joined_attrs, updated_history
    
    def _find_connected_components(self, joins: List[Tuple[str, str, str]]) -> List[Set[str]]:
        """Find connected components using DFS."""
        to_visit: Set[str] = set()
        component_groups: List[Set[str]] = []
        
        for join in joins:
            left, _, right = join
            to_visit.add(left)
            to_visit.add(right)
        
        def depth_first_search(node: str, current_group: Set[str]):
            """DFS to find all nodes in connected component."""
            current_group.add(node)
            
            for join in joins:
                left, _, right = join
                
                # Get the other node if this join involves current node
                if left == node:
                    other = right
                elif right == node:
                    other = left
                else:
                    continue
                
                # Visit other node if not yet visited
                if other in to_visit:
                    to_visit.remove(other)
                    depth_first_search(other, current_group)
        
        # Find all connected components
        while len(to_visit) > 0:
            node = to_visit.pop()
            current_group = set()
            depth_first_search(node, current_group)
            component_groups.append(current_group)
        
        return component_groups
    
    def _assign_component_ids_from_sketch_history(
        self,
        component_groups: List[Set[str]],
        sketch_history: Optional[Dict[str, Set[Tuple[int, Tuple[int, ...]]]]]
    ) -> Tuple[Dict[str, int], Set[int]]:
        """
        Assign component IDs with sketch history awareness.
        
        Extracts component IDs from (component_id, edge_tuple) signatures.
        """
        from collections import Counter
        
        node2component = {}
        next_component_id = 0
        used_component_ids = set()
        
        for group in component_groups:
            if sketch_history:
                # Extract historical component IDs from sketch signatures
                historical_ids = []
                for node in group:
                    if node in sketch_history:
                        # Extract just the component IDs
                        for comp_id, _ in sketch_history[node]:
                            historical_ids.append(comp_id)
                
                if historical_ids:
                    # Use most common historical ID that hasn't been used yet
                    id_counts = Counter(historical_ids)
                    
                    # Try IDs in order of frequency
                    component_id = None
                    for candidate_id, _ in id_counts.most_common():
                        if candidate_id not in used_component_ids:
                            component_id = candidate_id
                            break
                    
                    # If all historical IDs are used, assign new ID
                    if component_id is None:
                        component_id = next_component_id
                        while component_id in used_component_ids:
                            next_component_id += 1
                            component_id = next_component_id
                else:
                    # No history - assign new ID
                    component_id = next_component_id
                    while component_id in used_component_ids:
                        next_component_id += 1
                        component_id = next_component_id
            else:
                # No history - simple sequential assignment
                component_id = next_component_id
            
            # Assign this component ID to all nodes in group
            for node in group:
                node2component[node] = component_id
            
            used_component_ids.add(component_id)
            next_component_id = max(next_component_id, component_id + 1)
        
        return node2component, used_component_ids
    
    def _assign_edge_ids_with_history(
        self,
        joins: List[Tuple[str, str, str]],
        node2component: Dict[str, int],
        sketch_history: Optional[Dict[str, Set[Tuple[int, Tuple[int, ...]]]]]
    ) -> Dict[int, int]:
        """
        Assign edge IDs to joins to maximize sketch reuse.
        
        Strategy:
        - For each node, determine which joins it participates in
        - Try different edge ID assignments (permutations for small N, greedy for large)
        - Choose assignment maximizing nodes with matching (component, edges) signatures
        
        Returns:
            Dict mapping join_idx → edge_id
        """
        from collections import defaultdict
        
        # Map nodes to joins they participate in
        node2joins = defaultdict(list)
        for join_idx, join_tuple in enumerate(joins):
            left, op, right = join_tuple
            node2joins[left].append(join_idx)
            node2joins[right].append(join_idx)
        
        # For each node, find target edge tuples from history
        node_target_edges = {}
        for node in node2joins.keys():  # Explicitly iterate over keys
            if node not in node2component:
                # Skip nodes not in this query's components
                continue
            
            comp = node2component[node]
            degree = len(node2joins[node])  # Number of joins this node participates in
            
            if sketch_history and node in sketch_history:
                # Find historical signatures with matching component and degree
                matching_edges = [
                    edges for (c, edges) in sketch_history[node]
                    if c == comp and len(edges) == degree
                ]
                if matching_edges:
                    node_target_edges[node] = matching_edges
        
        num_joins = len(joins)
        
        if num_joins == 0:
            return {}
        
        # Choose strategy based on query size
        if num_joins <= 8:
            # Small queries: try all permutations (optimal)
            return self._optimal_edge_assignment(
                num_joins,
                node2joins,
                node_target_edges
            )
        else:
            # Large queries: greedy heuristic
            return self._greedy_edge_assignment(
                num_joins,
                node2joins,
                node_target_edges
            )
    
    def _optimal_edge_assignment(
        self,
        num_joins: int,
        node2joins: Dict[str, List[int]],
        node_target_edges: Dict[str, List[Tuple[int, ...]]]
    ) -> Dict[int, int]:
        """
        Find optimal edge assignment by trying all permutations.
        
        Returns:
            Best join_idx → edge_id mapping
        """
        
        best_assignment = None
        best_score = -1
        
        # Try each permutation of edge IDs
        for perm in permutations(range(num_joins)):
            join2edge = {join_idx: edge_id for join_idx, edge_id in enumerate(perm)}
            
            # Score this assignment: count nodes with matching signatures
            score = 0
            for node, join_indices in node2joins.items():
                # Compute this node's edge tuple with this assignment
                node_edges = tuple(sorted([join2edge[j] for j in join_indices]))
                
                # Check if it matches any historical signature
                if node in node_target_edges:
                    if node_edges in node_target_edges[node]:
                        score += 1
            
            # Update best if this is better
            if score > best_score:
                best_score = score
                best_assignment = join2edge
        
        # If no assignment found (shouldn't happen), use sequential
        if best_assignment is None:
            best_assignment = {i: i for i in range(num_joins)}
        
        return best_assignment
    
    def _greedy_edge_assignment(
        self,
        num_joins: int,
        node2joins: Dict[str, List[int]],
        node_target_edges: Dict[str, List[Tuple[int, ...]]]
    ) -> Dict[int, int]:
        """
        Greedy heuristic for large queries.
        
        Strategy:
        - Process joins in order
        - For each join, try available edge IDs
        - Choose edge ID that provides maximum immediate benefit
        """
        used_edges = set()
        join2edge = {}
        next_available = 0
        
        # Build reverse map: join → nodes
        join2nodes = {}
        for node, join_indices in node2joins.items():
            for join_idx in join_indices:
                if join_idx not in join2nodes:
                    join2nodes[join_idx] = []
                join2nodes[join_idx].append(node)
        
        # Process joins in order
        for join_idx in range(num_joins):
            best_edge = None
            best_score = -1
            
            # Try available edge IDs (limited search window)
            for candidate_edge in range(min(next_available, 0), next_available + 10):
                if candidate_edge in used_edges:
                    continue
                
                # Score: how many nodes would benefit?
                score = 0
                for node in join2nodes.get(join_idx, []):
                    if node not in node_target_edges:
                        continue
                    
                    # Compute partial edge tuple if we assign this edge
                    current_edges = [
                        join2edge[j] for j in node2joins[node]
                        if j in join2edge
                    ]
                    potential_edges = tuple(sorted(current_edges + [candidate_edge]))
                    
                    # Check if any target contains this as prefix
                    # (Node may have more joins not yet assigned)
                    for target in node_target_edges[node]:
                        if len(potential_edges) <= len(target):
                            # Check if potential is a subset of target
                            if all(e in target for e in potential_edges):
                                score += 1
                                break
                
                if score > best_score:
                    best_score = score
                    best_edge = candidate_edge
            
            # If no edge provides benefit, use next available
            if best_edge is None:
                while next_available in used_edges:
                    next_available += 1
                best_edge = next_available
            
            join2edge[join_idx] = best_edge
            used_edges.add(best_edge)
            next_available = max(next_available, best_edge + 1)
        
        return join2edge
    
    def _build_joined_attrs(
        self,
        joins: List[Tuple[str, str, str]],
        join2edge: Dict[int, int]
    ) -> Dict[str, Dict[str, Tuple[int, ...]]]:
        """
        Build alias2joined_attrs from join edge assignments.
        
        Returns:
            Dict[alias → Dict[attr → Tuple[edge_ids]]]
        """
        from collections import defaultdict
        
        # Track which edges each (alias, attr) participates in
        alias_attr_edges = defaultdict(lambda: defaultdict(list))
        
        for join_idx, join_tuple in enumerate(joins):
            left, op, right = join_tuple
            edge_id = join2edge[join_idx]
            
            # Parse left node (alias.attr)
            left_alias, left_attr = left.split('.')
            alias_attr_edges[left_alias][left_attr].append(edge_id)
            
            # Parse right node (alias.attr)
            right_alias, right_attr = right.split('.')
            alias_attr_edges[right_alias][right_attr].append(edge_id)
        
        # Convert to tuples and sort
        alias2joined_attrs = {}
        for alias, attrs in alias_attr_edges.items():
            alias2joined_attrs[alias] = {}
            for attr, edge_list in attrs.items():
                alias2joined_attrs[alias][attr] = tuple(sorted(edge_list))
        
        return alias2joined_attrs
    
    def _update_sketch_history(
        self,
        node2component: Dict[str, int],
        alias2joined_attrs: Dict[str, Dict[str, Tuple[int, ...]]],
        old_history: Optional[Dict[str, Set[Tuple[int, Tuple[int, ...]]]]]
    ) -> Dict[str, Set[Tuple[int, Tuple[int, ...]]]]:
        """
        Update sketch history with new (component, edges) signatures.
        
        Returns:
            Updated history with accumulated signatures
        """
        # Start with old history (deep copy of sets)
        new_history = {}
        if old_history:
            for node, signatures in old_history.items():
                new_history[node] = set(signatures)
        
        # Add current query's signatures
        for node, comp in node2component.items():
            # Get edge tuple for this node
            alias, attr = node.split('.')
            
            if alias in alias2joined_attrs and attr in alias2joined_attrs[alias]:
                edges = alias2joined_attrs[alias][attr]
            else:
                edges = ()  # No edges (shouldn't happen for nodes in joins)
            
            # Create signature: (component_id, edge_tuple)
            signature = (comp, edges)
            
            # Add to history
            if node not in new_history:
                new_history[node] = set()
            new_history[node].add(signature)
        
        return new_history

    def joins_of(self, table_id: str) -> List[Tuple[str, str, str]]:
        """Get all joins involving a table."""
        joins = []

        for join in self.joins:
            left, op, right = join

            id, _ = left.split(".")
            if id == table_id:
                joins.append(join)

            id, _ = right.split(".")
            if id == table_id:
                joins.append((right, op, left))

        return joins

    def joined_nodes(self, table_id: str) -> Set[str]:
        """Get all nodes (table.column) that this table joins with."""
        nodes: Set[str] = set()

        for join in self.joins:
            left, _, right = join

            id, _ = left.split(".")
            if id == table_id:
                nodes.add(left)

            id, _ = right.split(".")
            if id == table_id:
                nodes.add(right)

        return nodes

    def joined_with(self, node: str) -> Set[str]:
        """Get all nodes that join with this node."""
        nodes: Set[str] = set()

        for join in self.joins:
            left, _, right = join

            if left == node:
                nodes.add(right)

            if right == node:
                nodes.add(left)

        return nodes

    def random_node(self) -> str:
        """Get a random node from the join graph."""
        nodes = list(self.node2component.keys())
        idx = random.randint(0, len(nodes) - 1)
        return nodes[idx]
    
    # ========================================================================
    # CYCLE DETECTION AND REMOVAL FOR FAST-AGMS
    # ========================================================================
    
    def detect_cyclic_joins(self) -> bool:
        """
        Check if query has cyclic join conditions.
        
        Returns:
            True if query has cycles in join graph
        
        Example:
            query = Query("SELECT * FROM a, b, c WHERE a.x=b.x AND b.x=c.x AND a.x=c.x")
            has_cycles = query.detect_cyclic_joins()  # True
        """
        return self._has_cycle_in_join_graph()
    
    def get_redundant_joins(self) -> List[Tuple[str, str, str]]:
        """
        Get list of redundant join edges that create cycles.
        
        Returns:
            List of (left, op, right) tuples representing redundant joins
        
        Example:
            # Query: a.x=b.x AND b.x=c.x AND a.x=c.x
            redundant = query.get_redundant_joins()
            # Returns: [('a.x', '=', 'c.x')]  # Can be removed
        """
        if not self._has_cycle_in_join_graph():
            return []
        
        # Find spanning tree (acyclic subset)
        tree_edges = self._find_spanning_tree_joins()
        
        # Redundant edges = all edges - tree edges
        tree_set = set(tree_edges)
        redundant = []
        
        for join in self.joins:
            if join not in tree_set:
                # Check if reversed version is in tree
                left, op, right = join
                reversed_join = (right, op, left)
                if reversed_join not in tree_set:
                    redundant.append(join)
        
        return redundant
    
    def remove_cyclic_joins(self) -> List[Tuple[str, str, str]]:
        """
        Remove redundant join conditions that create cycles.
        
        This is REQUIRED for Fast-AGMS correctness because cyclic joins
        cause hash functions to not cancel properly during cross-correlation.
        
        Returns:
            List of removed join conditions (for logging)
        
        Raises:
            Warning: If non-removable cycles are detected
        
        Example:
            query = Query("SELECT * FROM a, b, c WHERE a.x=b.x AND b.x=c.x AND a.x=c.x")
            removed = query.remove_cyclic_joins()
            print(f"Removed {len(removed)} redundant joins")
            # Removed 1 redundant joins
        """
        if not self._has_cycle_in_join_graph():
            return []  # No cycles
        
        # Check for non-removable cycles first
        non_removable = self._detect_non_removable_cycles()
        if non_removable:
            self._warn_non_removable_cycles(non_removable)
        
        # Find spanning tree (acyclic subset)
        tree_edges = self._find_spanning_tree_joins()
        
        # Identify redundant edges
        redundant_joins = []
        new_joins = []
        
        tree_set = set(tree_edges)
        
        for join in self.joins:
            # Check if this join is in spanning tree
            left, op, right = join
            reversed_join = (right, op, left)
            
            if join in tree_set or reversed_join in tree_set:
                # Keep this join
                new_joins.append(join)
            else:
                # Remove this join (redundant)
                redundant_joins.append(join)
        
        # Update joins list
        self.joins = new_joins
        
        return redundant_joins
    
    def _has_cycle_in_join_graph(self) -> bool:
        """Check if join graph has any cycles using DFS."""
        if not self.joins:
            return False
        
        # Build adjacency list
        adj = defaultdict(set)
        nodes = set()
        
        for left, _, right in self.joins:
            adj[left].add(right)
            adj[right].add(left)
            nodes.add(left)
            nodes.add(right)
        
        # DFS with parent tracking
        visited = set()
        
        def dfs(node: str, parent: Optional[str]) -> bool:
            """Returns True if cycle detected."""
            visited.add(node)
            
            for neighbor in adj[node]:
                if neighbor not in visited:
                    if dfs(neighbor, node):
                        return True
                elif neighbor != parent:
                    # Found back edge → cycle!
                    return True
            
            return False
        
        # Check each connected component
        for node in nodes:
            if node not in visited:
                if dfs(node, None):
                    return True
        
        return False
    
    def _find_spanning_tree_joins(self) -> List[Tuple[str, str, str]]:
        """
        Find spanning tree of join graph using DFS.
        
        Returns:
            List of join tuples to keep (spanning tree edges)
        """
        if not self.joins:
            return []
        
        # Build adjacency list with edge info
        adj = defaultdict(list)
        nodes = set()
        
        for join in self.joins:
            left, op, right = join
            adj[left].append((right, join))
            adj[right].append((left, (right, op, left)))  # Store reversed
            nodes.add(left)
            nodes.add(right)
        
        # DFS to build spanning tree
        visited = set()
        tree_edges = []
        
        def dfs(node: str, parent: Optional[str]):
            """DFS to build spanning tree."""
            visited.add(node)
            
            for neighbor, join_tuple in adj[node]:
                if neighbor not in visited:
                    # Add edge to spanning tree
                    tree_edges.append(join_tuple)
                    # Continue DFS
                    dfs(neighbor, node)
        
        # Process each connected component
        for node in nodes:
            if node not in visited:
                dfs(node, None)
        
        return tree_edges
    
    def _detect_non_removable_cycles(self) -> List[List[Tuple[str, str, str]]]:
        """
        Detect cycles that cannot be removed (joins on different columns).
        
        A cycle is non-removable if the join predicates involve different columns.
        
        Example of non-removable cycle:
            A.x = B.x AND B.y = C.y AND A.x = C.x
            (A.x and B.y are different columns - can't infer A.x = C.x)
        
        Example of removable cycle:
            A.x = B.x AND B.x = C.x AND A.x = C.x
            (All on same column - can remove A.x = C.x)
        
        Returns:
            List of cycles where each cycle is a list of join tuples
        """
        non_removable_cycles = []
        
        # Build adjacency list with column tracking
        adj = defaultdict(list)
        nodes = set()
        
        for join in self.joins:
            left, op, right = join
            adj[left].append((right, join))
            adj[right].append((left, join))
            nodes.add(left)
            nodes.add(right)
        
        # Find all cycles using DFS
        visited = set()
        rec_stack = set()
        
        def dfs_find_cycles(node: str, parent: Optional[str], path: List[Tuple[str, str, str]]) -> List[List[Tuple[str, str, str]]]:
            """Find all cycles containing this node."""
            visited.add(node)
            rec_stack.add(node)
            cycles = []
            
            for neighbor, join_tuple in adj[node]:
                if neighbor == parent:
                    continue
                
                if neighbor not in visited:
                    # Continue DFS
                    sub_cycles = dfs_find_cycles(neighbor, node, path + [join_tuple])
                    cycles.extend(sub_cycles)
                elif neighbor in rec_stack:
                    # Found cycle - extract it
                    cycle = path + [join_tuple]
                    
                    # Check if this cycle is non-removable
                    if self._is_non_removable_cycle(cycle):
                        cycles.append(cycle)
            
            rec_stack.remove(node)
            return cycles
        
        # Find cycles from each component
        all_cycles = []
        for node in nodes:
            if node not in visited:
                cycles = dfs_find_cycles(node, None, [])
                all_cycles.extend(cycles)
        
        # Remove duplicates (cycles found from different starting nodes)
        unique_cycles = []
        seen = set()
        for cycle in all_cycles:
            # Normalize cycle (smallest node first)
            nodes_in_cycle = set()
            for left, _, right in cycle:
                nodes_in_cycle.add(left)
                nodes_in_cycle.add(right)
            cycle_key = frozenset(nodes_in_cycle)
            
            if cycle_key not in seen:
                seen.add(cycle_key)
                unique_cycles.append(cycle)
        
        return unique_cycles
    
    def _is_non_removable_cycle(self, cycle: List[Tuple[str, str, str]]) -> bool:
        """
        Check if a cycle is non-removable (joins on different columns).
        
        Args:
            cycle: List of join tuples forming a cycle
        
        Returns:
            True if cycle cannot be safely removed
        """
        if len(cycle) < 3:
            return False  # Need at least 3 edges for interesting cycles
        
        # Extract unique tables and their join columns
        table_columns = defaultdict(set)
        
        for left, _, right in cycle:
            left_table, left_col = left.split('.')
            right_table, right_col = right.split('.')
            
            table_columns[left_table].add(left_col)
            table_columns[right_table].add(right_col)
        
        # If any table joins on multiple different columns, cycle is non-removable
        for table, columns in table_columns.items():
            if len(columns) > 1:
                return True
        
        return False
    
    def _warn_non_removable_cycles(self, cycles: List[List[Tuple[str, str, str]]]):
        """
        Issue warnings for non-removable cycles.
        
        Args:
            cycles: List of non-removable cycles
        """
        for cycle in cycles:
            # Format cycle for display
            cycle_str = " AND ".join([f"{left} = {right}" for left, _, right in cycle])
            
            warnings.warn(
                f"Non-removable cycle detected: {cycle_str}. "
                f"This cycle involves joins on different columns and cannot be safely removed. "
                f"Fast-AGMS estimates may be incorrect for this query. "
                f"Consider rewriting the query to avoid this cycle pattern.",
                UserWarning,
                stacklevel=3
            )


# ============================================================================
# HELPER FUNCTIONS FOR WORKING WITH SQLGLOT EXPRESSIONS
# ============================================================================

def has_or(expr: exp.Expression) -> bool:
    """Check if expression contains OR."""
    if isinstance(expr, exp.Or):
        return True
    
    # Recursively check children
    for child in expr.iter_expressions():
        if has_or(child):
            return True
    
    return False


def count_predicates(expr: exp.Expression) -> int:
    """Count leaf predicates in expression."""
    if expr is None:
        return 0
    
    # Leaf predicates are comparisons
    comparisons = [exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ, 
                   exp.Like, exp.In, exp.Between]
    
    count = 0
    for node in expr.walk():
        if any(isinstance(node, cls) for cls in comparisons):
            count += 1
    
    return count


def get_predicate_sql(expr: exp.Expression) -> str:
    """Get SQL string for expression."""
    if expr is None:
        return ""
    return expr.sql()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Demonstrate sqlglot-based Query class."""
    
    print("="*70)
    print("QUERY CLASS WITH SQLGLOT PRIMITIVES")
    print("="*70)
    
    # Example 1: Simple AND query
    sql1 = """
    SELECT * 
    FROM employees e, departments d 
    WHERE e.dept_id = d.id 
      AND e.salary > 50000 
      AND e.age < 40
    """
    
    print("\n=== Example 1: Simple AND Query ===")
    query1 = Query(sql1)
    
    print(f"\nJoins: {len(query1.joins)}")
    for left, op, right in query1.joins:
        print(f"  {left} {op} {right}")
    
    print(f"\nSelection predicates:")
    for table, expr in query1.selection_predicates.items():
        print(f"\n  Table '{table}':")
        print(f"    Type: {type(expr).__name__}")
        print(f"    SQL: {get_predicate_sql(expr)}")
        print(f"    Has OR: {has_or(expr)}")
        print(f"    Count: {count_predicates(expr)}")
    
    # Example 2: Query with OR
    sql2 = """
    SELECT * FROM employees e
    WHERE (e.salary > 50000 OR e.salary < 20000)
      AND e.age < 40
    """
    
    print("\n=== Example 2: Query with OR ===")
    query2 = Query(sql2)
    
    print(f"\nSelection predicates:")
    for table, expr in query2.selection_predicates.items():
        print(f"\n  Table '{table}':")
        print(f"    Type: {type(expr).__name__}")
        print(f"    SQL: {get_predicate_sql(expr)}")
        print(f"    Has OR: {has_or(expr)}")
        print(f"    Count: {count_predicates(expr)}")

    stats67 = """
SELECT COUNT(*) 
FROM comments as c,
     posts as p,
     postLinks as pl,
     postHistory as ph,
     votes as v,
     users as u 
WHERE p.Id = pl.PostId 
  AND p.Id = ph.PostId 
  AND p.Id = c.PostId 
  AND u.Id = c.UserId 
  AND u.Id = v.UserId 
  AND c.Score=0 
  AND c.CreationDate>='2010-08-02 20:27:48'::timestamp 
  AND c.CreationDate<='2014-09-10 16:09:23'::timestamp 
  AND p.PostTypeId=1 
  AND p.Score=4 
  AND p.ViewCount<=4937 
  AND pl.CreationDate>='2011-11-03 05:09:35'::timestamp 
  AND ph.PostHistoryTypeId=1 
  AND u.Reputation<=270 
  AND u.Views>=0 
  AND u.Views<=51 
  AND u.DownVotes>=0;"""
    query67 = Query(stats67)
    print("\n=== Example: Stats 67 ===")
    print(f"\nJoins: {len(query67.joins)}")
    for left, op, right in query67.joins:
        print(f"  {left} {op} {right}")
    
    print(f"\nSelection predicates:")
    for table, expr in query67.selection_predicates.items():
        print(f"\n  Table '{table}':")
        print(f"    Type: {type(expr).__name__}")
        print(f"    SQL: {get_predicate_sql(expr)}")
        print(f"    Has OR: {has_or(expr)}")
        print(f"    Count: {count_predicates(expr)}")

    job33a = """
SELECT COUNT(*) FROM company_name cn1
JOIN movie_companies mc1 ON cn1.id = mc1.company_id
JOIN movie_info_idx mi_idx1 ON mc1.movie_id = mi_idx1.movie_id
JOIN info_type it1 ON it1.id = mi_idx1.info_type_id
JOIN movie_link ml ON mc1.movie_id = ml.movie_id
JOIN link_type lt ON lt.id = ml.link_type_id
JOIN movie_companies mc2 ON mc2.movie_id = ml.linked_movie_id
JOIN company_name cn2 ON cn2.id = mc2.company_id
JOIN movie_info_idx mi_idx2 ON mi_idx2.movie_id = ml.linked_movie_id
JOIN info_type it2 ON it2.id = mi_idx2.info_type_id
JOIN title t1 ON mc1.movie_id = t1.id
JOIN kind_type kt1 ON kt1.id = t1.kind_id
JOIN title t2 ON ml.linked_movie_id = t2.id
JOIN kind_type kt2 ON kt2.id = t2.kind_id
WHERE cn1.country_code = '[us]'
  AND it1.info = 'rating'
  AND it2.info = 'rating'
  AND kt1.kind IN ('tv series')
  AND kt2.kind IN ('tv series')
  AND lt.link IN ('sequel', 'follows', 'followed by')
  AND t2.production_year BETWEEN 2005
  AND it1.info = it2.info
  AND kt1.kind = kt2.kind
  AND mc2.movie_id = mi_idx2.movie_id
  AND mc2.movie_id = t2.id
  AND mi_idx1.movie_id = ml.movie_id
  AND mi_idx1.movie_id = t1.id
  AND mi_idx2.movie_id = t2.id
  AND ml.movie_id = t1.id;"""
    query33 = Query(job33a)
    print("\n=== Example: JOB 33 ===")
    print(f"\nJoins: {len(query33.joins)}")
    for left, op, right in query33.joins:
        print(f"  {left} {op} {right}")
    
    print(f"\nSelection predicates:")
    for table, expr in query33.selection_predicates.items():
        print(f"\n  Table '{table}':")
        print(f"    Type: {type(expr).__name__}")
        print(f"    SQL: {get_predicate_sql(expr)}")
        print(f"    Has OR: {has_or(expr)}")
        print(f"    Count: {count_predicates(expr)}")

    print(query33.joins)
    print(query33.alias2joined_attrs)

    job0 = """SELECT COUNT(*) FROM cast_info ci JOIN movie_companies mc ON ci.movie_id = mc.movie_id;"""
    query0 = Query(job0)
    print("\n=== Example: JOB 0 ===")
    print(f"\nJoins: {len(query0.joins)}")
    for left, op, right in query0.joins:
        print(f"  {left} {op} {right}")
    
    print(f"\nSelection predicates:")
    for table, expr in query0.selection_predicates.items():
        print(f"\n  Table '{table}':")
        print(f"    Type: {type(expr).__name__}")
        print(f"    SQL: {get_predicate_sql(expr)}")
        print(f"    Has OR: {has_or(expr)}")
        print(f"    Count: {count_predicates(expr)}")

    print(query0.joins)
    print(query0.alias2joined_attrs)

    for alias, name in query0.table_mapping_iter():
        print(f"  Alias: {alias}, Name: {name}")

    job71 = """SELECT * FROM complete_cast cc JOIN title t ON cc.movie_id = t.id WHERE t.production_year BETWEEN 1950 AND 2000;"""
    query71 = Query(job71)
    print("\n=== Example: JOB 71 ===")
    print(f"\nJoins: {len(query71.joins)}")
    for left, op, right in query71.joins:
        print(f"  {left} {op} {right}")
    
    print(f"\nSelection predicates:")
    for table, expr in query71.selection_predicates.items():
        print(f"\n  Table '{table}':")
        print(f"    Type: {type(expr).__name__}")
        print(f"    SQL: {get_predicate_sql(expr)}")
        print(f"    Has OR: {has_or(expr)}")
        print(f"    Count: {count_predicates(expr)}")

    print(query71.joins)
    print(query71.alias2joined_attrs)

    for alias, name in query71.table_mapping_iter():
        print(f"  Alias: {alias}, Name: {name}")

    return query67, query33

if __name__ == '__main__':
    stats67, job33 = example_usage()