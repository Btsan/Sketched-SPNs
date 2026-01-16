"""
Query class using sqlglot primitives for predicate representation.

Key changes:
- selection_predicates: Dict[str, sqlglot.expressions.Expression]
- Direct use of sqlglot AST for predicates
- No custom PredicateTree classes needed
"""

from typing import Generator, Tuple, Dict, List, Set, Optional
import re
import random
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

    def __init__(self, sql: str):
        """
        Initialize Query with sqlglot-based predicates.
        
        Args:
            sql: SQL query string
        """
        self.sql = sql
        
        # Parse with sqlglot
        try:
            self.parsed = sqlglot.parse_one(sql, read='postgres')
        except Exception as e:
            # Fallback: try without dialect
            self.parsed = sqlglot.parse_one(sql)
        
        # Extract join and selection predicates
        self.joins = []
        self.selection_predicates = self._extract_selection_predicates()
        
        # Label each transitive join component
        self.node2component, self.num_components = self.component_labeling(self.joins)

        # Label each attribute with their join(s)
        self.alias2joined_attrs: Dict[str, Dict[str, int]] = dict()
        for idx, join in enumerate(self.joins):
            left, _, right = join

            alias, attr = left.split(".")
            if alias not in self.alias2joined_attrs:
                self.alias2joined_attrs[alias] = dict()
            if attr not in self.alias2joined_attrs[alias]:
                self.alias2joined_attrs[alias][attr] = list()
            self.alias2joined_attrs[alias][attr].append(idx)

            alias, attr = right.split(".")
            if alias not in self.alias2joined_attrs:
                self.alias2joined_attrs[alias] = dict()
            if attr not in self.alias2joined_attrs[alias]:
                self.alias2joined_attrs[alias][attr] = list()
            self.alias2joined_attrs[alias][attr].append(idx)
        
        for alias in self.alias2joined_attrs:
            for attr in self.alias2joined_attrs[alias]:
                self.alias2joined_attrs[alias][attr] = tuple(
                    self.alias2joined_attrs[alias][attr]
                )
    
    # ========================================================================
    # SQLGLOT-BASED PREDICATE EXTRACTION
    # ========================================================================
    
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
        """Iterate over (alias, table_name) pairs."""
        if "WHERE" in self.sql:
            table_list = text_between(self.sql, "FROM", "WHERE")
        else:
            table_list = text_between(self.sql, "FROM", None)
        table_list = table_list.split(",")

        for table in table_list:
            table = table.strip()
            
            # First try splitting on AS otherwise split on space
            splits = re.split(" AS ", table, flags=re.IGNORECASE, maxsplit=1)
            if len(splits) == 1:
                splits = table.split(" ", maxsplit=1)
            
            name, alias = splits
            name = name.strip()
            alias = alias.strip()

            yield alias, name

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

    def component_labeling(self, joins: List[Tuple[str, str, str]]) -> Tuple[Dict[str, int], int]:
        """Label connected components in join graph."""
        to_visit: Set[str] = set()
        node2component: Dict[str, int] = {}
        num_components = 0

        for join in joins:
            left, _, right = join
            to_visit.add(left)
            to_visit.add(right)

        def depth_first_search(node: str, component: int):
            node2component[node] = component

            for join in joins:
                left, _, right = join

                # Get the other node if this join involves the current node
                if left == node:
                    other = right
                elif right == node:
                    other = left
                else:
                    continue

                # If the other node has already been visited then continue
                if other not in to_visit:
                    continue

                to_visit.remove(other)
                depth_first_search(other, component)

        while len(to_visit) > 0:
            node = to_visit.pop()
            depth_first_search(node, num_components)
            num_components += 1

        return node2component, num_components

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
    
    return query67, query33

if __name__ == '__main__':
    stats67, job33 = example_usage()