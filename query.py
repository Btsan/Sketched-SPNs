import numpy as np
import pandas as pd
from collections import defaultdict

class RelationNode(object):
    def __init__(self, name, keys, predicates={}):
        self.name = name
        self.keys = keys
        self.predicates = predicates
        self.sketch = None

def extract_graph(query):
    from_end = query.find('FROM') + 4
    where_start = query.find('WHERE')
    
    abbrv_to_table = {}
    for table_abbrv in query[from_end:where_start].split(','):
        table_abbrv = table_abbrv.strip().split(' ')
        table = table_abbrv[0].strip()
        abbrv = table_abbrv[-1].strip()
        abbrv_to_table[abbrv] = table

    keys = defaultdict(lambda: defaultdict(int))
    predicates = defaultdict(lambda: defaultdict(dict))
    joins = []

    where_end = where_start + 5
    for p in query[where_end:-1].split('AND'):
        if '=' in p:
            op = '='
        if '>=' in p:
            op = '>='
        elif '>' in p:
            op = '>'
        if '<=' in p:
            op = '<='
        elif '<' in p:
            op = '<'
        if '<>' in p or '!=' in p:
            op = '!='
        left, right = map(str.strip, p.split(op))
        ltable, lcol = left.split('.')
        triplet = [lcol, op, right]
        right = right.split('.')

        if len(right) == 2 and right[0] in abbrv_to_table:
            if op != '=':
                raise SystemExit(f'Unsupported join type {left}{op}{right[0]}.{right[1]}')
            else:
                rtable, rcol = right
                keys[abbrv_to_table[ltable]][lcol] += 1
                keys[abbrv_to_table[rtable]][rcol] += 1
                joins.append((abbrv_to_table[ltable], abbrv_to_table[rtable]))
        else:
            if triplet[2].startswith("'") and triplet[2].endswith("'"):
                triplet[2] = triplet[2].replace("'", '')
            elif triplet[2].startswith('"') and triplet[2].endswith('"'):
                triplet[2] = triplet[2].replace('"', '')
            elif triplet[2].endswith('::timestamp'):
                triplet[2] = pd.Timestamp(triplet[2][:-len('::timestamp')])
            else:
                triplet[2] = float(triplet[2])
            table = abbrv_to_table[ltable]
            val = triplet[2]
            assert op not in predicates[table][lcol], f'predicate on {table}.{lcol} was already specified in query'
            predicates[table][lcol][op] = val

    tables = [t for t in keys]
    nodes = [RelationNode(t, keys[t], predicates[t]) for t in tables]
    edges = np.zeros((len(tables), len(tables)), dtype=bool)
    for ltable, rtable in joins:
        lt = tables.index(ltable)
        rt = tables.index(rtable)
        edges[lt, rt] = True
        edges[rt, lt] = True
    return nodes, edges

def predicates_keys_from(query):
    from_end = query.find('FROM') + 4
    where_start = query.find('WHERE')

    tables_and_abbrvs = query[from_end:where_start].split(',')
    abbrv_to_table = {}
    keys = defaultdict(set)
    join_count = {}
    predicates = {}
    for table_abbrv in tables_and_abbrvs:
        table_name_abbrv = table_abbrv.strip().split(' ')
        table_name = table_name_abbrv[0].strip()
        abbrv = table_name_abbrv[-1].strip()
        abbrv_to_table[abbrv] = table_name
        # keys[table_name] = None
        join_count[table_name] = 0
        predicates[table_name] = {}
        
    num_tables = len(keys)

    where_end = where_start + 5
    and_list = query[where_end:-1].split('AND')

    for p in and_list:
        if '=' in p:
            op = '='
        if '>=' in p:
            op = '>='
        elif '>' in p:
            op = '>'
        if '<=' in p:
            op = '<='
        elif '<' in p:
            op = '<'
        left, right = p.split(op)
        left = left.split('.')
        triplet = [
                left[1].strip(),
                op,
                right.strip()
        ]
        right = right.split('.')
        left = list(map(str.strip, left))
        right = list(map(str.strip, right))
        assert len(left) == 2, f'predicate: {p} --> ({left}, {op}, {right})'
        if len(right) == 2 and op == '=' and right[0] in abbrv_to_table:
            keys[abbrv_to_table[left[0]]].add(left[1])
            keys[abbrv_to_table[right[0]]].add(right[1])

            # increment join counter for each table
            join_count[abbrv_to_table[left[0]]] += 1
            join_count[abbrv_to_table[right[0]]] += 1
        else:
            if triplet[2][0] == "'" and triplet[2][-1] == "'":
                # remove ticks from strings
                triplet[2] = triplet[2].replace("'", '')
            elif triplet[2][0] == '"' and triplet[2][-1] == '"':
                # remove ticks from strings
                triplet[2] = triplet[2].replace('"', '')
            elif triplet[2].endswith('::timestamp'):
                triplet[2] = pd.Timestamp(triplet[2][:-len('::timestamp')])
            else:
                triplet[2] = float(triplet[2])
            table = abbrv_to_table[left[0]]
            col = triplet[0]
            val = triplet[2]
            if table not in predicates:
                predicates[table] = {col: {}}
            if col not in predicates[table]:
                predicates[table][col] = {}
            predicates[table][col][op] = val 

    return predicates, dict(keys)

from typing import Generator, Tuple, Dict, List, Set
import re
import random

selection_ops_re = re.compile(r"(\>\=?|\<\=?|\<\>|\=|BETWEEN|IN|LIKE|NOT LIKE)")
attribute_re = re.compile(r"(_|[a-zA-Z])(_|\d|[a-zA-Z])*.(_|[a-zA-Z])+")


def text_between(input: str, start: str, end: str):
    # getting index of substrings
    idx_start = input.index(start)
    idx_end = input.index(end)

    # length of substring 1 is added to
    # get string from next character
    return input[idx_start + len(start) + 1 : idx_end]

class Query(object):
    sql: str
    joins: List[Tuple[str, str, str]]
    selects: Dict[str, Dict[str, Dict[str, str]]]
    node2component: Dict[str, int]
    num_components: int
    alias2joined_attrs: Dict[str, Dict[str, Tuple[int]]]

    def __init__(self, sql: str):
        self.sql = sql

        # extract join and selection predicates
        self.joins = []
        self.selects = dict()
        for left, op, right, is_select in self.condition_iter():
            if is_select:
                alias, col = left.split('.')
                if alias not in self.selects:
                    self.selects[alias] = {col: dict()}
                elif col not in self.selects[alias]:
                    self.selects[alias][col] = dict()
                self.selects[alias][col][op] = right
            else:
                self.joins.append((left, op, right))

        # label each transitive join component
        self.node2component, self.num_components = self.component_labeling(self.joins)

        # label each attribute with their join(s)
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
                self.alias2joined_attrs[alias][attr] = tuple(self.alias2joined_attrs[alias][attr])

    def __repr__(self) -> str:
        return self.sql

    def table_mapping_iter(self) -> Generator[Tuple[str, str], None, None]:

        table_list = text_between(self.sql, "FROM", "WHERE")
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

        # remove closing semicolon if present
        if self.sql.endswith(";"):
            sql_query = self.sql[:-1]
        else:
            sql_query = self.sql

        selections = re.split("\sWHERE\s", sql_query)[1]

        if " OR " in selections:
            raise NotImplementedError("OR selections are not supported yet.")

        if " BETWEEN " in selections:
            raise NotImplementedError("BETWEEN keyword not allowed")

        selections = re.split("\sAND\s", selections)
        # print(selections)

        # TODO support more complicated LIKE and OR statements
        # TODO support for parentheses

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

    def component_labeling(self, joins: List[Tuple[str, str, str]]) -> Dict[str, int]:
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

                # get the other node if this join involves the current node
                # if not then continue to the next join
                if left == node:
                    other = right
                elif right == node:
                    other = left
                else:
                    continue

                # if the other node has already been visited then continue
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
        # ensures that left always has the table id attribute
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
        nodes: Set[str] = set()

        for join in self.joins:
            left, _, right = join

            if left == node:
                nodes.add(right)

            if right == node:
                nodes.add(left)

        return nodes

    def random_node(self) -> str:
        nodes = list(self.node2component.keys())
        idx = random.randint(0, len(nodes) - 1)
        return nodes[idx]