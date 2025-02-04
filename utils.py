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