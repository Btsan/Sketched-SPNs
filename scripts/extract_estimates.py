import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='extract cardinality estimates from CSV')
parser.add_argument('--csv', help='CSV filepath')
parser.add_argument('--out', help='output text filepath')
args = parser.parse_args()

csv = pd.read_csv(args.csv)

blacklist = ('query',
             'parent',
             'cardinality',
             'num_tables',
             'join_attributes',
             'similarity')

blacklist_substr = ('time', 'err')

estimation_col = None
for col in csv.columns:
    check_1 = col not in blacklist
    check_2 = True
    for substr in blacklist_substr:
        check_2 &= substr not in col

    if check_1 and check_2:
        assert estimation_col is None, f"Multiple columns ({estimation_col}, {col}) may be estimates"
        estimation_col = col

csv[estimation_col].astype(int).to_csv(args.out, header=False, index=False)