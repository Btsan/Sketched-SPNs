
import pandas as pd

def get_dataframe(csv, names=None, columns=None, dates=None):
    if columns:
        columns = list(columns)
    if names:
        df = pd.read_csv(csv,
                        names=list(names),
                        usecols=columns,
                        escapechar='\\',
                        encoding='utf-8',
                        on_bad_lines='skip',
                        low_memory=False)
    else:
        df = pd.read_csv(csv,
                        usecols=columns,
                        escapechar='\\',
                        encoding='utf-8',
                        on_bad_lines='skip',
                        low_memory=False)
    if columns:
        df =  df[columns]
    
    if dates:
        for col in dates:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def get_workload(csv):
    workload = pd.read_csv(csv,
                           delimiter='|',
                           names=['query', 1, 'parent', 2, 'cardinality'],
                           usecols=['query', 'parent', 'cardinality'],
                           )[['query', 'parent', 'cardinality']]
    workload['cardinality'] = workload['cardinality'].astype('float64')
    return workload

# def check_workload