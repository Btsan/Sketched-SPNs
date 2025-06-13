from itertools import combinations
import warnings

import numpy as np      
import pandas as pd
# from sklearn.cross_decomposition import PLSCanonical as CCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scipy.stats import rankdata

def ecdf(X):
    r = rankdata(X, method='max', axis=0, nan_policy='omit')
    r = np.nan_to_num(r, copy=False, nan=0) / len(X)
    return r

# max_discrete_dim should increase with the number of distincts per column
def empirical_copula(data, types, max_onehot_dim=1024, max_discrete_dim=32, batch_size=10000):
    assert type(data) == pd.DataFrame
    one_hot = OneHotEncoder(max_categories=max_onehot_dim)
    copula = dict()
    for col in data:
        features = data[col].values.reshape(-1, 1) # N, 1
        if types[col] == 'DISCRETE':
            if data[col].nunique() > len(data) * 0.95:
                # if all values are unique, randomly set all features
                features = np.random.normal(size=(features.shape[0], max_discrete_dim))
            else:
                one_hot = one_hot.fit(features)
                gaussian = None
                proj = []
                for batch in range(0, features.shape[0], batch_size):
                    batch_features = features[batch:(batch + batch_size)] # B, 1
                    embeddings = one_hot.transform(batch_features).toarray() # B, max_onehot_dim
                    if embeddings.shape[-1] > max_discrete_dim:
                        if gaussian is None:
                            gaussian = np.random.normal(size=(embeddings.shape[-1], max_discrete_dim))
                        embeddings = np.dot(embeddings, gaussian)
                    proj.append(embeddings)
                features = np.concatenate(proj, axis=0)
        copula[col] = ecdf(features).astype(np.float32)
    return copula

def rdc_transform(data, types, k=20, s=1/6): # 
    """
    Transforms data into a copula representation using empirical copula and random nonlinear projections.
    Args:
        data (pd.DataFrame): Input data.
        types (dict): Dictionary mapping column names to their types (e.g., 'DISCRETE').
        k (int): Number of features to project onto.
        s (float): Scaling factor for the random projections.
    Returns:
        pd.DataFrame: Transformed data with MultiIndex columns representing the original columns and their features.
    Note:
        Smaller s -> less susceptible to noise in data and captures more general patterns.
        Larger s -> amplifies noise in data and captures more localized patterns. 
    """
    copula = empirical_copula(data, types)
    # projections = dict()
    projections = []
    for col, features in copula.items():
        # random nonlinear projection
        gaussian = np.random.normal(size=(features.shape[-1], k)).astype(np.float32) * (s * features.shape[-1])
        # projections[col] = list(np.sin(np.matmul(features, gaussian)))
        projections.append(np.matmul(features, gaussian))
    nonlinear_projections = np.sin(np.concatenate(projections, axis=1))
    nonlinear_projections = StandardScaler().fit_transform(nonlinear_projections)
    columns = pd.MultiIndex.from_product([copula.keys(), range(k)], names=['col', 'feat'])
    return pd.DataFrame(nonlinear_projections, columns=columns)

def rdc_cca(x ,y):
    cca = CCA(n_components=1)
    x_cca, y_cca = cca.fit_transform(x, y)
    rdc = np.corrcoef(x_cca.T, y_cca.T,)[0, 1]
    return rdc

def rdc(data=None, meta_types=None, rdc_features=None, projected_dim=20, projection_scale=1/6, var_thresh=1e-3, sample_size=-1):
    if rdc_features is None:
        assert data is not None and meta_types is not None, f'data {data} meta_types are {meta_types}'

        # if 0 < sample_size < len(data):
        #     data = data.sample(int(sample_size))

        rdc_features = rdc_transform(data.sample(int(sample_size)) if 0 < sample_size < len(data) else data, 
                                     meta_types, 
                                     k=projected_dim,
                                     s=projection_scale)
    else:
        if 0 < sample_size < len(rdc_features):
            rdc_features = rdc_features.sample(int(sample_size))
    # N, n_cols = rdc_features.shape
    N = len(rdc_features)

    # rebuild MultiIndex to remove stale entries
    # because pandas MultiIndex doesn't automatically remove removed columns
    rdc_features.columns = pd.MultiIndex.from_tuples(rdc_features.columns.values, names=rdc_features.columns.names)
    n_cols = len(rdc_features.columns.levels[0])
    projected_dim = len(rdc_features.columns.levels[1])
    cols = rdc_features.columns.levels[0].tolist()

    # print(rdc_features.columns, cols)

    omit = set()
    if data is not None and meta_types is not None:
        for i, col in enumerate(data.columns):
            num_distincts = data[col].nunique()
            if meta_types[col] == 'DISCRETE' and num_distincts >= len(data) * 0.99:
                # if all values are unique, it is independent
                omit.add(i)
            elif num_distincts == 1:
                # if all values are the same, it is independent
                omit.add(i)
    
    # initialize dependency matrix
    rdc_matrix = np.eye(n_cols, dtype=np.float64)

    # only compute dependency if sample size is sufficiently large
    if N > (projected_dim * 10):
        var_thresh = var_thresh / N
        for i, j in combinations(range(n_cols), 2):
            if i in omit or j in omit:
                continue
            # x = np.stack(rdc_features[rdc_features.columns[i]])
            # y = np.stack(rdc_features[rdc_features.columns[j]])
            x = rdc_features[cols[i]].values
            y = rdc_features[cols[j]].values
            # rdc_matrix[i, j] = rdc_matrix[j, i] = rdc_cca(x, y)
            # early stop if x or y has low variance (e.g., lots of duplicates)
            if False and (np.var(x, axis=0).max() < var_thresh or np.var(y, axis=0).max() < var_thresh):
                # don't use this - hard to set a good threshold
                rdc_matrix[i, j] = rdc_matrix[j, i] = 0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    # try to compute CCA
                    # if it fails, set to 0
                    try:
                        rdc_matrix[i, j] = rdc_matrix[j, i] = rdc_cca(x, y)
                    except Exception:
                        # if CCA fails, set to 0
                        rdc_matrix[i, j] = rdc_matrix[j, i] = 0

    return rdc_matrix


if __name__ == '__main__':
    from time import perf_counter
    from dataset import get_dataframe
    import experiments

    # test on cast_info (36 million rows)
    test_table = 'cast_info'
    _, _, tables_meta = experiments.get_config('job-light')
    names = tables_meta[test_table]['names']
    types = tables_meta[test_table]['col_types']
    csv = get_dataframe(f"./End-to-End-CardEst-Benchmark-master/datasets/imdb/{test_table}.csv", names=names, columns=types.keys())
    print(csv.describe())

    projected_dim = 20
    max_discrete_dim = 32
    batch_size = 10000
    
    t0 = perf_counter()
    rdc_matrix_0, rdc_features = rdc(csv, types, projected_dim=projected_dim)
    t1 = perf_counter()
    print(f"[{t1-t0:,.2f} s] RDC complete (copula transform + correlation, size {len(csv):,})")

    pearsons = np.corrcoef(csv.map(hash).T)
    difference = abs(abs(pearsons) - rdc_matrix_0)
    print("Pearsons:")
    print(pearsons)
    print("RDC:")
    print(rdc_matrix_0)
    print("Difference = ")
    print(difference)
    print(f"Agreement with Pearson: {difference.max():,.2e}")

    test_sizes = [1e5, 1e4, 1e3, 1e2]

    for size in test_sizes:
        t0 = perf_counter()
        rdc_matrix = rdc(rdc_features=rdc_features, projected_dim=projected_dim, sample_size=size)
        t1 = perf_counter()
        difference = abs(rdc_matrix - rdc_matrix_0).max()
        print(f"[{t1-t0:,.2f} s] RDC (correlation only, sample size {min(size, len(rdc_features)):,}) complete. Largest difference = {difference:,.2e}")