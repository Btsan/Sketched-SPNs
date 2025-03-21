from itertools import combinations

import numpy as np      
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import OneHotEncoder

from scipy.stats import rankdata

def ecdf(X):
    r = rankdata(X, method='max', axis=0, nan_policy='omit')
    r = np.nan_to_num(r, copy=False, nan=0) / len(X)
    return r

# max_discrete_dim should increase with the number of distincts per column
def empirical_copula(data, types, max_onehot_dim=2048, max_discrete_dim=64, batch_size=10000):
    assert type(data) == pd.DataFrame
    one_hot = OneHotEncoder(max_categories=max_onehot_dim)
    copula = dict()
    for col in data:
        features = data[col].values.reshape(-1, 1) # N, 1
        if types[col] == 'DISCRETE':
            if data[col].nunique() >= len(data) * 0.95:
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
        copula[col] = ecdf(features)
    return copula

def rdc_transform(data, types, k=20, s=1/6):
    copula = empirical_copula(data, types)
    projections = dict()
    for col, features in copula.items():
        # random nonlinear projection
        gaussian = np.random.normal(size=(features.shape[-1], k)) * (s * features.shape[-1])
        projections[col] = list(np.sin(np.matmul(features, gaussian)))
    return pd.DataFrame(projections)

def rdc_cca(x ,y):
    cca = CCA(n_components=1)
    x_cca, y_cca = cca.fit_transform(x, y)
    rdc = np.corrcoef(x_cca.T, y_cca.T,)[0, 1]
    return rdc

def rdc(data=None, types=None, rdc_features=None, projected_dim=20, var_thresh=1e-4, sample_size=-1):
    return_features = False
    if rdc_features is None:
        assert data is not None and types is not None

        if 0 < sample_size < len(data):
            data = data.sample(int(sample_size))

        rdc_features = rdc_transform(data, types, k=projected_dim)
        return_features = True
    else:
        if 0 < sample_size < len(rdc_features):
            rdc_features = rdc_features.sample(int(sample_size))
    N, n_cols = rdc_features.shape
    
    # initialize dependency matrix
    rdc_matrix = np.eye(n_cols, dtype=np.float64)

    # only compute dependency if sample size is sufficiently large
    if N > (projected_dim * 10):
        var_thresh = var_thresh / N
        for i, j in combinations(range(n_cols), 2):
            x = np.stack(rdc_features[rdc_features.columns[i]])
            y = np.stack(rdc_features[rdc_features.columns[j]])
            # rdc_matrix[i, j] = rdc_matrix[j, i] = rdc_cca(x, y)
            var_x = np.var(x, axis=0).max()
            var_y = np.var(y, axis=0).max()
            # print(f"{rdc_features.columns[i]}: var_x {var_x}, {rdc_features.columns[j]}: var_y {var_y}")
            # early stop if x or y has low variance (e.g., lots of duplicates)
            if var_x < var_thresh or var_y < var_thresh:
                rdc_matrix[i, j] = rdc_matrix[j, i] = 0
            else:
                rdc_matrix[i, j] = rdc_matrix[j, i] = rdc_cca(x, y)

    if return_features:
        return rdc_matrix, rdc_features
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