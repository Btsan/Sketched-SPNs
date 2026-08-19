from itertools import combinations
import warnings

# from sklearn.cross_decomposition import PLSCanonical as CCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import FeatureHasher
import pandas as pd
import numpy as np
from scipy.stats import rankdata

# Helper for the final ranking step
def ecdf(X):
    r = rankdata(X, method='max', axis=0, nan_policy='propagate')
    r = np.nan_to_num(r, copy=False, nan=0).astype(np.float32)
    r /= len(X)
    return r

def empirical_copula(data, types, max_onehot_dim=1024, max_discrete_dim=32, batch_size=10000):
    assert isinstance(data, pd.DataFrame)
    
    # 1. Setup Encoders
    # OHE for low cardinality
    one_hot = OneHotEncoder(max_categories=max_onehot_dim, sparse_output=True, handle_unknown='ignore')
    
    # Hasher for high cardinality columns
    # input_type='string' expects a list of strings per sample, so we wrap data later
    hasher = FeatureHasher(n_features=max_discrete_dim, input_type='string')

    copula = dict()
    
    for col in data:
        if types[col] == 'DISCRETE':
            n_unique = data[col].nunique()
            
            if (n_unique > len(data) * 0.95):
                copula[col] = np.random.uniform(size=(data.shape[0], 1))
                continue
            elif (n_unique > min(100_000, len(data) * 0.5)):
                
                # FeatureHasher expects an iterable of tokens (like ["cat", "dog"]).
                # use a generator to avoid creating the list in memory.
                raw_data_iter = ([str(x)] for x in data[col])
                
                # Transform -> Sparse Matrix -> Dense Array
                # efficient because max_discrete_dim is small (32)
                features = hasher.transform(raw_data_iter).toarray()
            else:
                features = data[col].to_numpy().reshape(-1, 1)
                one_hot.fit(features)
                
                gaussian = None
                proj = []
                
                # Use a deterministic seed for projection
                seed = abs(hash(col)) % (2**32)
                rng = np.random.default_rng(seed)

                for batch in range(0, features.shape[0], batch_size):
                    batch_features = features[batch:(batch + batch_size)]
                    embeddings_sparse = one_hot.transform(batch_features)
                    
                    curr_dim = embeddings_sparse.shape[1]
                    
                    if curr_dim > max_discrete_dim:
                        if gaussian is None:
                            gaussian = rng.normal(size=(curr_dim, max_discrete_dim))
                        
                        # Sparse @ Dense -> Dense (Safe & Fast)
                        batch_proj = embeddings_sparse @ gaussian
                        proj.append(batch_proj)
                    else:
                        proj.append(embeddings_sparse.toarray())

                features = np.concatenate(proj, axis=0)
            
            # Compute ECDF on the projected features
            copula[col] = ecdf(features)
            
        else:
            # Continuous data
            ranks = data[col].rank(method='max', na_option='keep').values
            ranks = np.nan_to_num(ranks, copy=False, nan=0).astype(np.float32)
            copula[col] = (ranks / len(data)).reshape(-1, 1)

    return copula

from sklearn.preprocessing import StandardScaler

def rdc_transform(data, types, k=20, s=1/6):
    """
    Transforms data for RDC calculation.
    """
    copula = empirical_copula(data, types)
    projections = []
    
    # We need a reproducible seed for the second projection layer too
    rng = np.random.default_rng(42)

    for col, features in copula.items():
        d = features.shape[-1]
        gaussian = rng.normal(loc=0.0, scale=s / np.sqrt(d), size=(d, k)).astype(np.float32)
        
        # Calculate projection for this column
        proj = features @ gaussian
        
        # Add bias term (b) for full Random Fourier Features: cos(wx + b)
        # bias = rng.uniform(0, 2*np.pi, size=(1, k))
        # proj += bias
        
        projections.append(proj)

    # Concatenate all features from all columns
    # Shape: (N, num_columns * k)
    nonlinear_projections = np.concatenate(projections, axis=1)
    
    # Apply nonlinearity
    nonlinear_projections = np.sin(nonlinear_projections)
    
    # Standardize
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

        # rdc_features = rdc_transform(data.sample(int(sample_size)) if 0 < sample_size < len(data) else data, 
        #                              meta_types, 
        #                              k=projected_dim,
        #                              s=projection_scale)
        rdc_features = generate_rdc_features_inplace(data.sample(int(sample_size)) if 0 < sample_size < len(data) else data, 
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
    # if N > (projected_dim * 10):
    var_thresh = var_thresh / N
    for i, j in combinations(range(n_cols), 2):
        if i in omit or j in omit:
            continue
        # x = np.stack(rdc_features[rdc_features.columns[i]])
        # y = np.stack(rdc_features[rdc_features.columns[j]])
        x = rdc_features[cols[i]].to_numpy()
        y = rdc_features[cols[j]].to_numpy()
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
                    rdc_matrix[i, j] = rdc_matrix[j, i] = rdc_cca_fast(x, y) # rdc_cca(x, y)
                except Exception:
                    # if CCA fails, set to 0
                    rdc_matrix[i, j] = rdc_matrix[j, i] = 0

    return rdc_matrix

# from scipy.linalg import eigh
def rdc_cca_fast(X, Y, reg_param=1e-5):
    """
    Computes the first canonical correlation coefficient between X and Y
    using direct linear algebra (SVD/Eigendecomposition).
    X, Y: centered matrices of shape (N, k)
    """
    # 1. Centering (Crucial for correlation)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    N = X.shape[0]
    
    # 2. Covariance Matrices
    # Add regularization to diagonal for numerical stability (like Ridge CCA)
    Cxx = (X.T @ X) / (N - 1) + reg_param * np.eye(X.shape[1])
    Cyy = (Y.T @ Y) / (N - 1) + reg_param * np.eye(Y.shape[1])
    Cxy = (X.T @ Y) / (N - 1)
    
    # 3. Solve Generalized Eigenvalue Problem (or SVD approach)
    # The canonical correlations are the singular values of:
    # Cxx^(-1/2) * Cxy * Cyy^(-1/2)
    
    # Efficient inversion using Cholesky or just inv (since k is small ~20)
    # inv(Cxx) is cheap because shape is (20, 20)
    Cxx_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cxx))
    Cyy_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cyy))
    
    # Omega = Cxx_inv_sqrt * Cxy * Cyy_inv_sqrt.T
    Omega = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt.T
    
    # The top singular value is the RDC
    return np.linalg.norm(Omega, ord=2)

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from scipy.stats import rankdata

def generate_rdc_features_inplace(data, types, k=20, s=1/6):
    """
    Generates RDC features directly into a pre-allocated array.
    Optimized for 36M+ rows.
    """
    N = len(data)
    n_cols = len(data.columns)
    
    # 1. PRE-ALLOCATE THE FINAL ARRAY
    # We use float32 to save 50% RAM. 
    # This reserves 14.4GB immediately. If this fails, you need np.memmap.
    print(f"Allocating {N} x {n_cols*k} feature matrix (~{N*n_cols*k*4/1e9:.2f} GB)...")
    try:
        final_features = np.zeros((N, n_cols * k), dtype=np.float32)
    except MemoryError:
        print("RAM insufficient. Switching to Disk-based Memory Mapping.")
        # Creates a file on disk effectively acting as RAM
        final_features = np.memmap('rdc_features.dat', dtype='float32', mode='w+', shape=(N, n_cols * k))

    # 2. SEEDING
    # We use a deterministic seed for reproducibility
    rng = np.random.default_rng()

    # 3. COLUMN-WISE PROCESSING
    # We process one column, write it to the big array, and delete temps immediately.
    
    for i, col in enumerate(data.columns):
        print(f"\t Processing column {col} ({i+1}/{n_cols})...")
        
        # --- A. EMPIRICAL COPULA STEP ---
        
        if types[col] == 'DISCRETE':
        # Case 1: High Cardinality Discrete (Hashing)
        # Use this for movie_id, person_id, etc.
            if (data[col].nunique() > min(50_000, len(data) * 0.2)):
                # Use strict Hashing. 
                # Note: We project directly to 'k' dimensions to skip the intermediate 32-dim step
                # if we want maximum speed, OR we hash to 32 then project to K.
                # Let's stick to the RDC structure: Hash(32) -> RandomProj(K)
                
                hasher = FeatureHasher(n_features=32, input_type='string')
                # Generator expression to avoid materializing list
                raw_iter = ([str(x)] for x in data[col])
                # Hash -> Dense
                copula_feat = hasher.transform(raw_iter).toarray().astype(np.float32)

            # Case 2: Low Cardinality Discrete (OHE)
            elif types[col] == 'DISCRETE':
                one_hot = OneHotEncoder(max_categories=1024, sparse_output=True, handle_unknown='ignore')
                features = data[col].to_numpy().reshape(-1, 1)
                one_hot.fit(features)
                
                gaussian = None
                proj = []

                for batch in range(0, features.shape[0], 100_000):
                    batch_features = features[batch:(batch + 100_000)]
                    embeddings_sparse = one_hot.transform(batch_features)
                    
                    curr_dim = embeddings_sparse.shape[1]
                    
                    # limit maximum discrete dimensions to 32
                    if curr_dim > 32:
                        if gaussian is None:
                            gaussian = rng.normal(size=(curr_dim, 32))
                        
                        # Sparse @ Dense -> Dense (Safe & Fast)
                        batch_proj = embeddings_sparse @ gaussian
                        proj.append(batch_proj)
                    else:
                        proj.append(embeddings_sparse.toarray())

                # Ensure output is float32!
                copula_feat = np.concatenate(proj, axis=0, dtype=np.float32)
            # copula_feat = (rankdata(copula_feat, axis=0) / N).astype(np.float32)
        # Case 3: Continuous
        else:
            ranks = data[col].rank(method='max', na_option='keep').values
            ranks = np.nan_to_num(ranks, copy=False, nan=0).astype(np.float32)
            copula_feat = (ranks / len(data)).reshape(-1, 1)

        # --- B. RDC NON-LINEAR PROJECTION STEP ---
        
        input_dim = copula_feat.shape[1]
        
        # Generate random weights (d, k)
        # Scale = s (constant bandwidth)
        W = rng.normal(loc=0.0, scale=s, size=(input_dim, k)).astype(np.float32)
        
        # Project: (N, d) @ (d, k) -> (N, k)
        # We calculate sin(XW)
        # We can write directly into the final array slice to save memory
        start_idx = i * k
        end_idx = start_idx + k
        
        # Option 1: Direct write (Fastest RAM usage)
        # np.sin(..., out=...) avoids creating a temporary return array
        np.sin(copula_feat @ W, out=final_features[:, start_idx:end_idx])

        # Standardize In-Place (Crucial for Clustering)
        # (x - mean) / std
        col_slice = final_features[:, start_idx:end_idx]
        mean = col_slice.mean(axis=0)
        std = col_slice.std(axis=0) + 1e-6 # Avoid div/0
        
        # In-place update
        final_features[:, start_idx:end_idx] -= mean
        final_features[:, start_idx:end_idx] /= std

        # Clean up temps to free RAM for next column
        del copula_feat, W
    
    # # Standardize
    # nonlinear_projections = StandardScaler().fit_transform(nonlinear_projections)
    
    columns = pd.MultiIndex.from_product([data.columns, range(k)], names=['col', 'feat'])
    return pd.DataFrame(final_features, columns=columns)

    # return final_features


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