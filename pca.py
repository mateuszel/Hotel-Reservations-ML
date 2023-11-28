import numpy as np
import pandas as pd

def PCA(df, no_of_pc):
    X = df.values
    mean_X = np.mean(X, axis=0)
    st_d_X = np.std(X, axis=0)
    
    standarized_X = (X-mean_X)/st_d_X
    cov_mat = np.cov(standarized_X, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    srted = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[srted]
    eigen_vectors = eigen_vectors[srted]

    eigenvector_subset = eigen_vectors[:, 0:no_of_pc]
    X_reduced = np.dot(eigenvector_subset.transpose(), standarized_X.transpose()).transpose()
     
    return pd.DataFrame(X_reduced, columns=[f'PC{i}' for i in range(1, no_of_pc+1)])

print(PCA(pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}), 2))