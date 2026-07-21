import numpy as np

def psi(data, var_result, horizon):
    coefficients = var_coefficients(var_result)

    n = data.shape[1]
    p = var_result.k_ar

    A = []
    for i in range(1, p + 1):
        filtre = coefficients.index.astype(str).str.startswith(f"L{i}.")

        A_i = coefficients.loc[filtre].to_numpy()
        if A_i.shape != (n, n) and A_i.T.shape == (n, n):
            A_i = A_i.T

        if A_i.shape != (n, n):
            raise ValueError(
                f"A_{i} a pour dimension {A_i.shape}, "
                f"au lieu de {(n, n)}."
            )

        A.append(A_i)
    psi_matrices = [np.eye(n)]
    for j in range(1, horizon + 1):
        psi_j = np.zeros((n, n))

        for s in range(1, min(j, p) + 1):
            psi_j += A[s - 1] @ psi_matrices[j - s]

        psi_matrices.append(psi_j)

    return np.stack(psi_matrices)


    
  
