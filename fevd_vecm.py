import numpy as np
from scipy.linalg import cholesky


def vecm_fevd(alpha, beta, gammas, sigma_u,
              horizon=40):
    """
    FEVD orthogonalisée d'un VECM.

    Parameters
    ----------
    alpha : (n,r)
    beta  : (n,r)
    gammas : list of Gamma_i
        longueur p-1
    sigma_u : (n,n)
        covariance résiduelle du VECM
    horizon : int

    Returns
    -------
    fevd : (horizon,n,n)
        fevd[h,i,j]
        contribution du choc j
        à la variance de prévision de la variable i
        à l'horizon h+1.
    """

    n = alpha.shape[0]
    p = len(gammas) + 1

    # --------------------------------------------------
    # Conversion VECM -> VAR(p)
    # --------------------------------------------------

    Pi = alpha @ beta.T

    A = []

    if p == 1:
        A.append(np.eye(n) + Pi)

    else:

        A1 = np.eye(n) + Pi + gammas[0]
        A.append(A1)

        for i in range(1, p - 1):
            A.append(gammas[i] - gammas[i - 1])

        A.append(-gammas[-1])

    # --------------------------------------------------
    # Coefficients MA(C_h)
    # --------------------------------------------------

    C = [np.eye(n)]

    for h in range(1, horizon):

        Ch = np.zeros((n, n))

        for j in range(1, min(h, p) + 1):
            Ch += A[j - 1] @ C[h - j]

        C.append(Ch)

    # --------------------------------------------------
    # Orthogonalisation Cholesky
    # --------------------------------------------------

    P = cholesky(sigma_u, lower=True)

    Theta = [Ch @ P for Ch in C]

    # --------------------------------------------------
    # FEVD
    # --------------------------------------------------

    fevd = np.zeros((horizon, n, n))

    for H in range(1, horizon + 1):

        num = np.zeros((n, n))
        den = np.zeros(n)

        for h in range(H):

            th = Theta[h]

            num += th**2
            den += np.sum(th**2, axis=1)

        fevd[H-1] = num / den[:, None]

    return fevd


from scipy.linalg import null_space


def permanent_impact_matrix(alpha, beta, gammas):

    n = alpha.shape[0]

    alpha_perp = null_space(alpha.T)
    beta_perp = null_space(beta.T)

    G = np.eye(n)

    for gamma in gammas:
        G -= gamma

    C1 = (
        beta_perp
        @ np.linalg.inv(
            alpha_perp.T @ G @ beta_perp
        )
        @ alpha_perp.T
    )

    return C1
