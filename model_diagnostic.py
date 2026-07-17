import pandas as pd
import numpy as np


def check_stability(model):

    roots = model.roots

    stability_df = pd.DataFrame({
        "Root": roots,
        "Modulus": np.abs(roots)
    })

    stability_df["Stable Root"] = (
        stability_df["Modulus"] > 1
    )

    stability = (
        stability_df["Stable Root"]
        .all()
    )

    print()
    print("=" * 100)
    print("STABILITY TEST")
    print("=" * 100)

    print()

    if stability:
        print("Stable VAR/VECM system")
    else:
        print("Unstable VAR/VECM system")

    return stability_df

def portmanteau_test(
        model,
        nlags=None):
    if nlags is None:
        nlags=model.k_ar+10

    results = model.test_whiteness(
        nlags=nlags
    )

    print()
    print("=" * 100)
    print("PORTMANTEAU TEST")
    print("=" * 100)

    print()

    print(
        f"Statistic : {results.test_statistic}"
    )

    print(
        f"P-value   : {results.pvalue}"
    )

    print()

    if results.pvalue > 0.05:
        print(
            "Cannot reject H0 : residuals are not autocorrelated."
        )
    else:
        print(
            "Reject H0 : residual autocorrelation remains."
        )

    return results

def lm_test(
    model,
    nlags=None):
    if nlags is None:
            nlags=model.k_ar+10

    results = model.test_serial_correlation(
    lags=nlags
    )

    print()
    print("=" * 100)
    print("LM TEST")
    print("=" * 100)

    print(results)

    return results

def homoskedasticity_test(model):
    """
    Residual heteroskedasticity test.
    """

    results = model.test_normality()

    print()
    print("=" * 100)
    print("NORMALITY TEST")
    print("=" * 100)

    print(results)

    return results

def heteroskedasticity_test(model):
    """
    Multivariate heteroskedasticity test.
    """

    results = model.test_whiteness()

    print()
    print("=" * 100)
    print("HETEROSKEDASTICITY CHECK")
    print("=" * 100)

    print(results)

    return results
def diagnostic_report(
        model,
        nlags=None):
    if nlags is None:
        nlags=model.k_ar+10

    print()
    print("=" * 100)
    print("MODEL DIAGNOSTICS REPORT")
    print("=" * 100)

    stability = check_stability(
        model
    )

    portmanteau = portmanteau_test(
        model,
        nlags=nlags
    )

    normality = homoskedasticity_test(
        model
    )

    return {
        "stability": stability,
        "portmanteau": portmanteau,
        "normality": normality
    }

from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd


def vecm_portmanteau_test(
        vecm_results,
        nlags=12):
    """
    Portmanteau / Ljung-Box test on VECM residuals.
    """

    residuals = pd.DataFrame(
        vecm_results.resid,
        columns=vecm_results.names
    )

    print()
    print("=" * 100)
    print("VECM PORTMANTEAU TEST")
    print("=" * 100)

    results = []

    for column in residuals.columns:

        test = acorr_ljungbox(
            residuals[column],
            lags=[nlags],
            return_df=True
        )

        pvalue = test["lb_pvalue"].iloc[-1]

        results.append({
            "Variable": column,
            "P-value": pvalue,
            "Residual White Noise": pvalue > 0.05
        })

    results_df = pd.DataFrame(results)

    print(results_df)

    return results_df

import pandas as pd
import numpy as np


def vecm_stability_test(vecm_results):
    """
    Stability test for VECM.

    Parameters
    ----------
    vecm_results : VECMResults

    Returns
    -------
    pd.DataFrame
    """

    var_rep = vecm_results.var_rep

    roots = np.linalg.eigvals(var_rep)

    stability_df = pd.DataFrame({

        "Root": roots,

        "Modulus": np.abs(roots)

    })

    stability_df["Stable Root"] = (
        stability_df["Modulus"] < 1
    )

    print()
    print("=" * 100)
    print("VECM STABILITY TEST")
    print("=" * 100)
    print()

    print(stability_df)

    print()

    if stability_df["Stable Root"].all():

        print(
            "Stable VECM system"
        )

    else:

        print(
            "Unstable VECM system"
        )

    return stability_df

def is_vecm_stable(vecm_results):

    roots = np.linalg.eigvals(
        vecm_results.var_rep
    )

    return np.all(
        np.abs(roots) < 1
    )
