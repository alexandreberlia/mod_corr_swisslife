import pandas as pd
import numpy as np


def check_stability(model):
    """
    Check system stability.

    Returns
    -------
    pd.DataFrame
    """

    roots = model.roots

    stability_df = pd.DataFrame({
        "Root": roots,
        "Modulus": np.abs(roots)
    })

    stability_df["Inside Unit Circle"] = (
        stability_df["Modulus"] < 1
    )

    stability = (
        stability_df["Inside Unit Circle"]
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
        nlags=12):
    """
    Breusch-Godfrey LM test.
    """

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
        nlags=12):
    """
    Complete diagnostic report.
    """

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
