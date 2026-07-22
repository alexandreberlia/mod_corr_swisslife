"""
Diagnostics VAR / VECM — version corrigée
=========================================
Corrections principales :
1. hosking_portmanteau : ddl = K^2 (h - p) au lieu de K^2 h ; C0 cohérent avec Gamma_h
2. li_mcleod_portmanteau : le VRAI Li-McLeod (correction additive, sur les NIVEAUX)
3. mcleod_li_arch_test : l'ancien "li_mcleod_test" renommé pour ce qu'il est (test ARCH)
4. check_stability : gère les VECM (matrice compagnon via var_rep, K - r racines
   unitaires attendues PAR CONSTRUCTION)
5. vecm_portmanteau_test : multivarié (Hosking) au lieu de Ljung-Box univarié
6. lm_test supprimé (test_serial_correlation n'existe pas pour VARResults)
7. arch_test : un vrai test d'heteroscedasticite (ARCH-LM par equation + McLeod-Li)
8. chi2.sf au lieu de 1 - chi2.cdf
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.diagnostic import het_arch


# ======================================================================
# 1. STABILITE (VAR et VECM)
# ======================================================================

def check_stability(results, coint_rank=None, tol_unit=0.02):
    """
    VAR  : results.roots = racines du polynome caracteristique
           -> stable si TOUS les modules > 1 (convention statsmodels).
    VECM : pas d'attribut .roots ; on construit la matrice compagnon depuis
           la representation VAR (var_rep). Valeurs propres -> stable si
           module < 1, MAIS un VECM de rang r possede K - r valeurs propres
           egales a 1 PAR CONSTRUCTION. On valide donc :
           - les K - r plus grandes ~ 1
           - toutes les autres < 1
    """
    print()
    print("=" * 100)
    print("STABILITY TEST")
    print("=" * 100)
    print()

    if hasattr(results, "roots"):  # ---- VARResults ----
        moduli = np.abs(results.roots)
        table = pd.DataFrame({
            "Root": results.roots,
            "Modulus": moduli,
            "Outside Unit Circle": moduli > 1,
        })
        ok = bool((moduli > 1).all())
        print(table)
        print()
        print("Stable VAR system" if ok else "Unstable VAR system")
        return table

    # ---- VECMResults ----
    A = results.var_rep                      # (p, K, K)
    p, K, _ = A.shape
    companion = np.zeros((K * p, K * p))
    companion[:K, :] = np.hstack([A[i] for i in range(p)])
    if p > 1:
        companion[K:, :-K] = np.eye(K * (p - 1))

    eig = np.linalg.eigvals(companion)
    moduli = np.abs(eig)
    order = np.argsort(moduli)[::-1]
    eig, moduli = eig[order], moduli[order]

    r = coint_rank if coint_rank is not None else results.coint_rank
    n_unit = K - r                            # racines unitaires attendues

    expected_unit = np.zeros(len(eig), dtype=bool)
    expected_unit[:n_unit] = True

    table = pd.DataFrame({
        "Eigenvalue": eig,
        "Modulus": moduli,
        "Expected Unit Root": expected_unit,
    })
    print(table)
    print()

    unit_ok = np.allclose(moduli[:n_unit], 1.0, atol=tol_unit)
    rest_ok = bool((moduli[n_unit:] < 1.0).all())

    if unit_ok and rest_ok:
        print(f"Valid VECM dynamics: {n_unit} unit root(s) (= K - r, by "
              f"construction), all other eigenvalues inside the unit circle.")
    else:
        print("WARNING: eigenvalue pattern inconsistent with a rank-"
              f"{r} VECM (check rank / specification).")
    return table


# ======================================================================
# 2. PORTMANTEAU MULTIVARIES (residus en NIVEAUX)
# ======================================================================

def _autocov_matrices(resid, nlags):
    """C_h = (1/T) sum_t e_t e_{t-h}', residus centres, denominateur T."""
    resid = np.asarray(resid, dtype=float)
    resid = resid - resid.mean(axis=0)
    T, k = resid.shape
    C0 = resid.T @ resid / T
    C = [resid[h:].T @ resid[:-h] / T for h in range(1, nlags + 1)]
    return resid, T, k, C0, C


def hosking_portmanteau(residuals, nlags=12, n_model_lags=0, adjusted=True,
                        verbose=True, title="HOSKING PORTMANTEAU TEST"):
    """
    Portmanteau multivarie de Hosking sur les residus EN NIVEAUX.

    Parameters
    ----------
    residuals    : (T, K) residus du modele
    nlags        : horizon h du test (h > n_model_lags obligatoirement)
    n_model_lags : ordre dynamique du modele estime
                   VAR(p) -> p ; VECM -> k_ar_diff + 1 (approximation usuelle)
                   *** indispensable : ddl = K^2 (h - p) ***
    adjusted     : True -> poids T^2/(T-h) (petit echantillon)
                   False -> version brute (poids T)
    """
    _, T, k, C0, C = _autocov_matrices(residuals, nlags)
    C0_inv = np.linalg.inv(C0)

    Q = 0.0
    for h, Ch in enumerate(C, start=1):
        term = np.trace(Ch.T @ C0_inv @ Ch @ C0_inv)
        Q += term * (T ** 2 / (T - h) if adjusted else T)

    df = k ** 2 * (nlags - n_model_lags)
    if df <= 0:
        raise ValueError("nlags doit etre > n_model_lags (ddl = K^2(h-p)).")
    pvalue = chi2.sf(Q, df)

    if verbose:
        print()
        print("=" * 80)
        print(title + ("  (adjusted)" if adjusted else "  (raw)"))
        print("=" * 80)
        print()
        print(f"Q Statistic        : {Q:,.4f}")
        print(f"Degrees of Freedom : {df}   [= K^2 (h - p) = "
              f"{k}^2 x ({nlags} - {n_model_lags})]")
        print(f"P-value            : {pvalue:.6g}")
        print()
        if pvalue > 0.05:
            print("Cannot reject H0 : residuals resemble white noise.")
        else:
            print("Reject H0 : residual autocorrelation detected.")

    return {"Q": Q, "df": df, "pvalue": pvalue,
            "nlags": nlags, "n_model_lags": n_model_lags,
            "adjusted": adjusted}


def li_mcleod_portmanteau(residuals, nlags=12, n_model_lags=0, verbose=True):
    """
    VRAI test de Li-McLeod (1981) : statistique brute + correction ADDITIVE
        Q_LM = T * sum_h tr(C_h' C0^-1 C_h C0^-1) + K^2 h (h+1) / (2T)
    Sur les residus EN NIVEAUX (pas au carre !). ddl = K^2 (h - p).
    """
    _, T, k, C0, C = _autocov_matrices(residuals, nlags)
    C0_inv = np.linalg.inv(C0)

    Q = T * sum(np.trace(Ch.T @ C0_inv @ Ch @ C0_inv) for Ch in C)
    Q += k ** 2 * nlags * (nlags + 1) / (2 * T)

    df = k ** 2 * (nlags - n_model_lags)
    if df <= 0:
        raise ValueError("nlags doit etre > n_model_lags.")
    pvalue = chi2.sf(Q, df)

    if verbose:
        print()
        print("=" * 80)
        print("LI-McLEOD PORTMANTEAU TEST (levels, additive correction)")
        print("=" * 80)
        print()
        print(f"Q Statistic        : {Q:,.4f}")
        print(f"Degrees of Freedom : {df}")
        print(f"P-value            : {pvalue:.6g}")
        print()
        if pvalue > 0.05:
            print("Cannot reject H0 : residuals resemble white noise.")
        else:
            print("Reject H0 : residual autocorrelation detected.")

    return {"Q": Q, "df": df, "pvalue": pvalue}


# ======================================================================
# 3. TESTS D'HETEROSCEDASTICITE (vrais, cette fois)
# ======================================================================

def mcleod_li_arch_test(residuals, nlags=12):
    """
    Ex-"li_mcleod_test" : portmanteau sur les residus AU CARRE.
    C'est le test de McLeod-Li (1983) -> detecte l'ARCH, PAS
    l'autocorrelation des niveaux. Pas de correction de ddl ici
    (les carres ne sont pas contraints par l'estimation OLS du VAR).
    """
    resid_sq = np.asarray(residuals, dtype=float) ** 2
    return hosking_portmanteau(
        resid_sq, nlags=nlags, n_model_lags=0, adjusted=True,
        title="McLEOD-LI ARCH TEST (squared residuals)"
    )


def arch_test(residuals, names=None, nlags=4):
    """ARCH-LM d'Engle, equation par equation."""
    resid = pd.DataFrame(np.asarray(residuals))
    if names is not None:
        resid.columns = names

    print()
    print("=" * 100)
    print("ARCH-LM TEST (per equation)")
    print("=" * 100)
    print()

    rows = []
    for col in resid.columns:
        stat, pval, _, _ = het_arch(resid[col].values, nlags=nlags)
        rows.append({"Variable": col, "LM stat": stat, "P-value": pval,
                     "No ARCH": pval > 0.05})
    out = pd.DataFrame(rows)
    print(out)
    return out


# ======================================================================
# 4. WRAPPERS STATSMODELS (labels corrects)
# ======================================================================

def portmanteau_test(model, nlags=None, adjusted=True):
    """test_whiteness de statsmodels ; adjusted=True par defaut."""
    if nlags is None:
        nlags = model.k_ar + 10
    results = model.test_whiteness(nlags=nlags, adjusted=adjusted)

    print()
    print("=" * 100)
    print(f"PORTMANTEAU TEST (statsmodels, h={nlags}, adjusted={adjusted})")
    print("=" * 100)
    print()
    print(f"Statistic : {results.test_statistic}")
    print(f"P-value   : {results.pvalue:.6g}")
    print()
    if results.pvalue > 0.05:
        print("Cannot reject H0 : residuals are not autocorrelated.")
    else:
        print("Reject H0 : residual autocorrelation remains.")
    return results


def normality_test(model):
    """Jarque-Bera multivarie (ex-'homoskedasticity_test', mal nomme)."""
    results = model.test_normality()
    print()
    print("=" * 100)
    print("NORMALITY TEST (multivariate Jarque-Bera)")
    print("=" * 100)
    print(results)
    return results


# ======================================================================
# 5. OUTIL DE COMPARAISON — pour trancher les conclusions divergentes
# ======================================================================

def compare_portmanteau(model, nlags=12):
    """
    Les 4 statistiques sur les MEMES residus et le MEME horizon.
    Ordre attendu des statistiques : brut < Li-McLeod ~ Hosking ajuste.
    Si les conclusions divergent entre implementations a h identique,
    verifier d'abord les ddl utilises par chacune.
    """
    resid = model.resid
    p = model.k_ar

    print()
    print("#" * 100)
    print(f"COMPARISON ON IDENTICAL RESIDUALS  (h = {nlags}, p = {p}, "
          f"df = {model.neqs**2} x ({nlags} - {p}))")
    print("#" * 100)

    sm_raw = model.test_whiteness(nlags=nlags, adjusted=False)
    sm_adj = model.test_whiteness(nlags=nlags, adjusted=True)
    own_raw = hosking_portmanteau(resid, nlags, n_model_lags=p,
                                  adjusted=False, verbose=False)
    own_adj = hosking_portmanteau(resid, nlags, n_model_lags=p,
                                  adjusted=True, verbose=False)
    own_lm = li_mcleod_portmanteau(resid, nlags, n_model_lags=p,
                                   verbose=False)

    table = pd.DataFrame([
        {"Test": "statsmodels raw",      "Stat": sm_raw.test_statistic, "P-value": sm_raw.pvalue},
        {"Test": "statsmodels adjusted", "Stat": sm_adj.test_statistic, "P-value": sm_adj.pvalue},
        {"Test": "own Hosking raw",      "Stat": own_raw["Q"],          "P-value": own_raw["pvalue"]},
        {"Test": "own Hosking adjusted", "Stat": own_adj["Q"],          "P-value": own_adj["pvalue"]},
        {"Test": "own Li-McLeod",        "Stat": own_lm["Q"],           "P-value": own_lm["pvalue"]},
    ])
    print()
    print(table.to_string(index=False))
    print()
    print("Sanity check: raw < adjusted, and own vs statsmodels doivent "
          "quasiment coincider. Sinon -> bug d'implementation.")
    return table


# ======================================================================
# 6. VECM : PORTMANTEAU MULTIVARIE + RAPPORT DE VALIDATION
# ======================================================================

def vecm_portmanteau_test(vecm_results, nlags=12, k_ar_diff=None):
    """
    Portmanteau MULTIVARIE (Hosking) sur les residus du VECM.
    Capture aussi les autocorrelations CROISEES (que le Ljung-Box
    equation par equation rate completement).
    n_model_lags = k_ar_diff + 1 (ordre du VAR equivalent, approx. usuelle).
    """
    if k_ar_diff is None:
        k_ar_diff = getattr(vecm_results, "k_ar_diff", None)
        if k_ar_diff is None:
            raise ValueError("Preciser k_ar_diff (non trouve dans results).")

    resid = pd.DataFrame(vecm_results.resid, columns=vecm_results.names)
    return hosking_portmanteau(
        resid, nlags=nlags, n_model_lags=k_ar_diff + 1, adjusted=True,
        title="VECM MULTIVARIATE PORTMANTEAU (Hosking adjusted)"
    )


def vecm_validation_report(vecm_results, johansen_results,
                           portmanteau_results=None, alpha_signif=0.05):
    diagnostics = []
    n_variables = len(vecm_results.names)
    estimated_rank = johansen_results["rank"]
    vecm_rank = vecm_results.coint_rank

    diagnostics.append({
        "Diagnostic": "Valid Cointegration Rank",
        "Result": f"{vecm_rank}/{n_variables}",
        "Pass": 0 < vecm_rank < n_variables,
    })
    diagnostics.append({
        "Diagnostic": "Johansen Consistency",
        "Result": vecm_rank == estimated_rank,
        "Pass": vecm_rank == estimated_rank,
    })

    # Mecanisme de correction d'erreur : SIGNIFICATIVITE de alpha,
    # pas simple non-nullite numerique (toujours vraie a 1e-6 pres).
    try:
        alpha_p = np.asarray(vecm_results.pvalues_alpha)
        alpha_ok = bool((alpha_p < alpha_signif).any())
        alpha_msg = (f"{(alpha_p < alpha_signif).sum()} significant "
                     f"coefficient(s) at {alpha_signif:.0%}")
    except Exception:
        alpha_ok = bool(np.any(np.abs(vecm_results.alpha) > 1e-6))
        alpha_msg = "significance unavailable (fallback: non-zero check)"

    diagnostics.append({
        "Diagnostic": "Error Correction Mechanism (alpha significant)",
        "Result": alpha_msg,
        "Pass": alpha_ok,
    })

    if portmanteau_results is not None:
        white = portmanteau_results["pvalue"] > 0.05
        diagnostics.append({
            "Diagnostic": "Residual Whiteness (multivariate)",
            "Result": f"p = {portmanteau_results['pvalue']:.4g}",
            "Pass": bool(white),
        })

    report = pd.DataFrame(diagnostics)
    print()
    print("=" * 100)
    print("VECM VALIDATION REPORT")
    print("=" * 100)
    print()
    print(report.to_string(index=False))
    print()
    print("VECM ECONOMETRICALLY VALID" if report["Pass"].all()
          else "SOME DIAGNOSTICS REQUIRE ATTENTION")
    return report


# ======================================================================
# 7. RAPPORT GLOBAL VAR
# ======================================================================

def diagnostic_report(model, nlags=None):
    if nlags is None:
        nlags = model.k_ar + 10

    print()
    print("=" * 100)
    print("MODEL DIAGNOSTICS REPORT")
    print("=" * 100)

    stability = check_stability(model)
    portmanteau = portmanteau_test(model, nlags=nlags, adjusted=True)
    arch = arch_test(model.resid, names=model.names)
    normality = normality_test(model)

    return {
        "stability": stability,
        "portmanteau": portmanteau,
        "arch": arch,
        "normality": normality,
    }
