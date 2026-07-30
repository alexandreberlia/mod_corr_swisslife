"""demo_leadlag.py — validation et mode d'emploi de leadlag.py

Lancer :  python demo_leadlag.py
"""
import numpy as np
import pandas as pd

from leadlag import leadlag_ccf, leadlag_probit, scan


def ar1(n, rho, rng):
    a = np.zeros(n)
    for t in range(1, n):
        a[t] = rho * a[t - 1] + rng.normal()
    return a


# --- 1. Validation : avance connue de +5 -------------------------------------
rng = np.random.default_rng(42)
n = 300
x = ar1(n, 0.85, rng)
y = np.roll(x, 5) + rng.normal(0, 1.2, n)
y[:5] = rng.normal(0, 1.2, 5)

print("=" * 72)
print("1. VALIDATION — la vraie avance est +5")
print("=" * 72)
r = leadlag_ccf(x, y, max_lag=12, prewhiten_series=True, n_boot=800)
print(r.summary())
print("\nCCF brute autour du pic   :", np.round(
    leadlag_ccf(x, y, max_lag=12, prewhiten_series=False, n_boot=1).stat[10:20], 2))
print("CCF pre-blanchie          :", np.round(r.stat[10:20], 2))
print("-> la CCF brute etale le signal sur 8 decalages, la blanchie en isole un.")

# --- 2. Taille du test sous H0 -----------------------------------------------
print("\n" + "=" * 72)
print("2. TAILLE DU TEST — deux AR(1) rho=0.9 INDEPENDANTS, 60 replications")
print("=" * 72)
for pw in (True, False):
    rej, mx = 0, []
    for s in range(60):
        g = np.random.default_rng(1000 + s)
        rr = leadlag_ccf(ar1(200, .9, g), ar1(200, .9, g), max_lag=10,
                         prewhiten_series=pw, n_boot=300, seed=s)
        rej += rr.p_global < .05
        mx.append(abs(rr.best_stat))
    print(f"  {'pre-blanchie' if pw else 'brute       '} : "
          f"rejets a 5% = {rej}/60 ({rej/60:.0%}) | |rho|max moyen = {np.mean(mx):.2f}")
print("  -> nominal 5%. La CCF brute rejette 3x trop souvent : correlation fallacieuse.")

# --- 3. Probit sur cible binaire ---------------------------------------------
print("\n" + "=" * 72)
print("3. PROBIT DECALE — la vraie avance est +4")
print("=" * 72)
g = np.random.default_rng(7)
n = 280
xb = ar1(n, .8, g)
lat = np.roll(xb, 4)
ev = (g.random(n) < 1 / (1 + np.exp(-(-1.0 - 1.4 * lat)))).astype(int)
rp = leadlag_probit(xb, ev, max_lag=10, n_boot=300)
print(rp.summary())

# --- 4. Balayage multi-variables ---------------------------------------------
print("\n" + "=" * 72)
print("4. BALAYAGE — 4 candidats, dont 3 sans aucun lien avec la cible")
print("=" * 72)
g = np.random.default_rng(11)
n = 250
sig = ar1(n, .8, g)
cible = np.roll(sig, 6) + g.normal(0, 1.5, n)
cand = {"vrai_signal_av6": sig,
        "bruit_persistant_a": ar1(n, .9, g),
        "bruit_persistant_b": ar1(n, .9, g),
        "bruit_persistant_c": ar1(n, .9, g)}
print(scan(cand, cible, max_lag=12, n_boot=400).to_string(index=False))
print("\n-> comparer p_globale et p_bonferroni : avec 4 candidats le seuil se durcit.")
