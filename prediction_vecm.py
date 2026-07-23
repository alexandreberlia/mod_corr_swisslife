
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def simuler_trajectoire_autour_prevision(
           data,
           vecm_model,
           n,
           frequence=object,
           methode="bootstrap",
           random_state=None                                                      
          ):
    
    forecast_values,lower_values,upper_values=vecm_model.predict(steps=n,
                                                                alpha=0.05)
    last_date=data.index[-1]
    future_index=pd.date_range(start=last_date,
                               periods=n,
                               freq=frequence)
    prevision_moyenne=pd.DataFrame(
      forecast_values,
      index=future_index,
      columns=data.columns
    )
    lower=pd.DataFrame(
      lower_values,
      index=future_index,
      columns=data.columns
    )
    upper=pd.DataFrame(
      upper_values,
      index=future_index,
      columns=data.columns
    )
    rng = np.random.default_rng(random_state)
  
    nombre_pas = len(prevision_moyenne)
    nombre_variables = len(prevision_moyenne.columns)
  
    residus = np.asarray(
        vecm_model.resid,
        dtype=float
    )
  
    # Centrage des résidus historiques
    residus_centres = (
        residus
        - residus.mean(axis=0, keepdims=True)
    )
  
    if methode == "bootstrap":
  
        indices = rng.integers(
            low=0,
            high=len(residus_centres),
            size=nombre_pas
        )
  
        chocs_futurs = residus_centres[indices]
  
    elif methode == "normale":
  
        covariance = np.cov(
            residus_centres,
            rowvar=False
        )
  
        chocs_futurs = rng.multivariate_normal(
            mean=np.zeros(nombre_variables),
            cov=covariance,
            size=nombre_pas
        )
  
    else:
        raise ValueError(
            "La méthode doit être 'bootstrap' ou 'normale'."
        )
  
    # Phi[0] = identité, Phi[1], Phi[2], etc.
    matrices_ma = vecm_model.ma_rep(
        maxn=nombre_pas
    )
  
    ecarts_propages = np.zeros(
        (nombre_pas, nombre_variables)
    )
  
    for horizon in range(nombre_pas):
  
        for retard_choc in range(horizon + 1):
  
            ecarts_propages[horizon] += (
                matrices_ma[retard_choc]
                @ chocs_futurs[horizon - retard_choc]
            )
  
    trajectoire = (
        prevision_moyenne.to_numpy(dtype=float)
        + ecarts_propages
    )
  
    trajectoire = pd.DataFrame(
        trajectoire,
        index=prevision_moyenne.index,
        columns=prevision_moyenne.columns
    )
  
    return trajectoire, chocs_futurs, ecarts_propages, prevision_moyenne

def tracer_trajectoire_simulee(
    data,
    prevision_moyenne,
    trajectoire_simulee,
    nombre_points_historiques=None,
    lower=lower,
    upper=upper
):
    """
    Compare la prévision moyenne du VECM avec une trajectoire
    affectée par des chocs résiduels.
    """
    nombre_variables = len(data.columns)

    fig, axes = plt.subplots(
        nrows=nombre_variables,
        ncols=1,
        figsize=(12, 4 * nombre_variables),
        sharex=True
    )

    if nombre_variables == 1:
        axes = [axes]

    for axe, variable in zip(
        axes,
        data.columns
    ):

        axe.plot(
            data.index[-40:],
            data[variable].iloc[-40:],
            color="black",
            label="Historique"
        )

        axe.plot(
            prevision_moyenne.index,
            prevision_moyenne[variable],
            color="red",
            linestyle="--",
            label="Prévision moyenne"
        )
        axe.fill_between(
        prevision_moyenne.index.to_pydatetime(),
        lower[variable],
        upper[variable],
        color="green",
        alpha=0.25,
        label="Intervalle à 95 %",
    )

        axe.plot(
            trajectoire_simulee.index,
            trajectoire_simulee[variable],
            color="blue",
            linestyle="-.",
            marker="o",
            label="Trajectoire avec résidus"
        )
        
        axe.axvline(
            data.index[-1],
            color="grey",
            linestyle=":",
            label="Début de la prévision"
        )

        axe.set_title(
            f"Prévision VECM avec chocs : {variable}"
        )

        axe.set_ylabel(variable)
        axe.grid(alpha=0.3)
        axe.legend()
        import numpy as np

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

