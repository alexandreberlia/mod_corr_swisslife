import pandas as pd
import matplotlib.pyplot as plt


def compute_irf_var(
        var_results,
        periods=24):
    return var_results.irf(periods)

def plot_irf_var(
        var_results,
        periods=24):
    """
    Plot all impulse response functions.
    """

    irf = var_results.irf(periods)

    irf.plot(
        orth=False
    )

    plt.show()

    return irf


def plot_single_irf_var(
        var_results,
        impulse,
        response,
        periods=24):
    """
    Plot a single impulse response.
    """

    irf = var_results.irf(periods)

    irf.plot(
        impulse=impulse,
        response=response
    )

    plt.show()

    return irf


def irf_dataframe_var(
        var_results,
        periods=24):
    """
    Export IRF values as DataFrame.
    """

    irf = var_results.irf(periods)

    variables = var_results.names

    results = []

    for impulse_idx, impulse in enumerate(variables):

        for response_idx, response in enumerate(variables):

            for horizon in range(periods + 1):

                results.append({

                    "Impulse": impulse,

                    "Response": response,

                    "Horizon": horizon,

                    "Value":
                        irf.irfs[
                            horizon,
                            response_idx,
                            impulse_idx
                        ]
                })

    return pd.DataFrame(results)


def export_irf_var(
        var_results,
        periods=24,
        excel_name="IRF_Results"):

    df = irf_dataframe(
        var_results,
        periods
    )

    df.to_excel(
        f"{excel_name}.xlsx",
        index=False
    )

    return df
