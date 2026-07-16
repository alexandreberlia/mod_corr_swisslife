import pandas as pd
import matplotlib.pyplot as plt


def compute_fevd(
        var_results,
        periods=24):

    return var_results.fevd(periods)


def plot_fevd_stacked(
        var_results,
        variable,
        periods=24):

    import matplotlib.pyplot as plt
    import pandas as pd

    fevd = var_results.fevd(periods)

    variables = var_results.names

    variable_idx = variables.index(variable)

    df = pd.DataFrame(
        fevd.decomp[variable_idx],
        columns=variables
    )

    df.index = range(1, periods + 1)

    ax = df.plot.area(
        figsize=(12, 6),
        alpha=0.8
    )

    ax.set_title(
        f"FEVD - {variable}"
    )

    ax.set_xlabel("Horizon")

    ax.set_ylabel("Contribution")

    plt.legend(
        title="Shock",
        bbox_to_anchor=(1.05, 1)
    )

    plt.tight_layout()
    plt.show()

def fevd_dataframe(
        var_results,
        periods=24):

    fevd = var_results.fevd(periods)

    variables = var_results.names

    results = []

    for variable_idx, variable in enumerate(variables):

        for horizon in range(periods):

            for shock_idx, shock in enumerate(variables):

                results.append({

                    "Variable": variable,

                    "Horizon": horizon + 1,

                    "Shock": shock,

                    "Contribution (%)":

                        fevd.decomp[
                            variable_idx,
                            horizon,
                            shock_idx
                        ] * 100
                })

    return pd.DataFrame(results)


def fevd_horizon(
        var_results,
        horizon=12):
    fevd = var_results.fevd(horizon)

    variables = var_results.names

    table = pd.DataFrame(
        fevd.decomp[:,horizon-1,:],
        index=variables,
        columns=variables
    )

    return table * 100


def export_fevd(
        var_results,
        periods=24,
        excel_name="FEVD_Results"):

    df = fevd_dataframe(
        var_results,
        periods
    )

    df.to_excel(
        f"{excel_name}.xlsx",
        index=False
    )

    return df
