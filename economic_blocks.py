import pandas as pd


def get_series(variable_name, dict_of_df):
    """
    Retrieve a time series from dict_of_df.
    """

    for df in dict_of_df.values():

        if variable_name in df.columns:
            return df[[variable_name]]

    raise KeyError(
        f"{variable_name} not found in dict_of_df."
    )
    
def build_macro_core_block(dict_of_df):

    return pd.concat(
        [
            get_series(
                "GDP US Chained Dollars YoY SA (GDP)",
                dict_of_df
            ),
            get_series(
                "US CPI Urban Consumers Less Fo (Inflation)",
                dict_of_df
            ),
            get_series(
                "U-3 US Unemployment Rate Total (Employment)",
                dict_of_df
            ),
            get_series(
                "US Industrial Production YOY S (Economic Dynamic)",
                dict_of_df
            ),
        ],
        axis=1
    ).dropna()

def build_macro_policy_block(dict_of_df):

    return pd.concat(
        [
            get_series(
                "GDP US Chained Dollars YoY SA (GDP)",
                dict_of_df
            ),
            get_series(
                "US CPI Urban Consumers Less Fo (Inflation)",
                dict_of_df
            ),
            get_series(
                "U-3 US Unemployment Rate Total (Employment)",
                dict_of_df
            ),
            get_series(
                "FED FUNDS (FED)",
                dict_of_df
            ),
        ],
        axis=1
    ).dropna()

def build_growth_block(dict_of_df):

    return pd.concat(
        [
            get_series(
                "GDP US Chained Dollars YoY SA (GDP)",
                dict_of_df
            ),
            get_series(
                "US Industrial Production YOY S (Economic Dynamic)",
                dict_of_df
            ),
            get_series(
                "Adjusted Retail Sales Less Aut YoY (Details selling)",
                dict_of_df
            ),
            get_series(
                "US Personal Income YoY SA (Household)",
                dict_of_df
            ),
        ],
        axis=1
    ).dropna()

def build_inflation_block(dict_of_df):

    return pd.concat(
        [
            get_series(
                "US CPI Urban Consumers Less Fo (Inflation)",
                dict_of_df
            ),
            get_series(
                "US PPI Finished Goods Less Foo (Inflation)",
                dict_of_df
            ),
            get_series(
                "US Personal Consumption Expend (Inflation)",
                dict_of_df
            ),
            get_series(
                "US Avg Hourly Earnings Private (Inflation)",
                dict_of_df
            ),
        ],
        axis=1
    ).dropna()

def build_employment_block(dict_of_df):

    return pd.concat(
        [
            get_series(
                "U-3 US Unemployment Rate Total (Employment)",
                dict_of_df
            ),
            get_series(
                "US Employees on Nonfarm Payrol (Employment)",
                dict_of_df
            ),
            get_series(
                "ADP National Employment Report (Employment)",
                dict_of_df
            ),
        ],
        axis=1
    ).dropna()

import pandas as pd


def build_consumer_block(dict_of_df):
    """
    Consumer Block

    Consumer Confidence
    Michigan Sentiment
    Retail Sales
    Personal Consumption
    """

    return pd.concat(
        [
            get_series(
                "Conference Board Consumer Conf (Customer Trust)",
                dict_of_df
            ),
            get_series(
                "University of Michigan Consume (Customer Trust)",
                dict_of_df
            ),
            get_series(
                "Adjusted Retail Sales Less Aut YoY (Details selling)",
                dict_of_df
            ),
            get_series(
                "US Personal Consumption Expend % (Household)",
                dict_of_df
            )
        ],
        axis=1
    ).dropna()
