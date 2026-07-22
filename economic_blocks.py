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

def build_production_block(dict_of_df):

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
                "US Durable Goods New Orders To (Economic Dynamic)",
                dict_of_df
            ),
            get_series(
                "ISM Manufacturing PMI SA (Business Conditions)",
                dict_of_df
            )
        ],
        axis=1
    ).dropna()

def stationary_block(dict_of_df):

    return pd.concat(
        [
            get_series(
                "GDP US Chained Dollars YoY SA (GDP)",
                dict_of_df
            ),
            get_series(
                "US Employees on Nonfarm Payrol (Employment)",
                dict_of_df
            ),
            get_series(
                "ISM Manufacturing PMI SA (Business Conditions)",
                dict_of_df
            ),
            get_series(
                "Philadelphia Fed Business Outl (Business Conditions)",
                dict_of_df
            ),
            get_series(
                "Market News International Chic (Business Conditions)",
                dict_of_df
            ),
            get_series(
                "ADP National Employment Report (Employment)",
                dict_of_df
            ),
            get_series(
                "US Industrial Production YOY S (Economic Dynamic)",
                dict_of_df
            ),
            get_series(
                "Chicago Fed National Activity (Business Conditions)",
                dict_of_df
            ),
            get_series(
                "Conference Board US Leading In (Business Conditions)",
                dict_of_df
            ),
            get_series(
                "US Durable Goods New Orders To (Economic Dynamic)",
                dict_of_df
            ),
            get_series(
                "Adjusted Retail Sales Less Aut MoM (Details selling)",
                dict_of_df
            ),
            get_series(
                "ISM Services PMI (Business Conditions)",
                dict_of_df
            ),
            get_series(
                "Richmond Manufacturing Survey (Business Conditions)",
                dict_of_df
            ),
            get_series(
                "Adjusted Retail Sales Less Aut YoY (Details selling)",
                dict_of_df
            ),
            get_series(
                "US Personal Consumption Expend % (Household)",
                dict_of_df
            ),
            get_series(
                "US Personal Consumption Expend (Household)",
                dict_of_df
            ),
        ],
        axis=1
    ).dropna()


def I_1_block(dict_of_df):

    return pd.concat(
        [
            get_series(
                "US Avg Hourly Earnings Private (Inflation)",
                dict_of_df
            ),
            get_series(
                "US CPI Urban Consumers Less Fo (Inflation)",
                dict_of_df
            ),
            get_series(
                "US Personal Consumption Expend (Inflation)",
                dict_of_df
            ),
            get_series(
                "Private Housing Authorized by (Housing)",
                dict_of_df
            ),
            get_series(
                "US Composite PMI SA (Business Conditions)",
                dict_of_df
            ),
            get_series(
                "US Manufacturing PMI SA (Business Conditions)",
                dict_of_df
            ),
            get_series(
                "US Services PMI Business Activ (Business Conditions)",
                dict_of_df
            ),
            get_series(
                "US NAR Total Existing Homes Sa (Housing)",
                dict_of_df
            ),
            get_series(
                "Conference Board Consumer Conf (Customer Trust)",
                dict_of_df
            ),
            get_series(
                "University of Michigan Consume (Customer Trust)",
                dict_of_df
            ),
        ],
        axis=1
    ).dropna()

def indice_block(dict_of_df):
    return pd.concat([
            get_series(
                "S&P 500 (Stock Index)",
                dict_of_df
            ),
            get_series(
                "MSCI world (Stock Index)",
                dict_of_df
            ),
            get_series(
                "NASDAQ (Stock Index)",
                dict_of_df
            ),
            get_series(
                "Gold Spot   $/Oz (Materials)",
                dict_of_df
            ),
            get_series(
                "Generic 1st 'CL' Future (Materials)",
                dict_of_df
            ),
            get_series(
                "Iron Ore Spot Price Index 62% (Materials)",
                dict_of_df
            ),
            get_series(
                "LME COPPER    3MO ($) (Materials)",
                dict_of_df
            ),
            get_series(
                "RUSSELL 2000 INDEX (Stock Index)",
                dict_of_df
            ),
            get_series(
                "BBG BTC Index (Crypto)",
                dict_of_df
            ),
            get_series(
                "BBG ETH Index (Crypto)",
                dict_of_df
            ),
            get_series(
                "Financial (Sub S&P500 Indicator)",
                dict_of_df
            ),
            get_series(
                "Information Technologie (Sub S&P500 Indicator)",
                dict_of_df
            ),
            get_series(
                "Santé (Sub S&P500 Indicator)",
                dict_of_df
            ),

        ],
        axis=1
    ).dropna()


