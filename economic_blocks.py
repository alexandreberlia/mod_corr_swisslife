import pandas as pd


def build_growth_block(dict_of_df):
    """
    Growth / Economic Activity Block

    GDP
    Industrial Production
    Retail Sales
    Personal Income
    """

    return pd.concat(
        [
            dict_of_df["df1"][
                ["GDP US Chained Dollars YoY SA (GDP)"]
            ],
            dict_of_df["df5"][
                ["US Industrial Production YOY S (Economic Dynamic)"]
            ],
            dict_of_df["df8"][
                ["Adjusted Retail Sales Less Aut YoY (Details selling)"]
            ],
            dict_of_df["df9"][
                ["US Personal Income YoY SA (Household)"]
            ],
        ],
        axis=1,
    ).dropna()

import pandas as pd


def build_inflation_block(dict_of_df):
    """
    Inflation Block

    CPI
    PPI
    PCE
    Average Hourly Earnings
    """

    return pd.concat(
        [
            dict_of_df["df4"][
                ["US CPI Urban Consumers Less Fo (Inflation)"]
            ],
            dict_of_df["df4"][
                ["US PPI Finished Goods Less Foo (Inflation)"]
            ],
            dict_of_df["df4"][
                ["US Personal Consumption Expend (Inflation)"]
            ],
            dict_of_df["df4"][
                ["US Avg Hourly Earnings Private (Inflation)"]
            ],
        ],
        axis=1,
    ).dropna()

import pandas as pd


def build_employment_block(dict_of_df):
    """
    Labour Market Block

    Unemployment
    Payrolls
    ADP
    """

    return pd.concat(
        [
            dict_of_df["df2"][
                ["U-3 US Unemployment Rate Total (Employment)"]
            ],
            dict_of_df["df2"][
                ["US Employees on Nonfarm Payrol (Employment)"]
            ],
            dict_of_df["df2"][
                ["ADP National Employment Report (Employment)"]
            ],
        ],
        axis=1,
    ).dropna()

import pandas as pd


def build_leading_indicators_block(dict_of_df):
    """
    Leading Indicators Block

    Conference Board LEI
    Chicago Fed
    Composite PMI
    Manufacturing PMI
    Services PMI
    """

    return pd.concat(
        [
            dict_of_df["df6"][
                ["Conference Board US Leading In (Business Conditions)"]
            ],
            dict_of_df["df6"][
                ["Chicago Fed National Activity (Business Conditions)"]
            ],
            dict_of_df["df6"][
                ["US Composite PMI SA (Business Conditions)"]
            ],
            dict_of_df["df6"][
                ["ISM Manufacturing PMI SA (Business Conditions)"]
            ],
            dict_of_df["df6"][
                ["ISM Services PMI (Business Conditions)"]
            ],
        ],
        axis=1,
    ).dropna()

import pandas as pd


def build_housing_block(dict_of_df):
    """
    Housing Block

    Building Permits
    Existing Home Sales
    """

    return pd.concat(
        [
            dict_of_df["df7"][
                ["Private Housing Authorized by (Housing)"]
            ],
            dict_of_df["df7"][
                ["US NAR Total Existing Homes Sa (Housing)"]
            ],
        ],
        axis=1,
    ).dropna()

import pandas as pd


def build_consumer_block(dict_of_df):
    """
    Consumer Block

    Consumer Confidence
    Michigan Sentiment
    Retail Sales
    Consumption
    """

    return pd.concat(
        [
            dict_of_df["df10"][
                ["Conference Board Consumer Conf (Customer Trust)"]
            ],
            dict_of_df["df10"][
                ["University of Michigan Consume (Customer Trust)"]
            ],
            dict_of_df["df8"][
                ["Adjusted Retail Sales Less Aut YoY (Details selling)"]
            ],
            dict_of_df["df9"][
                ["US Personal Consumption Expend % (Household)"]
            ],
        ],
        axis=1,
    ).dropna()

import pandas as pd


def build_macro_core_block(dict_of_df):
    """
    Core Macroeconomic Block

    GDP
    Inflation
    Unemployment
    Industrial Production
    """

    return pd.concat(
        [
            dict_of_df["df1"][
                ["GDP US Chained Dollars YoY SA (GDP)"]
            ],
            dict_of_df["df4"][
                ["US CPI Urban Consumers Less Fo (Inflation)"]
            ],
            dict_of_df["df2"][
                ["U-3 US Unemployment Rate Total (Employment)"]
            ],
            dict_of_df["df5"][
                ["US Industrial Production YOY S (Economic Dynamic)"]
            ],
        ],
        axis=1,
    ).dropna()

import pandas as pd


def build_macro_policy_block(dict_of_df):
    """
    GDP
    Inflation
    Unemployment
    FED Funds
    """

    return pd.concat(
        [
            dict_of_df["df1"][
                ["GDP US Chained Dollars YoY SA (GDP)"]
            ],
            dict_of_df["df4"][
                ["US CPI Urban Consumers Less Fo (Inflation)"]
            ],
            dict_of_df["df2"][
                ["U-3 US Unemployment Rate Total (Employment)"]
            ],
            dict_of_df["df15"][
                ["FED FUNDS (FED)"]
            ],
        ],
        axis=1,
    ).dropna()
