import pandas as pd

# Conversion des colonnes de dates
def convert_date_columns(df):
    for i in range(0, len(df.columns), 2):
        column = df.columns[i]
        df[column] = pd.to_datetime(df[column], dayfirst=True)

    return df


def count_dfs(df):
    return df.iloc[1, ::2].nunique()


def create_df(df):
    chosen_column = df.columns[0]

    new_df = pd.DataFrame()
    new_df[chosen_column] = df[chosen_column]

    cols_to_drop = []

    for i in range(len(df.columns)):
        if df[df.columns[i]].equals(df[chosen_column]):

            if i + 1 < len(df.columns):
                next_column = df.columns[i + 1]

                new_df[next_column] = df[next_column]

                cols_to_drop.extend(
                    [df.columns[i], next_column]
                )

    df.drop(columns=cols_to_drop, inplace=True)

    new_df.set_index(chosen_column, inplace=True)

    return new_df


def generate_dataframes(df):

    num_dataframes = count_dfs(df)

    for i in range(1, num_dataframes + 3):

        new_df = create_df(df)

        new_df = new_df.dropna()

        globals()[f"df{i}"] = new_df

        if new_df.empty:
            break


# ------------------------
# EXECUTION
# ------------------------

df = convert_date_columns(df)

generate_dataframes(df)

dict_of_df = {}

i = 1

while True:

    df_name = f"df{i}"

    if df_name in globals():

        if isinstance(globals()[df_name], pd.DataFrame):
            dict_of_df[df_name] = globals()[df_name]

        i += 1

    else:
        break


def convert_to_weeks():

    global dict_of_df

    for key, dataframe in dict_of_df.items():

        weekly_df = dataframe.resample("W").mean()

        weekly_df = (
            weekly_df
            .interpolate(method="linear")
            .round(2)
        )

        dict_of_df[key] = weekly_df

        globals()[key] = weekly_df


convert_to_weeks()

print(f"{len(dict_of_df)} dataframes créés")

mapping = {}

for i in range(0, len(df_original.columns) - 1, 2):

    date_col = df_original.columns[i]
    value_col = df_original.columns[i + 1]

    if str(date_col).startswith("Dates for "):
        mapping[f"Unnamed: {i+1}"] = value_col

print(mapping)
for name in dict_of_df:

    dict_of_df[name].rename(
        columns=mapping,
        inplace=True
    )

    globals()[name] = dict_of_df[name]
