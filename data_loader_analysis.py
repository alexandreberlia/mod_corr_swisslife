import pandas as pd

excel_file = pd.ExcelFile("cleaned_macro_data.xlsx")

dict_of_df = {}

for sheet in excel_file.sheet_names:

    temp_df = pd.read_excel(
        excel_file,
        sheet_name=sheet,
        index_col=0
    )

    temp_df.index = pd.to_datetime(
        temp_df.index
    )

    dict_of_df[sheet] = temp_df

for i, sheet in enumerate(dict_of_df.keys(), start=1):

    globals()[f"df{i}"] = dict_of_df[sheet]
