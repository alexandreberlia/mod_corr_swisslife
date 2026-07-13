from config import *
from data_loader import *
from data_cleaning import *
from grouping import *
from data_dictionary import *
from normalization import *
from export import *

df = load_and_preview_csv("Data_in_value.csv")

df = rename_columns(df)

df = correct_and_save_columns(
    df,
    "corrected_file"
)

df = convert_date_columns(df)

generate_dataframes(df)

dict_of_df = return_dict_of_df()

convert_to_weeks()

export_cleaned_data()
