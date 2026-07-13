 def export_cleaned_data():

    with pd.ExcelWriter(
        "cleaned_macro_data.xlsx"
    ) as writer:

        for name, dataframe in dict_of_df.items():

            dataframe.to_excel(
                writer,
                sheet_name=name
            )

    return "cleaned_macro_data.xlsx"


export_cleaned_data()
