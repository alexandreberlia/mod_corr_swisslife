dict_of_df = {}
i=1
while True:
    var_name = f'df{i}'
    if var_name in globals():
        if isinstance(globals()[var_name], pd.DataFrame):
                dict_of_df[var_name] = globals()[var_name]
        i += 1
    else:
        break
def return_dict_of_df(save_excel='no', excel_name='Dataframes', cell_width=50):
    if save_excel == "yes":
        dict_df = pd.DataFrame(dict_of_df)
        dict_df.to_excel(f"{excel_name}.xlsx")
        wb = load_workbook(f"{excel_name}.xlsx")
        adjust_dimensions(wb, max_column_width=cell_width)
        wb.save(f"{excel_name}.xlsx")
    return dict_of_df
print(len(dict_of_df))
