 {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [],
   "source": [
    "list_of_needed_dataframes = ['(GDP)', '(Employment)', '(Inflation)', '(Economic Dynamic)', '(Business Conditions)', '(Housing)', '(Details selling)', '(Household)', '(Customer Trust)']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {},
   "outputs": [],
   "source": [
    "def return_dataframes_with_specific_name______________(name):\n",
    "    return {df_name:df for df_name, df in dict_of_df.items() if any(name in col for col in df.columns)}\n",
    "\n",
    "#return_dataframes_with_specific_name(name='GDP')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {},
   "outputs": [],
   "source": [
    "def return_columns_with_specific_names______________(name, df):\n",
    "    return [col for col in df.columns if name in col]\n",
    "\n",
    "#return_columns_with_specific_names('GDP', df1)"
   ]
  }

{
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create new subdataframe for each of our needs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_sub_dataframe_further_analysis______________():\n",
    "    global GDP_df, Employment_df, Inflation_df, Economic_Dynamic_df, Business_Conditions_df, Housing_df, Details_Selling_df, Household_df, Customer_Trust_df\n",
    "\n",
    "    list_of_needed_dataframes = ['(GDP)', '(Employment)', '(Inflation)', '(Economic Dynamic)', '(Business Conditions)', '(Housing)', '(Details selling)', '(Household)', '(Customer Trust)']\n",
    "    \n",
    "    GDP_data = {}\n",
    "    Employment_data = {}\n",
    "    Inflation_data = {}\n",
    "    Economic_Dynamic_data = {}\n",
    "    Business_Conditions_data = {}\n",
    "    Housing_data = {}\n",
    "    Details_Selling_data = {}\n",
    "    Household_data = {}\n",
    "    Customer_Trust_data = {}\n",
    "\n",
    "    # Iterate through the global variables\n",
    "    for i in range(1, count_dfs(df) + 2):\n",
    "        df_name = f'df{i}'\n",
    "        if df_name in globals():\n",
    "            df = globals()[df_name]\n",
    "            date_col = df.index.name if df.index.name else f'{df_name}_date'\n",
    "            df = df.reset_index().rename(columns={'index': date_col})\n",
    "            for cols in df.columns:\n",
    "                if cols == date_col:\n",
    "                    continue\n",
    "                for name in list_of_needed_dataframes:\n",
    "                    if name in cols:\n",
    "                        if name == '(GDP)':\n",
    "                            GDP_data[f'Dates for {df_name}: {cols}'] = df[date_col]\n",
    "                            GDP_data[f'{cols}'] = df[cols]\n",
    "                        elif name == '(Employment)':\n",
    "                            Employment_data[f'Dates for {df_name}: {cols}'] = df[date_col]\n",
    "                            Employment_data[f'{cols}'] = df[cols]\n",
    "                        elif name == '(Inflation)':\n",
    "                            Inflation_data[f'Dates for {df_name}: {cols}'] = df[date_col]\n",
    "                            Inflation_data[f'{cols}'] = df[cols]\n",
    "                        elif name == '(Economic Dynamic)':\n",
    "                            Economic_Dynamic_data[f'Dates for {df_name}: {cols}'] = df[date_col]\n",
    "                            Economic_Dynamic_data[f'{cols}'] = df[cols]\n",
    "                        elif name == '(Business Conditions)':\n",
    "                            Business_Conditions_data[f'Dates for {df_name}: {cols}'] = df[date_col]\n",
    "                            Business_Conditions_data[f'{cols}'] = df[cols]\n",
    "                        elif name == '(Housing)':\n",
    "                            Housing_data[f'Dates for {df_name}: {cols}'] = df[date_col]\n",
    "                            Housing_data[f'{cols}'] = df[cols]\n",
    "                        elif name == '(Details selling)':\n",
    "                            Details_Selling_data[f'Dates for {df_name}: {cols}'] = df[date_col]\n",
    "                            Details_Selling_data[f'{cols}'] = df[cols]\n",
    "                        elif name == '(Household)':\n",
    "                            Household_data[f'Dates for {df_name}: {cols}'] = df[date_col]\n",
    "                            Household_data[f'{cols}'] = df[cols]\n",
    "                        elif name == '(Customer Trust)':\n",
    "                            Customer_Trust_data[f'Dates for {df_name}: {cols}'] = df[date_col]\n",
    "                            Customer_Trust_data[f'{cols}'] = df[cols]\n",
    "\n",
    "    # Convert dictionaries to dataframes\n",
    "    GDP_df = pd.DataFrame(GDP_data)\n",
    "    Employment_df = pd.DataFrame(Employment_data)\n",
    "    Inflation_df = pd.DataFrame(Inflation_data)\n",
    "    Economic_Dynamic_df = pd.DataFrame(Economic_Dynamic_data)\n",
    "    Business_Conditions_df = pd.DataFrame(Business_Conditions_data)\n",
    "    Housing_df = pd.DataFrame(Housing_data)\n",
    "    Details_Selling_df = pd.DataFrame(Details_Selling_data)\n",
    "    Household_df = pd.DataFrame(Household_data)\n",
    "    Customer_Trust_df = pd.DataFrame(Customer_Trust_data)\n",
    "\n",
    "#create_sub_dataframe_further_analysis()"
   ]
  }
 {
   "cell_type": "code",
   "execution_count": 159,
   "metadata": {},
   "outputs": [],
   "source": [
    "def put_pairs_in_list______________(dfx, dfy):\n",
    "    indexes_list = []\n",
    "    for i in range(0, len(dfx.columns), 2):\n",
    "        for j in range(0, len(dfy.columns), 2):\n",
    "            if \"Dates for\" in dfx.columns[i] and \"Dates for\" in dfy.columns[j]:\n",
    "                pairs = [dfx.columns[i], dfx.columns[i+1], dfy.columns[j], dfy.columns[j+1]]\n",
    "                indexes_list.append(pairs)\n",
    "    return indexes_list\n",
    "\n",
    "#put_pairs_in_list(GDP_df, Employment_df)"
   ]
  }

{
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Create a sub dataframe for every variable in the pair"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_pair_dataframes______________(pair_list):\n",
    "    pair_dfs = []\n",
    "    for pair in pair_list:\n",
    "        dates1_col, var1_col, dates2_col, var2_col = pair\n",
    "        \n",
    "        # Retrieve dataframes and columns from globals\n",
    "        df1 = next((globals()[df] for df in globals() if isinstance(globals()[df], pd.DataFrame) and dates1_col in globals()[df].columns), None)\n",
    "        df2 = next((globals()[df] for df in globals() if isinstance(globals()[df], pd.DataFrame) and dates2_col in globals()[df].columns), None)\n",
    "        \n",
    "        if df1 is not None and df2 is not None:\n",
    "            # Create new dataframes for the pairs\n",
    "            df_pair1 = df1[[dates1_col, var1_col]].set_index(dates1_col)\n",
    "            df_pair2 = df2[[dates2_col, var2_col]].set_index(dates2_col)\n",
    "            \n",
    "            pair_dfs.append([df_pair1, df_pair2])\n",
    "        \n",
    "    return pair_dfs\n",
    "\n",
    "#create_pair_dataframes(put_pairs_in_list(GDP_df, Employment_df))"
   ]
  }
