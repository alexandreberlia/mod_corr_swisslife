
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [],
   "source": [
    "def lin_regression_sklearn______________(std_limit, variable_to_apply_regression_on):\n",
    "    original_df = find_dates_above_below_stds(std_limit=std_limit)\n",
    "    linear_regression_df = filter_the_original_df(original_df)\n",
    "    Y = linear_regression_df[variable_to_apply_regression_on] \n",
    "    variables_list = []\n",
    "    for i in range(len(linear_regression_df.columns)):\n",
    "        if variable_to_apply_regression_on == linear_regression_df.columns[i]:\n",
    "            continue\n",
    "        else:\n",
    "            variables_list.append(linear_regression_df.columns[i])\n",
    "    X = linear_regression_df[variables_list]  \n",
    "    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)\n",
    "    model = LinearRegression()\n",
    "    model.fit(X_train, Y_train)\n",
    "    Y_pred = model.predict(X_test)\n",
    "    mse = mean_squared_error(Y_test, Y_pred)\n",
    "    r2 = r2_score(Y_test, Y_pred)\n",
    "    print(f\"Mean Squared Error: {mse}\")\n",
    "    print(f\"R^2 Score: {r2}\")\n",
    "    print(\"Coefficients:\", model.coef_)\n",
    "    print(\"Intercept:\", model.intercept_)\n",
    "\n",
    "#lin_regression_sklearn(2, variable_to_apply_regression_on='Adjusted Retail Sales Less Aut MoM (Details selling)')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Find the dataframes which match our needs"
   ]
  },
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
  },
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
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "metadata": {},
   "outputs": [],
   "source": [
    "#list_of_new_dfs = ['GDP_df', 'Employment_df', 'Inflation_df', 'Economic_Dynamic_df', 'Business_Conditions_df', 'Housing_df', 'Details_Selling_df', 'Household_df', 'Customer_Trust_df']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### GDP Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We remember that:\n",
    "\n",
    "* GDP > 0 : Expansion \n",
    "\n",
    "* GDP < 0 : recession\n",
    "\n",
    "* Pivot : 0\n",
    "\n",
    "* Dynamique --> 3-4 derniers trimestres, est ce haussier ou baissier ?\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Find periods of expansion and recession"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Function to find these periods"
   ]
  },
 
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Graph them"
   ]
  },
  
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Find the dynamic of the last 3-4 trimesters"
   ]
  },
 
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### We are currently in a period of slow economic growth, where growth is slowing down. However, we are yet to turn in a period of recession"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Employment Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We remember that:\n",
    "\n",
    "* There is no pivot point for unemployment\n",
    "    * We define the mins and maxs according to the recession and high expansion periods (flotting GDP pivot)\n",
    "* Find the dynamic of the trimesters after the pivot point\n",
    "    * What is the lapse of time such that the pivot of unemployment changes ?\n",
    "    * What is the historical average to change this tendency ?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Finding the relation between unemployment and GDP"
   ]
  },
  

  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [],
   "source": [
    "l1 = [\n",
    "    \"Chicago Fed National Activity (Business Conditions)\",\n",
    "    \"Conference Board US Leading In (Business Conditions)\",\n",
    "    \"ISM Manufacturing PMI SA (Business Conditions)\",\n",
    "    \"ISM Services PMI (Business Conditions)\",\n",
    "    \"Market News International Chic (Business Conditions)\",\n",
    "    \"Philadelphia Fed Business Outl (Business Conditions)\",\n",
    "    \"Richmond Manufacturing Survey (Business Conditions)\",\n",
    "    \"US Composite PMI SA (Business Conditions)\",\n",
    "    \"US Empire State Manufacturing (Business Conditions)\",\n",
    "    \"US Manufacturing PMI SA (Business Conditions)\",\n",
    "    \"US Services PMI Business Activ (Business Conditions)\",\n",
    "    \"Conference Board Consumer Conf (Customer Trust)\",\n",
    "    \"University of Michigan Consume (Customer Trust)\",\n",
    "    \"Adjusted Retail Sales Less Aut MoM (Details selling)\",\n",
    "    \"Adjusted Retail Sales Less Aut YoY (Details selling)\",\n",
    "    \"US Auto Sales Total Annualized (Details selling)\",\n",
    "    \"US Capacity Utilization % of T (Economic Dynamic)\",\n",
    "    \"US Durable Goods New Orders To (Economic Dynamic)\",\n",
    "    \"US Industrial Production YOY S (Economic Dynamic)\",\n",
    "    \"ADP National Employment Report (Employment)\",\n",
    "    \"Conference Board US Leading In (Employment)\",\n",
    "    \"U-3 US Unemployment Rate Total (Employment)\",\n",
    "    \"US Employees on Nonfarm Payrol (Employment)\",\n",
    "    \"GDP US Chained Dollars QoQ SAA (GDP)\",\n",
    "    \"GDP US Chained Dollars YoY SA (GDP)\",\n",
    "    \"US GDP Nominal Dollars YoY SA (GDP)\",\n",
    "    \"US GDP Price Index QoQ SAAR (GDP)\",\n",
    "    \"US Personal Consumption Expend % (Household)\",\n",
    "    \"US Personal Consumption Expend (Household)\",\n",
    "    \"US Personal Income YoY SA (Household)\",\n",
    "    \"Private Housing Authorized by (Housing)\",\n",
    "    \"US NAR Total Existing Homes Sa (Housing)\",\n",
    "    \"US Avg Hourly Earnings Private (Inflation)\",\n",
    "    \"US CPI Urban Consumers Less Fo (Inflation)\",\n",
    "    \"US PPI Finished Goods Less Foo (Inflation)\",\n",
    "    \"US Personal Consumption Expend (Inflation)\",\n",
    "    \"Generic 1st 'CL' Future (Materials)\",\n",
    "    \"Gold Spot   $/Oz (Materials)\",\n",
    "    \"Iron Ore Spot Price Index 62% (Materials)\",\n",
    "    \"LME COPPER    3MO ($) (Materials)\"\n",
    "]"
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
  },
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
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "metadata": {},
   "outputs": [],
   "source": [
    "def normalize_columns_new______________(df1, df2):\n",
    "    scaler = MinMaxScaler()\n",
    "\n",
    "    # Ensure indices are unique\n",
    "    df1 = df1.reset_index().drop_duplicates().set_index(df1.index.name)\n",
    "    df2 = df2.reset_index().drop_duplicates().set_index(df2.index.name)\n",
    "\n",
    "    combined = pd.concat([df1, df2], axis=1)\n",
    "    normalized_values = scaler.fit_transform(combined)\n",
    "    normalized_df = pd.DataFrame(normalized_values, columns=combined.columns, index=combined.index)\n",
    "    \n",
    "    return normalized_df[df1.columns], normalized_df[df2.columns]"
   ]
  },
 
