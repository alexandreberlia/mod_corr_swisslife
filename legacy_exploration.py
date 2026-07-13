{
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {},
   "outputs": [],
   "source": [
    "def retain_highest_relevance______________(std_limit, variable_to_apply_regression_on, coef_need = None, std_error = None, t_value = None, p_value = None):\n",
    "    list_needed = []\n",
    "    df = lin_regression_stats_model(std_limit = std_limit, variable_to_apply_regression_on=variable_to_apply_regression_on)\n",
    "    for i in range(1, len(df.tables[1])):\n",
    "        if coef_need is not None and abs(float(df.tables[1][i][1].data)) >= coef_need:\n",
    "            list_needed.append(df.tables[1][i][0].data)\n",
    "        if std_error is not None and abs(float(df.tables[1][i][2].data)) >= std_error:\n",
    "            list_needed.append(df.tables[1][i][0].data)\n",
    "        if t_value is not None and abs(float(df.tables[1][i][3].data)) >= t_value:\n",
    "            list_needed.append(df.tables[1][i][0].data)\n",
    "        if p_value is not None and abs(float(df.tables[1][i][4].data)) >= p_value:\n",
    "            list_needed.append(df.tables[1][i][0].data)\n",
    "    list_needed = list(set(list_needed))        \n",
    "\n",
    "    return list_needed\n",
    "\n",
    "#len(retain_highest_relevance(2, 'Adjusted Retail Sales Less Aut MoM (Details selling)', coef_need=0.2))"
   ]
  },
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
   "cell_type": "code",
   "execution_count": 151,
   "metadata": {},
   "outputs": [],
   "source": [
    "def find_expansion_recession______________(df, name='GDP', threshold=0):\n",
    "    all_periods = []\n",
    "\n",
    "    columns_names = [col for col in df.columns if name in col and pd.api.types.is_numeric_dtype(df[col])]\n",
    "    for col in columns_names:\n",
    "        periods = []\n",
    "        above_threshold = df.iloc[0][col] > threshold\n",
    "        current_period = [df.index[0], None, above_threshold]\n",
    "\n",
    "        for index, value in df[col].items():\n",
    "            if value > threshold and not above_threshold:\n",
    "                current_period[1] = index\n",
    "                periods.append(current_period + [col])\n",
    "                current_period = [index, None, True]\n",
    "                above_threshold = True\n",
    "            elif value < threshold and above_threshold:\n",
    "                current_period[1] = index\n",
    "                periods.append(current_period + [col])\n",
    "                current_period = [index, None, False]\n",
    "                above_threshold = False\n",
    "\n",
    "        if current_period[1] is None:\n",
    "            current_period[1] = df.index[-1]\n",
    "            periods.append(current_period + [col])\n",
    "\n",
    "        for period in periods:\n",
    "            start_index, end_index, above_threshold, col_name = period\n",
    "            period_data = df.loc[start_index:end_index]\n",
    "            extreme_value = period_data[col_name].max() if above_threshold else period_data[col_name].min()\n",
    "            period.extend([period_data, extreme_value])\n",
    "\n",
    "        all_periods.append(periods)\n",
    "\n",
    "    return all_periods\n",
    "\n",
    "#x = find_expansion_recession(GDP_df)"
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
   "cell_type": "code",
   "execution_count": 152,
   "metadata": {},
   "outputs": [],
   "source": [
    "def graph_expansion_recession_past_dynamic______________(weeks, name, threshold=0):\n",
    "    df_dict = return_dataframes_with_specific_name(name=name)\n",
    "\n",
    "    for df_name, df in df_dict.items():\n",
    "        columns_names = return_columns_with_specific_names(name=name, df=df)\n",
    "        periods = find_expansion_recession(df, name=name, threshold=threshold)\n",
    "        \n",
    "        for column in columns_names:\n",
    "            a = 0\n",
    "            b = 0\n",
    "            plt.figure(figsize=(24, 12))\n",
    "            plt.plot(df.index, df[column], label=f'{df_name} - {column}')  # Plotting with DataFrame and column\n",
    "            \n",
    "            # Shade the periods above or below the threshold\n",
    "            for period_group in periods:\n",
    "                for period in period_group:\n",
    "                    if period[3] == column:\n",
    "                        start_index, end_index, above_threshold, col_name, period_data, extreme_value = period\n",
    "                        if above_threshold:\n",
    "                            a += 1\n",
    "                            plt.fill_between(df.loc[start_index:end_index].index, threshold, df.loc[start_index:end_index][column], color='green', alpha=0.3)\n",
    "                        else:\n",
    "                            b += 1\n",
    "                            plt.fill_between(df.loc[start_index:end_index].index, df.loc[start_index:end_index][column], threshold, color='red', alpha=0.3)\n",
    "\n",
    "            # Adding horizontal line at the threshold\n",
    "            plt.axhline(y=threshold, color='blue', linestyle='-', linewidth=2, label='Threshold')\n",
    "            plt.axhline(y=threshold, color='blue', linestyle='-', linewidth=0, label=f'{a} periods of expansion')\n",
    "            plt.axhline(y=threshold, color='blue', linestyle='-', linewidth=0, label=f'{b} periods of recession')\n",
    "\n",
    "            # Adding vertical line 'weeks' before the most recent date\n",
    "            if len(df.index) > 0:\n",
    "                most_recent_date = df.index[-1]\n",
    "                weeks_before = most_recent_date - pd.DateOffset(weeks=weeks)\n",
    "                plt.axvline(x=weeks_before, color='r', linestyle='--', linewidth=2, label=f'Last {weeks} weeks')\n",
    "\n",
    "            plt.xlabel('Time')\n",
    "            plt.ylabel('Value')\n",
    "            plt.title(f'{column} Over Time')\n",
    "            plt.legend()\n",
    "            plt.show()\n",
    "\n",
    "# Example usage:\n",
    "#graph_expansion_recession_past_dynamic(17*3, 'GDP', threshold=0)"
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
   "cell_type": "code",
   "execution_count": 153,
   "metadata": {},
   "outputs": [],
   "source": [
    "def past_trimesters_dynamic______________(weeks, name):\n",
    "    df_dict = return_dataframes_with_specific_name(name=name)\n",
    "    name_and_characteristics = []\n",
    "    \n",
    "    for df_name, df in df_dict.items():\n",
    "        columns_names = return_columns_with_specific_names(name=name, df=df)\n",
    "        \n",
    "        for column in columns_names:\n",
    "            # Extract the relevant rows based on the index condition\n",
    "            df_filtered = df[df.index >= df.index[-1] - pd.DateOffset(weeks=weeks)]\n",
    "            \n",
    "            if df_filtered.empty:\n",
    "                continue\n",
    "            \n",
    "            values = df_filtered[column]\n",
    "            \n",
    "            # Calculate required statistics\n",
    "            max_value = values.max()\n",
    "            min_value = values.min()\n",
    "            average_value = values.mean()\n",
    "            median_value = values.median()\n",
    "            slope, intercept, r_value, p_value, std_err = linregress(df_filtered.index.map(pd.Timestamp.toordinal), values)\n",
    "            dispersion = values.std()\n",
    "            \n",
    "            characteristics = {\n",
    "                'Dataframe': df_name,\n",
    "                'Column': column,\n",
    "                'Max': float(max_value),\n",
    "                'Min': float(min_value),\n",
    "                'Average': float(average_value),\n",
    "                'Median': float(median_value),\n",
    "                'Slope': float(slope),\n",
    "                'Standard deviation': float(dispersion)\n",
    "            }\n",
    "            \n",
    "            name_and_characteristics.append(characteristics)\n",
    "    \n",
    "    return name_and_characteristics\n",
    "\n",
    "#past_trimesters_dynamic(3*17, 'GDP')"
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
   "execution_count": 154,
   "metadata": {},
   "outputs": [],
   "source": [
    "def normalize_y_axis_threshold______________(svgal_window=75, polyorder=3, number=1, window=104, threshold_small=0.5, threshold_big=0.7, decimal=3):\n",
    "    l = list_of_global_all_corr_parametrized(number=number, window=window, threshold_small=threshold_small, threshold_big=threshold_big, decimal=decimal)\n",
    "    smoothed_df = smooth_dataframe(svgal_window=svgal_window, polyorder=polyorder, number=number, window=window, threshold_small=threshold_small, threshold_big=threshold_big, decimal=decimal)\n",
    "    results = []\n",
    "    for i in range(len(dict_of_df)):\n",
    "        for j in range(len(dict_of_df)):\n",
    "            dfx_name = f'df{i+1}'\n",
    "            dfy_name = f'df{j+1}'\n",
    "            if dfx_name in globals() and dfy_name in globals():\n",
    "                dfx, dfy = smoothed_df[i], smoothed_df[j]\n",
    "                for x in range(len(l)):\n",
    "                    for y in range(len(dfx.columns)):\n",
    "                        for z in range(len(dfy.columns)):\n",
    "                            if (l[x][0] == dfx.columns[y]) and (l[x][1] == dfy.columns[z]):\n",
    "                                normalized_df = normalize_columns(dfy, dfx, dfx.columns[y], dfy.columns[z])\n",
    "                                results.append(([dfx.columns[y], dfy.columns[z]], normalized_df))\n",
    "    return results\n",
    "\n",
    "#normalize_y_axis_threshold()"
