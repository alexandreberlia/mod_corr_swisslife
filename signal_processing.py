

   "#smooth_dataframe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Plot both graphs but on the same y axis ( normalize it on the first column/variable )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "def normalize_columns(df_to_normalize, df_reference, df_reference_column, df_to_normalize_column, save_excel='no', excel_name='Normalized Data', cell_width=75):\n",
    "    df_normalized = df_to_normalize.copy()\n",
    "    mean_ref = df_reference[df_reference_column].mean()\n",
    "    std_ref = df_reference[df_reference_column].std()\n",
    "    \n",
    "    mean_to_norm = df_normalized[df_to_normalize_column].mean()\n",
    "    std_to_norm = df_normalized[df_to_normalize_column].std()\n",
    "    \n",
    "    normalized_values = (df_normalized[df_to_normalize_column] - mean_to_norm) / std_to_norm * std_ref + mean_ref\n",
    "    df_normalized[df_to_normalize_column] = normalized_values\n",
    "\n",
    "    # Merge on index, and fill missing values using interpolation\n",
    "    merged_df = df_reference[[df_reference_column]].join(df_normalized[[df_to_normalize_column]], how='outer')\n",
    "    merged_df.interpolate(method='time', inplace=True)\n",
    "    merged_df.dropna(inplace=True)\n",
    "    if save_excel ==\"yes\":\n",
    "        # Create a new workbook\n",
    "        wb = Workbook()\n",
    "        ws = wb.active\n",
    "        ws.title = 'Normalized Data'\n",
    "\n",
    "        # Write the results to the worksheet\n",
    "        for i, col in enumerate(merged_df.columns, start=1):\n",
    "            ws[f'A{i}'] = col\n",
    "            for j, value in enumerate(merged_df[col], start=2):\n",
    "                ws.cell(row=j, column=i, value=value)\n",
    "\n",
    "        # Adjust dimensions automatically\n",
    "        adjust_dimensions(wb, max_column_width=cell_width)\n",
    "\n",
    "        # Save the workbook with the adjusted dimensions\n",
    "        wb.save(f\"{excel_name}.xlsx\")\n",
    "    \n",
    "    return merged_df\n",
    "\n",
    "#x = normalize_columns(df28, df1, df1.columns[0], df28.columns[0])\n",
    "#x.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "def normalize_y_axis(svgal_window = 75, polyorder = 3, number=1, window=104, threshold_small=0.5, threshold_big=0.7, decimal = 3):\n",
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
    "#normalize_y_axis()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Plot the median against the column for each dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    ,
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Further indices analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Creating a dataframe with the statistics of each series"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For each series, we want:\n",
    "* Min/Max and the dates for the Min/Max\n",
    "* The median value\n",
    "* The average value\n",
    "* The avg standard deviation value\n",
    "* A plot of the series over time\n",
    "    * The evolution of the standard deviation around the mean"
   ]
  },
 ,
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Starting the model"
   ]
  },
  ,
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### For each month, find what period we are in (recession, soft landing, recovery, expansion)"
   ]
  },
  ,
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Dictionary of economic cycles"
   ]
  },
  ,
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Find the correlation between the indices/economic cycles and the market's trends"
   ]
  },
  ,
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [],
   "source": [
    "#def max_corr(index_name):\n",
    "#    best_i = None\n",
    "#    best_j = None\n",
    "#    max_correlation1 = float('-inf')\n",
    "#    max_correlation2 = float('-inf')\n",
    "\n",
    "    # Using list to collect all results and then convert to DataFrame\n",
    "#    results = []\n",
    "\n",
    "#    for i in range(2000):\n",
    "#        for j in range(10):\n",
    "#            if j < i:\n",
    "#                df = corr_between_indices_market_trend(index_name=index_name, smooth_index='yes', indices_window=i, indices_order=j)\n",
    "#                current_correlation1 = df.iloc[0]\n",
    "#                current_correlation2 = df.iloc[1]\n",
    "#                results.append((i, j, current_correlation1, current_correlation2))\n",
    "\n",
    "    # Convert results to a DataFrame\n",
    "#    results_df = pd.DataFrame(results, columns=['i', 'j', 'correlation1', 'correlation2'])\n",
    "\n",
    "    # Calculate the combined correlation\n",
    "#    results_df['combined_correlation'] = results_df['correlation1'].abs() + results_df['correlation2'].abs()\n",
    "\n",
    "    # Find the row with the maximum combined correlation\n",
    "#    max_row = results_df.loc[results_df['combined_correlation'].idxmax()]\n",
    "\n",
    "#    best_i = max_row['i']\n",
    "#    best_j = max_row['j']\n",
    "#    max_correlation1 = max_row['correlation1']\n",
    "#    max_correlation2 = max_row['correlation2']\n",
    "\n",
    "#    return best_i, best_j, max_correlation1, max_correlation2\n",
    "\n",
    "#max_corr(\"NASDAQ (Stock Index)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {},
   "outputs": [],
   "source": [
    "#def max_corr(index_name):\n",
    "#    best_i = None\n",
    "#    best_j = None\n",
    "#    max_correlation1 = float('-inf')\n",
    "#    max_correlation2 = float('-inf')\n",
    "\n",
    "    \n",
    "#    for i in range(2000):\n",
    "#        for j in range(10):\n",
    "#            if j < i:\n",
    "#                df = corr_between_indices_market_trend(index_name=index_name, smooth_index='yes', indices_window=i, indices_order=j)\n",
    "#                current_correlation1 = df.iloc[0]\n",
    "#                current_correlation2 = df.iloc[1]\n",
    "#                if abs(current_correlation1) + abs(current_correlation2) > abs(max_correlation1) + abs(max_correlation2):\n",
    "#                    max_correlation1 = current_correlation1\n",
    "#                    max_correlation2 = current_correlation2\n",
    "#                    best_i = i\n",
    "#                    best_j = j\n",
    "        \n",
    "#    return best_i, best_j, max_correlation1, max_correlation2\n",
    "\n",
    "#max_corr(\"NASDAQ (Stock Index)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Plot it all"
   ]
  }
