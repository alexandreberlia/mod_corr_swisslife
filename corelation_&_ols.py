
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Analysis of a chosen dataframe"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Corr Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "#df4.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### CORRELATION"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### See how the correlation evolves over time\n",
    "Graph the correlation of each variable against every other variable individually"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Note:** 19813 days between beginning and end of dataframe, so every 180 days means (19813/180) = 110 data points\n",
    "To have 110 data points, we divide 652/110 and this gives us 6, which is how many windows we should have (represents the skip in rows).\n",
    "\n",
    "However, due to how little information that results in, we set window=27, giving us the correlations for every two years"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### COVARIANCE"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### See how the covariance evolves over time\n",
    "Graph the covariance of each variable against every other variable individually"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Global analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Plotting two dataframes over time\n",
    "\n",
    "This will return $x*y$ plots, where x is the number of columns in the first dataframe and y is the number of columns in the second"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "def two_dataframes_over_time_global(dfx, dfy, folder_name='time_plots', image_format='png'):\n",
    "    merged_df = dfx.join(dfy, how='outer')\n",
    "\n",
    "    merged_df.interpolate(method='time', inplace=True)\n",
    "\n",
    "    # Ensure the output directory exists\n",
    "    if not os.path.exists(folder_name):\n",
    "        os.makedirs(folder_name)\n",
    "\n",
    "    for col1 in dfx.columns:\n",
    "        for col2 in dfy.columns:\n",
    "            fig, ax1 = plt.subplots(figsize=(20, 10))\n",
    "\n",
    "            ax1.set_xlabel('Time')\n",
    "            ax1.set_ylabel(f'{col1}', color='blue')\n",
    "            ax1.plot(merged_df.index, merged_df[f'{col1}'], color='blue', label=f'{col1}')\n",
    "            ax1.tick_params(axis='y', labelcolor='blue')\n",
    "\n",
    "            ax2 = ax1.twinx()\n",
    "            ax2.set_ylabel(f'{col2}', color='red')\n",
    "            ax2.plot(merged_df.index, merged_df[f'{col2}'], color='red', label=f'{col2}')\n",
    "            ax2.tick_params(axis='y', labelcolor='red')\n",
    "\n",
    "            ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=27))\n",
    "            ax1.tick_params(axis='x', rotation=20)\n",
    "\n",
    "            plt.title(f'{col1} and {col2} over time')\n",
    "\n",
    "            # Save the plot instead of showing it\n",
    "            file_name = f\"{col1}_{col2}_time.{image_format}\"  # Create a filename based on column names\n",
    "            file_path = os.path.join(folder_name, file_name)\n",
    "            plt.savefig(file_path, format=image_format, bbox_inches='tight')\n",
    "\n",
    "            plt.close()  # Close the plot to free up memory\n",
    "\n",
    "    return f\"Plots successfully saved in the '{folder_name}' folder.\"\n",
    "\n",
    "#two_dataframes_over_time(df1, df2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Plotting correlation between two dataframes over time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "def show_rolling_corr_global(dfx, dfy, window=105, folder_name='rolling_corr_plots', image_format='png'):\n",
    "    dfx = dfx.sort_index()\n",
    "    dfy = dfy.sort_index()\n",
    "    \n",
    "    merged_df = dfx.join(dfy, how='outer')\n",
    "\n",
    "    merged_df.interpolate(method='time', inplace=True)\n",
    "\n",
    "    # Ensure the output directory exists\n",
    "    if not os.path.exists(folder_name):\n",
    "        os.makedirs(folder_name)\n",
    "\n",
    "    for col1 in dfx.columns:\n",
    "        for col2 in dfy.columns:\n",
    "            rolling_corr = merged_df[col1].rolling(window=window).corr(merged_df[col2])\n",
    "\n",
    "            plt.figure(figsize=(10, 6))\n",
    "            plt.plot(rolling_corr, label='Rolling Correlation')\n",
    "            plt.title(f'Rolling Correlation between {col1} and {col2} over time')\n",
    "            plt.xlabel('Date')\n",
    "            plt.ylabel('Value of Correlation')\n",
    "            plt.legend()\n",
    "            plt.tight_layout()\n",
    "\n",
    "            # Set x-tick positions and labels for better readability\n",
    "            xtick_positions = rolling_corr.index[::max(1, len(rolling_corr)//10)]\n",
    "            xtick_labels = [date.strftime('%Y-%m-%d') for date in xtick_positions]\n",
    "            plt.xticks(ticks=xtick_positions, labels=xtick_labels, rotation=20)\n",
    "\n",
    "            # Save the plot instead of showing it\n",
    "            file_name = f\"{col1}_{col2}_rolling_corr.{image_format}\"  # Create a filename based on column names\n",
    "            file_path = os.path.join(folder_name, file_name)\n",
    "            plt.savefig(file_path, format=image_format, bbox_inches='tight')\n",
    "\n",
    "            plt.close()  # Close the plot to free up memory\n",
    "\n",
    "    return f\"Rolling correlation plots successfully saved in the '{folder_name}' folder.\"\n",
    "\n",
    "\n",
    "#show_rolling_corr_global(df1, df18)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Finding the average correlations over two years and selecting those above a certain threshold"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Global corr matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_corr_matrix_global(dataframe):\n",
    "    # Compute the correlation matrix\n",
    "    corr_matrix = dataframe.corr()\n",
    "\n",
    "    # Set up the matplotlib figure\n",
    "    plt.figure(figsize=(80, 80))\n",
    "\n",
    "    # Generate a mask for the upper triangle\n",
    "    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n",
    "\n",
    "    # Draw the heatmap with the mask and correct aspect ratio\n",
    "    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,\n",
    "                annot=True, fmt=\".2f\", square=True, linewidths=.5, cbar_kws={\"shrink\": .5})\n",
    "\n",
    "    # Rotate the x-axis labels for better readability\n",
    "    plt.xticks(rotation=90)\n",
    "\n",
    "    # Show plot\n",
    "    plt.show()\n",
    "\n",
    "# Example usage\n",
    "#create_corr_matrix_global(df_copy)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Intensity of correlation (COVARIANCE)\n",
    "\n",
    "We find the covariance between each variable and graph it"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Foundations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "def show_rolling_cov_global(dfx, dfy, window=105, folder_name='global_covariance_plots', image_format='png'):\n",
    "    dfx = dfx.sort_index()\n",
    "    dfy = dfy.sort_index()\n",
    "    \n",
    "    merged_df = dfx.join(dfy, how='outer')\n",
    "    merged_df.interpolate(method='time', inplace=True)\n",
    "\n",
    "    # Ensure the output directory exists\n",
    "    if not os.path.exists(folder_name):\n",
    "        os.makedirs(folder_name)\n",
    "\n",
    "    for col1 in dfx.columns:\n",
    "        for col2 in dfy.columns:\n",
    "            rolling_cov = merged_df[col1].rolling(window=window).cov(merged_df[col2])\n",
    "\n",
    "            plt.figure(figsize=(10, 6))\n",
    "            plt.plot(rolling_cov, label='Rolling Covariance')\n",
    "            plt.title(f'Rolling Covariance between {col1} and {col2} over time')\n",
    "            plt.xlabel('Date')\n",
    "            plt.ylabel('Value of Covariance')\n",
    "            plt.legend()\n",
    "            plt.tight_layout()\n",
    "\n",
    "            xtick_positions = rolling_cov.index[::max(1, len(rolling_cov)//10)]\n",
    "            xtick_labels = [date.strftime('%Y-%m-%d') for date in xtick_positions]\n",
    "            plt.xticks(ticks=xtick_positions, labels=xtick_labels, rotation=20)\n",
    "\n",
    "            # Save the plot with tight bounding box to avoid cut-off issues\n",
    "            file_name = f\"rolling_cov_{col1}_{col2}.{image_format}\"\n",
    "            file_path = os.path.join(folder_name, file_name)\n",
    "            plt.savefig(file_path, format=image_format, bbox_inches='tight')\n",
    "\n",
    "            plt.close()\n",
    "    return f\"Images successfully saved in the '{folder_name}' folder.\"\n",
    "\n",
    "#show_rolling_cov(df1, df18)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "def value_rolling_cov_global(df1, df2, window=105):\n",
    "    # Ensure the indices of df1 and df2 are datetime and aligned\n",
    "    df1 = df1.sort_index()\n",
    "    df2 = df2.sort_index()\n",
    "    df1.dropna(inplace=True)\n",
    "    df2.dropna(inplace=True)\n",
    "\n",
    "    # Align the DataFrames by their index (dates) using an outer join\n",
    "    merged_df = df1.join(df2, how='outer')\n",
    "\n",
    "    # Interpolate missing values (optional)\n",
    "    merged_df.interpolate(method='time', inplace=True)\n",
    "    merged_df.dropna(inplace=True)\n",
    "\n",
    "    # Get the column names of both dataframes\n",
    "    col_names_df1 = df1.columns\n",
    "    col_names_df2 = df2.columns\n",
    "\n",
    "    compared_pairs = set()\n",
    "    results = []\n",
    "\n",
    "    for col1 in col_names_df1:\n",
    "        for col2 in col_names_df2:\n",
    "            pair = (col1, col2)\n",
    "            if pair not in compared_pairs:\n",
    "                rolling_cov = merged_df[col1].rolling(window=window).cov(merged_df[col2])\n",
    "                avg_roll_cov_over_periods = rolling_cov.mean()\n",
    "                if np.isinf(avg_roll_cov_over_periods) or np.isnan(avg_roll_cov_over_periods):\n",
    "                    continue\n",
    "                else:\n",
    "                    result_string = f\"Average rolling covariance over {window} periods between {col1} and {col2}: {avg_roll_cov_over_periods}\"\n",
    "                    results.append(result_string)\n",
    "\n",
    "                    compared_pairs.add(pair)\n",
    "\n",
    "    return results\n",
    "\n",
    "# Example usage:\n",
    "# results = value_rolling_cov_global(df18, df26)\n",
    "# for result in results:\n",
    "#     print(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "def filter_2year_covariances_above_threshold_global(dfx, dfy, window=105, threshold_small=0.5):\n",
    "    results = value_rolling_cov_global(dfx, dfy, window=window)\n",
    "    filtered_results = []\n",
    "\n",
    "    for result in results:\n",
    "        cov_value = float(result.split(\":\")[-1].strip())\n",
    "        if cov_value > threshold_small or cov_value < -threshold_small:\n",
    "            filtered_results.append(result)\n",
    "\n",
    "    return filtered_results\n",
    "\n",
    "# Example usage:\n",
    "# filtered_results = filter_2year_covariances_above_threshold_global(df2, df21)\n",
    "# for result in filtered_results:\n",
    "#     print(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "def list_all_2year_covariance_values_above_threshold_global(window=105, threshold_small=0.5):\n",
    "    x = []\n",
    "    for i in range(len(dict_of_df)):\n",
    "        for j in range(i + 1, len(dict_of_df) + 1):\n",
    "            dfx_name = f\"df{i}\"\n",
    "            dfy_name = f\"df{j}\"\n",
    "            if dfx_name in globals() and dfy_name in globals():\n",
    "                dfx = globals()[dfx_name]\n",
    "                dfy = globals()[dfy_name]\n",
    "                dfx = dfx.dropna()\n",
    "                dfy = dfy.dropna()\n",
    "                x += filter_2year_covariances_above_threshold_global(dfx, dfy, window=window, threshold_small=threshold_small)\n",
    "    return x\n",
    "\n",
    "# Example usage:\n",
    "# p = list_all_2year_covariance_values_above_threshold_global()\n",
    "# pprint(p)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "# corr_pairs_global = get_pairs_from_results_global(list_all_2year_covariance_values_above_threshold_global())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_pairs_from_results_global_cov(window=105, threshold_small=0.5, save_excel='no', excel_name='Pairs', cell_width=75):\n",
    "    results = list_all_2year_covariance_values_above_threshold_global(window=window, threshold_small=threshold_small)\n",
    "    pairs = []\n",
    "    for result in results:\n",
    "        parts = result.split('between')[1].strip().split('and')\n",
    "        pair = (parts[0].strip(), parts[1].strip().split(':')[0].strip())\n",
    "        pairs.append(pair)\n",
    "    if save_excel == 'yes':\n",
    "        # Create a new workbook\n",
    "        wb = Workbook()\n",
    "        ws = wb.active\n",
    "        ws.title = 'Pairs'\n",
    "\n",
    "        # Write the results to the worksheet\n",
    "        for i, pair in enumerate(pairs, start=1):\n",
    "            ws[f'A{i}'] = pair[0]\n",
    "            ws[f'B{i}'] = pair[1]\n",
    "\n",
    "        # Adjust dimensions automatically\n",
    "        adjust_dimensions(wb, max_column_width=cell_width)\n",
    "\n",
    "        # Save the workbook with the adjusted dimensions\n",
    "        wb.save(f\"{excel_name}.xlsx\")\n",
    "    return pairs\n",
    "\n",
    "#pprint(corr_pairs_global)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "def retrieve_matrix_covariances_for_pairs_global(window=105, threshold_small=0.5):\n",
    "    pairs = get_pairs_from_results_global_cov(window=window, threshold_small=threshold_small)\n",
    "    pair_covariances = {}\n",
    "    for i in range(1, len(dict_of_df)):\n",
    "        for j in range(i + 1, len(dict_of_df) + 1):\n",
    "            dfx_name = f\"df{i}\"\n",
    "            dfy_name = f\"df{j}\"\n",
    "            if dfx_name in globals() and dfy_name in globals():\n",
    "                df1 = globals()[dfx_name]\n",
    "                df2 = globals()[dfy_name]\n",
    "                merged_df = df1.join(df2, how='outer').interpolate(method='time')\n",
    "                merged_df = merged_df.dropna()\n",
    "                cov_matrix = merged_df.cov()\n",
    "                for pair in pairs:\n",
    "                    if pair[0] in merged_df.columns and pair[1] in merged_df.columns:\n",
    "                        pair_covariances[pair] = cov_matrix.loc[pair[0], pair[1]]\n",
    "            else:\n",
    "                continue\n",
    "    return pair_covariances\n",
    "\n",
    "# Example usage:\n",
    "# pair_covariances_global = retrieve_matrix_covariances_for_pairs_global(corr_pairs_global)\n",
    "# pprint(pair_covariances_global)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "def collect_covariances_above_threshold_global(window=105, threshold_small=0.5, threshold_big=0.7):\n",
    "    x = list_all_2year_covariance_values_above_threshold_global(window=window, threshold_small=threshold_small)\n",
    "    corr_pairs = get_pairs_from_results_global_cov(window=window, threshold_small=threshold_small)\n",
    "    y = retrieve_matrix_covariances_for_pairs_global(window=window, threshold_small=threshold_small)\n",
    "    \n",
    "    grouped_values = []\n",
    "\n",
    "    for i in range(len(x)):\n",
    "        pair = corr_pairs[i]\n",
    "        \n",
    "        tuple_key_str = f\"{pair[0]}, {pair[1]}\"\n",
    "        \n",
    "        # Extract numeric value from x\n",
    "        x_parts = x[i].split(':')\n",
    "        x_number = float(x_parts[-1].strip())\n",
    "        \n",
    "        # Get the y value\n",
    "        y_value = y[pair]\n",
    "        \n",
    "        # Create tuple string and append to grouped_values\n",
    "        tuple_string = f\"{tuple_key_str}:\"\n",
    "        grouped_values.append((tuple_string, x_number, y_value))\n",
    "    \n",
    "    df = pd.DataFrame(grouped_values, columns=['Variables being compared', 'Average covariance value over two years', 'Value in the covariance matrix'])\n",
    "    \n",
    "    # Set 'Variables being compared' column as index and transpose the DataFrame\n",
    "    df.set_index('Variables being compared', inplace=True)\n",
    "    df = df.transpose()\n",
    "    \n",
    "    # Filter columns based on threshold_big\n",
    "    columns_to_keep = df.columns[df.iloc[1] >= threshold_big]\n",
    "    filtered_df = df[columns_to_keep]\n",
    "    \n",
    "    return filtered_df\n",
    "\n",
    "# Example usage:\n",
    "# collect_covariances_above_threshold_global(window=27, threshold_small=0.5, threshold_big=0.7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_smallest_covariance_difference_global(number=1, window=105, threshold_small=0.5, threshold_big=0.7):\n",
    "    dfx = collect_covariances_above_threshold_global(window=window, threshold_small=threshold_small, threshold_big=threshold_big)\n",
    "    columns_to_drop = []\n",
    "\n",
    "    for col in dfx.columns:\n",
    "        # Check if there are at least two rows to compute the difference\n",
    "        if dfx.shape[0] > 1 and abs(dfx[col].diff().iloc[1]) >= number:\n",
    "            columns_to_drop.append(col)\n",
    "\n",
    "    dfx.drop(columns=columns_to_drop, inplace=True)\n",
    "\n",
    "    return dfx\n",
    "\n",
    "# Example usage:\n",
    "# compute_smallest_covariance_difference_global(number=0.1, window=27, threshold_small=0.5, threshold_big=0.7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_close_cov_variable_pairs_global(number=1, window=105, threshold_small=0.5, threshold_big=0.7):\n",
    "    df = compute_smallest_covariance_difference_global(number=number, window=window, threshold_small=threshold_small, threshold_big=threshold_big)\n",
    "    variable_pairs = []\n",
    "    \n",
    "    for col in df.columns:\n",
    "        # Split the column name by ', ' and add the resulting pair to the list\n",
    "        pair = col.split(', ')\n",
    "        pair[1] = pair[1].rstrip(':')\n",
    "        variable_pairs.append((pair[0], pair[1]))\n",
    "    \n",
    "    return variable_pairs\n",
    "\n",
    "# Example usage:\n",
    "# extract_close_cov_variable_pairs_global(number=1, window=27, threshold_small=0.5, threshold_big=0.7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_close_cov_variable_pairs_global(number=1, window=105, threshold_small=0.5, threshold_big=0.7, decimal=3, folder_name='close_global_covariance_plots', image_format='png'):\n",
    "    variable_pairs = extract_close_cov_variable_pairs_global(number=number, window=window, threshold_small=threshold_small, threshold_big=threshold_big)\n",
    "    dfx = collect_covariances_above_threshold_global(window=window, threshold_small=threshold_small, threshold_big=threshold_big)\n",
    "\n",
    "    # Ensure the output directory exists\n",
    "    if not os.path.exists(folder_name):\n",
    "        os.makedirs(folder_name)\n",
    "\n",
    "    for x, y in variable_pairs:\n",
    "        plotted = False\n",
    "\n",
    "        for i in range(len(dict_of_df)):\n",
    "            for j in range(i + 1, len(dict_of_df) + 1):\n",
    "                dfx_name = f\"df{i}\"\n",
    "                dfy_name = f\"df{j}\"\n",
    "\n",
    "                if dfx_name in globals() and dfy_name in globals():\n",
    "                    df1 = globals()[dfx_name]\n",
    "                    df2 = globals()[dfy_name]\n",
    "\n",
    "                    if x in df1.columns and y in df2.columns:\n",
    "                        merged_df = df1.join(df2, how='outer')\n",
    "                        merged_df.interpolate(method='time', inplace=True)\n",
    "\n",
    "                        # Find common index range where both x and y have non-null values\n",
    "                        common_index = merged_df[[x, y]].dropna().index\n",
    "                        if len(common_index) > 0:\n",
    "                            min_idx = common_index.min()\n",
    "                            max_idx = common_index.max()\n",
    "\n",
    "                            # Filter the merged_df to the common index range\n",
    "                            zoomed_df = merged_df.loc[min_idx:max_idx]\n",
    "\n",
    "                            # Plotting with zoomed-in x-axis and dynamic y-axis\n",
    "                            fig, ax1 = plt.subplots(figsize=(20, 10))\n",
    "\n",
    "                            ax1.set_xlabel('Time')\n",
    "                            ax1.set_ylabel(f'{x}', color='blue')\n",
    "                            ax1.plot(merged_df.index, merged_df[x], color='blue', label=f'{x}')\n",
    "                            ax1.tick_params(axis='y', labelcolor='blue')\n",
    "\n",
    "                            ax2 = ax1.twinx()\n",
    "                            ax2.set_ylabel(f'{y}', color='red')\n",
    "                            ax2.plot(merged_df.index, merged_df[y], color='red', label=f'{y}')\n",
    "                            ax2.tick_params(axis='y', labelcolor='red')\n",
    "\n",
    "                            # Set x-axis limits to zoom in\n",
    "                            ax1.set_xlim(min_idx, max_idx)\n",
    "\n",
    "                            # Dynamic y-axis limits\n",
    "                            ax1.set_ylim(zoomed_df[x].min(), zoomed_df[x].max())\n",
    "                            ax2.set_ylim(zoomed_df[y].min(), zoomed_df[y].max())\n",
    "\n",
    "                            ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=27))\n",
    "                            ax1.tick_params(axis='x', rotation=20)\n",
    "\n",
    "                            # Retrieve the annotation values from dfx\n",
    "                            pair_key = f\"{x}, {y}:\"\n",
    "                            if pair_key in dfx.columns:\n",
    "                                avg_cov_value = dfx[pair_key]['Average covariance value over two years']\n",
    "                                matrix_cov_value = dfx[pair_key]['Value in the covariance matrix']\n",
    "                                annotation_text = f\"Avg Cov (2 years) ≈ {avg_cov_value:.{decimal}f}\\nMatrix Cov ≈ {matrix_cov_value:.{decimal}f}\"\n",
    "\n",
    "                                # Annotate the graph\n",
    "                                ax1.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction',\n",
    "                                             fontsize=12, bbox=dict(boxstyle=\"round,pad=0.3\", edgecolor='black', facecolor='white'))\n",
    "\n",
    "                            # Save the plot with a dynamic filename\n",
    "                            file_name = f\"close_cov_{x}_{y}.{image_format}\"\n",
    "                            file_path = os.path.join(folder_name, file_name)\n",
    "                            plt.title(f'{x} and {y} over time')\n",
    "                            plt.savefig(file_path, format=image_format, bbox_inches='tight')\n",
    "\n",
    "                            plt.close()  # Close the plot to free up memory\n",
    "\n",
    "                            plotted = True\n",
    "\n",
    "        if not plotted:\n",
    "            print(f\"Variables {x} and {y} not found in any dataframe.\")\n",
    "\n",
    "    return f\"Images successfully saved in the '{folder_name}' folder.\"\n",
    "\n",
    "#plot_close_cov_variable_pairs_global(number=1, threshold_small=0.4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Finding Maximas"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Global average"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "def global_avg_all_corr_parametrized(number=1, window=104, threshold_small=0.5, threshold_big=0.7, decimal = 3):\n",
    "    df = all_corr_parametrized(number=number, window=window, threshold_small=threshold_small, threshold_big=threshold_big, decimal=decimal)\n",
    "    \n",
    "    if df.empty:\n",
    "        return float('NaN')  # Return NaN if DataFrame is empty\n",
    "    \n",
    "    means = []\n",
    "    for row_name in df.index:\n",
    "        means.append(df.loc[row_name].mean())\n",
    "    \n",
    "    if means:\n",
    "        return sum(means) / len(means)  # Return average of means\n",
    "    else:\n",
    "        return float('NaN')  # Return NaN if no means were calculated\n",
    "\n",
    "\n",
    "#global_avg_all_corr_parametrized()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Find the median of each series in the final dataframe"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Put all the rows in a list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [],
   "source": [
    "def rows_in_list(number=1, window=104, threshold_small=0.5, threshold_big=0.7, decimal=3, save_excel='no', excel_name='Pairs List', cell_width=75):\n",
    "    indexes_list = []\n",
    "    df = all_corr_parametrized(number=number, window=window, threshold_small=threshold_small, threshold_big=threshold_big, decimal=decimal)\n",
    "    for i in df.index:\n",
    "        indexes_list.append(i)\n",
    "\n",
    "    if save_excel == 'yes':\n",
    "        # Create a new workbook\n",
    "        wb = Workbook()\n",
    "        ws = wb.active\n",
    "        ws.title = 'Pairs'\n",
    "\n",
    "        # Write the results to the worksheet\n",
    "        for i, index in enumerate(indexes_list, start=1):\n",
    "            ws[f'A{i}'] = index\n",
    "\n",
    "        # Adjust dimensions automatically\n",
    "        adjust_dimensions(wb, max_column_width=cell_width)\n",
    "\n",
    "        # Save the workbook with the adjusted dimensions\n",
    "        wb.save(f\"{excel_name}.xlsx\")\n",
    "    return indexes_list\n",
    "\n",
    "#rows_in_list()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Split the strings into the two columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "def list_of_global_all_corr_parametrized(number=1, window=104, threshold_small=0.5, threshold_big=0.7, decimal = 3):\n",
    "    list_of_rows = rows_in_list(number=number, window=window, threshold_small=threshold_small, threshold_big=threshold_big, decimal=decimal)\n",
    "    variables = []\n",
    "    for s in list_of_rows:\n",
    "            parts = [part.strip() for part in s.split(',')]\n",
    "            parts[1] = parts[1][:-1]\n",
    "            variables.append(parts)\n",
    "    return variables\n",
    "\n",
    "#list_of_global_all_corr_parametrized(threshold_small = 0, threshold_big = 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Smooth the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "def smooth_dataframe(svgal_window=75, polyorder=3, number=1, window=104, threshold_small=0.5, threshold_big=0.7, decimal=3):\n",
    "    \"\"\"\n",
    "    Apply Savitzky-Golay smoothing filter to each column of the dataframe.\n",
    "    \n",
    "    Parameters:\n",
    "    - df: Pandas DataFrame containing the data to be smoothed.\n",
    "    - window: The length of the window. Must be an odd integer.\n",
    "    - polyorder: The order of the polynomial used in the filtering. Typically 3 or less.\n",
    "    \n",
    "    Returns:\n",
    "    - smoothed_df: Pandas DataFrame with smoothed values.\n",
    "    \"\"\"\n",
    "    svgal_df = []\n",
    "    for i in range(1, len(dict_of_df) + 1):\n",
    "        df_name = f'df{i}'\n",
    "        if df_name in globals():\n",
    "            df = globals()[df_name]\n",
    "            smoothed_df = pd.DataFrame(index=df.index)\n",
    "            for col in df.columns:\n",
    "                smoothed_values = savgol_filter(df[col], svgal_window, polyorder)\n",
    "                smoothed_df[col] = smoothed_values\n",
    "            svgal_df.append(smoothed_df)\n",
    "    return svgal_df\n",
    "\n",
    
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "metadata": {},
   "outputs": [],
   "source": [
    "def find_periods_above_below_threshold______________(number=1, window=104, threshold_small=0.5, threshold_big=0.7, decimal=3, svgal_window=75, polyorder=3):\n",
    "    df_list = normalize_y_axis_threshold(number=number, window=window, threshold_small=threshold_small, threshold_big=threshold_big, decimal=decimal, svgal_window=svgal_window, polyorder=polyorder)\n",
    "    \n",
    "    all_periods = []\n",
    "    \n",
    "    for df_item in df_list:\n",
    "        columns = df_item[0]\n",
    "        df = df_item[1]\n",
    "        periods = []\n",
    "        \n",
    "        for col in columns:\n",
    "            if \"(GDP)\" in col:\n",
    "                threshold = 0\n",
    "                above_threshold = df.iloc[0][col] > threshold\n",
    "                current_period = [df.index[0], None, above_threshold]\n",
    "                \n",
    "                for index, row in df.iterrows():\n",
    "                    value = row[col]\n",
    "                    \n",
    "                    if value > threshold and not above_threshold:\n",
    "                        current_period[1] = index\n",
    "                        periods.append((current_period[0], current_period[1], current_period[2], col, df))\n",
    "                        current_period = [index, None, True]\n",
    "                        above_threshold = True\n",
    "                    elif value < threshold and above_threshold:\n",
    "                        current_period[1] = index\n",
    "                        periods.append((current_period[0], current_period[1], current_period[2], col, df))\n",
    "                        current_period = [index, None, False]\n",
    "                        above_threshold = False\n",
    "                \n",
    "                if current_period[1] is None:\n",
    "                    current_period[1] = df.index[-1]\n",
    "                    periods.append((current_period[0], current_period[1], current_period[2], col, df))\n",
    "        \n",
    "        all_periods.append(periods)\n",
    "    \n",
    "    return all_periods\n",
    "\n",
    "#find_periods_above_below_threshold()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "metadata": {},
   "outputs": [],
   "source": [
    "def extreme_values_of_all_dfs_threshold______________(number=1, window=104, threshold_small=0.5, threshold_big=0.7, decimal=3, svgal_window=75, polyorder=3):\n",
    "    list_final = []\n",
    "    periods_list = find_periods_above_below_threshold(number=number, window=window, threshold_small=threshold_small, threshold_big=threshold_big, decimal=decimal, svgal_window=svgal_window, polyorder=polyorder)\n",
    "    \n",
    "    for periods in periods_list:\n",
    "        list_of_lists = []\n",
    "        for period in periods:\n",
    "            start_index, end_index, above_threshold, col_name, df = period\n",
    "            period_data = df.loc[start_index:end_index]\n",
    "            \n",
    "            if above_threshold:\n",
    "                max_value = period_data[col_name].max()\n",
    "                max_date = period_data[col_name].idxmax()\n",
    "                above = 'Above threshold'\n",
    "                x = [start_index, end_index, above, max_value, max_date, [col_name]]\n",
    "                list_of_lists.append(x)\n",
    "            else:\n",
    "                min_value = period_data[col_name].min()\n",
    "                min_date = period_data[col_name].idxmin()\n",
    "                below = 'Below threshold'\n",
    "                y = [start_index, end_index, below, min_value, min_date, [col_name]]\n",
    "                list_of_lists.append(y)\n",
    "        list_final.append(list_of_lists)\n",
    "    \n",
    "    return list_final\n",
    "\n",
    "#extreme_values_of_all_dfs_threshold()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_columns_with_threshold_and_extremes______________(number=2, window=104, threshold_small=0.5, threshold_big=0.7, decimal=3, svgal_window=75, polyorder=3):\n",
    "    df_list = normalize_y_axis(number=number, window=window, threshold_small=threshold_small, threshold_big=threshold_big, decimal=decimal, svgal_window=svgal_window, polyorder=polyorder)\n",
    "    extreme_values = extreme_values_of_all_dfs_threshold(number=number, window=window, threshold_small=threshold_small, threshold_big=threshold_big, decimal=decimal, svgal_window=svgal_window, polyorder=polyorder)\n",
    "    x = 0\n",
    "\n",
    "    colors = ['blue', 'orange']\n",
    "    dot_colors = ['blue', 'orange']\n",
    "\n",
    "    for df_item in df_list:\n",
    "        [df_name_x, df_name_y], df = df_item\n",
    "        if ('(GDP)' in df_name_x or '(GDP)' in df_name_y) and ('(Employment)' in df_name_x or '(Employment)' in df_name_y):\n",
    "            x += 1\n",
    "            title = f'{x}. {df_name_x} and {df_name_y} with Threshold Line and Extremes'\n",
    "            print(title)\n",
    "            \n",
    "            plt.figure(figsize=(12, 6))\n",
    "            plt.axhline(y=0, color='red', linestyle='--', label='Threshold = 0')\n",
    "            \n",
    "            for col_idx, col in enumerate(df.columns):\n",
    "                color = colors[col_idx % len(colors)]\n",
    "                plt.plot(df.index, df[col], label=f'{col}', color=color, linewidth=1.5)\n",
    "\n",
    "            dot_color_idx = 0\n",
    "            for extremes in extreme_values:\n",
    "                for extreme in extremes:\n",
    "                    if df_name_x in extreme[5] or df_name_y in extreme[5]:\n",
    "                        extreme_values_list = [extreme[3]]\n",
    "                        extreme_dates_list = [extreme[4]]\n",
    "\n",
    "                        for extreme_date, extreme_value in zip(extreme_dates_list, extreme_values_list):\n",
    "                            plt.scatter(extreme_date, extreme_value, color=dot_colors[dot_color_idx], edgecolor='black', zorder=5, label='Extreme Value' if col not in locals() else \"\")\n",
    "                            locals()[col] = True\n",
    "                            dot_color_idx = 1 - dot_color_idx\n",
    "                \n",
    "            plt.title(title)\n",
    "            plt.xlabel('Date')\n",
    "            plt.ylabel('Value')\n",
    "            plt.legend()\n",
    "            plt.show()\n",
    "        else:\n",
    "            continue\n",
    "\n",
    "#plot_columns_with_threshold_and_extremes(threshold_big=0, threshold_small=0)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Smooth the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {},
   "outputs": [],
   "source": [
    "def smooth_the_dataframe______________(df, svgal_window=75, polyorder=3):\n",
    "    smoothed_df = pd.DataFrame(index=df.index)\n",
    "    for col in df.columns:\n",
    "        smoothed_values = savgol_filter(df[col].values, svgal_window, polyorder)\n",
    "        smoothed_df[col] = smoothed_values\n",
    "    return smoothed_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Put all the pairs in a list"
   ]
  },
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
  {
   "cell_type": "code",
   "execution_count": 162,
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_and_plot_pairs______________(pair_dataframes, svgal_window=300, polyorder=2):\n",
    "    for idx, (df_pair1, df_pair2) in enumerate(pair_dataframes):\n",
    "        smoothed_df1 = smooth_the_dataframe(df_pair1, svgal_window, polyorder)\n",
    "        smoothed_df2 = smooth_the_dataframe(df_pair2, svgal_window, polyorder)\n",
    "        normalized_df1, normalized_df2 = normalize_columns(smoothed_df1, smoothed_df2)\n",
    "        \n",
    "        # Find the overlapping date range\n",
    "        common_index = normalized_df1.index.intersection(normalized_df2.index)\n",
    "        normalized_df1 = normalized_df1.loc[common_index]\n",
    "        normalized_df2 = normalized_df2.loc[common_index]\n",
    "        \n",
    "        # Calculate the median\n",
    "        median_df1 = normalized_df1.median().median()\n",
    "        median_df2 = normalized_df2.median().median()\n",
    "\n",
    "        # Plot the data\n",
    "        plt.figure(figsize=(10, 6))\n",
    "        plt.plot(normalized_df1.index, normalized_df1[normalized_df1.columns[0]], label=normalized_df1.columns[0])\n",
    "        if len(normalized_df1.columns) > 1:\n",
    "            plt.plot(normalized_df1.index, normalized_df1[normalized_df1.columns[1]], label=normalized_df1.columns[1])\n",
    "        plt.plot(normalized_df2.index, normalized_df2[normalized_df2.columns[0]], label=normalized_df2.columns[0])\n",
    "        if len(normalized_df2.columns) > 1:\n",
    "            plt.plot(normalized_df2.index, normalized_df2[normalized_df2.columns[1]], label=normalized_df2.columns[1])\n",
    "\n",
    "        # Plot median lines\n",
    "        plt.axhline(y=median_df1, color='blue', linestyle='--', label=f'Median {normalized_df1.columns[0]}')\n",
    "        plt.axhline(y=median_df2, color='green', linestyle='--', label=f'Median {normalized_df2.columns[0]}')\n",
    "\n",
    "        plt.legend()\n",
    "        plt.xlabel('Date')\n",
    "        plt.ylabel('Normalized Values')\n",
    "        plt.title(f'Normalized and Smoothed Plot: Pair {idx + 1}')\n",
    "        plt.show()\n",
    "\n",
    "#process_and_plot_pairs(create_pair_dataframes(put_pairs_in_list(GDP_df, Employment_df)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Graph the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "metadata": {},
   "outputs": [],
   "source": [
    "def two_dataframes_over_time______________(dfx, dfy):\n",
    "    for i in range(len(dfx.columns)):\n",
    "        for j in range(len(dfy.columns)):\n",
    "            if (\"Dates for\" in dfx.columns[i]) and (\"Dates for\" in dfy.columns[j]):\n",
    "                selected_columns1 = [dfx.columns[i], dfx.columns[i+1]]\n",
    "                df1 = dfx[selected_columns1]\n",
    "                df1.set_index(dfx.columns[i], inplace=True)\n",
    "                selected_columns2 = [dfy.columns[j], dfy.columns[j+1]]\n",
    "                df2 = dfy[selected_columns2]\n",
    "                df2.set_index(dfy.columns[j], inplace=True)\n",
    "                merged_df = df1.join(df2, how='inner')\n",
    "                merged_df.interpolate(method='linear', inplace=True)\n",
    "                \n",
    "                two_variables_over_time(merged_df)\n",
    "            else:\n",
    "                continue\n",
    "\n",
    "#two_dataframes_over_time(GDP_df, Employment_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Choses à faire\n",
    "<s>\n",
    "\n",
    "1. Remplace les codes par les noms\n",
    "\n",
    "2. Régler le problème des dernières valeurs qui n'apparaissent pas\n",
    "\n",
    "3. Implémenter de nouvelles variables\n",
    "\n",
    "4. Visualiser le tout\n",
    "\n",
    "5. Modifier les graphiques afin de garder seulement la période durant laquelle les dates correspondent\n",
    "\n",
    "6. Mettre le niveau de corrélation sur le graphique\n",
    "\n",
    "7. Rajouter les variables qui viennent du même dataframe\n",
    "\n",
    "8. Supérieur à 0.5 court-terme (27 windows) et supérieur à 0.7 long terme (toute la matrice)\n",
    "\n",
    "9. Résoudre le problème de rolling (tout mettre en weekly (ou daily) afin de pouvoir changer les rolls)\n",
    "\n",
    "10. Grapher la médiane face à la distribution et trouver les max et min\n",
    "    - Tout mettre sur le meme axe\n",
    "\n",
    "11. Voir quelle série est en avance sur l'autre\n",
    "    - Trouver quel facteur est un leader\n",
    "\n",
    "12. Trouver la dynamique des derniers trimestre du GDP\n",
    "\n",
    "</s>"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
