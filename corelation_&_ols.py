
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
