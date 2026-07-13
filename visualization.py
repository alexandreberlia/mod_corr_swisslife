"def create_corr_matrix(dataframe):\n",
    "    corr_matrix = dataframe.corr()\n",
    "    sns.heatmap(corr_matrix, annot= True)\n",
    "    plt.xticks(rotation = 90)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "    return corr_matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Lineplots and normalization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "def standard_lineplot(dataframe):\n",
    "    dataframe.plot()\n",
    "    plt.xlabel('Time')\n",
    "    plt.ylabel('Value')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    return dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "def return_column_names(dataframe):\n",
    "    column_names = dataframe.columns.tolist()\n",
    "    return column_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'darkblue', 'magenta', 'violet', 'chocolate', 'darkgreen', 'yellow']\n",
    "\n",
    "def plot_all_of_a_df(dataframe):\n",
    "    column_names = return_column_names(dataframe)\n",
    "\n",
    "    fig, ax1 = plt.subplots(figsize=(20, 10))\n",
    "\n",
    "    # Plot the first axis\n",
    "    ax1.set_xlabel('Time')\n",
    "    ax1.set_ylabel(f'{column_names[0]}', color=colors[0])\n",
    "    ax1.plot(dataframe.index, dataframe[column_names[0]], color=colors[0], label=column_names[0])\n",
    "    ax1.tick_params(axis='y', labelcolor=colors[0])\n",
    "\n",
    "    # Initialize the current axis for further twinx\n",
    "    current_ax = ax1\n",
    "\n",
    "    # Loop through the remaining columns to create twin axes\n",
    "    for i in range(1, len(column_names)):\n",
    "        new_ax = ax1.twinx()\n",
    "        new_ax.spines['right'].set_position(('outward', 60 * i))\n",
    "        new_ax.set_ylabel(f'{column_names[i]}', color=colors[i % len(colors)])\n",
    "        new_ax.plot(dataframe.index, dataframe[column_names[i]], color=colors[i % len(colors)], label=column_names[i])\n",
    "        new_ax.tick_params(axis='y', labelcolor=colors[i % len(colors)])\n",
    "        current_ax = new_ax\n",
    "\n",
    "    # Combine legends from all axes\n",
    "    lines = []\n",
    "    labels = []\n",
    "    for ax in fig.axes:\n",
    "        ax_lines, ax_labels = ax.get_legend_handles_labels()\n",
    "        lines.extend(ax_lines)\n",
    "        labels.extend(ax_labels)\n",
    "\n",
    "    ax1.legend(lines, labels, loc='best')\n",
    "    ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=15))  # Adjust nbins as needed for density\n",
    "    ax1.tick_params(axis='x', rotation=20)  # Adjust rotation angle as needed\n",
    "\n",
    "    fig.tight_layout()\n",
    "    plt.title('Evolution of index values over time')\n",
    "    plt.show()\n",
    "\n",
    "#plot_all_of_a_df(df4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Individual Comparisons"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### GRAPHS\n",
    "\n",
    "Graph each variables against every other variable individually\n",
    "\n",
    "This will return $ \\sum_{i=1}^{n} (i-1) $ graphs, where n is the number of columns of the dataframe\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "def two_variables_over_time(dataframe):\n",
    "    column_names = return_column_names(dataframe)\n",
    "    compared_pairs = set()  # Set to store already compared pairs\n",
    "\n",
    "    for i in range(len(column_names)):\n",
    "        for j in range(i + 1, len(column_names)):  # Iterate from i+1 to len(column_names) to avoid duplicates\n",
    "            pair = (column_names[i], column_names[j])\n",
    "            if pair not in compared_pairs:\n",
    "                fig, ax1 = plt.subplots(figsize=(20, 10))\n",
    "\n",
    "                ax1.set_xlabel('Time')\n",
    "                ax1.set_ylabel(f'{column_names[i]}', color='blue')\n",
    "                ax1.plot(dataframe.index, dataframe[f'{column_names[i]}'], color='blue', label=f'{column_names[i]}')\n",
    "                ax1.tick_params(axis='y', labelcolor='blue')\n",
    "\n",
    "                ax2 = ax1.twinx()\n",
    "                ax2.set_ylabel(f'{column_names[j]}', color='red')\n",
    "                ax2.plot(dataframe.index, dataframe[f'{column_names[j]}'], color='red', label=f'{column_names[j]}')\n",
    "                ax2.tick_params(axis='y', labelcolor='red')\n",
    "\n",
    "                ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=30))  # Adjust nbins as needed for density\n",
    "                ax1.tick_params(axis='x', rotation=20)  # Adjust rotation angle as needed\n",
    "\n",
    "                plt.title(f'{column_names[i]} and {column_names[j]} over time')\n",
    "                plt.show()\n",
    "\n",
    "                compared_pairs.add(pair)  # Add the pair to the set of compared pairs\n",
    "\n",
    "#two_variables_over_time(df4)"
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {},
   "outputs": [],
   "source": [
    "#alternations_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Performances graphs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {},
   "outputs": [],
   "source": [
    "#plot_combined_cycles_and_stds(df = predict_values_model_1(new_model='no', what_to_predict = 'Economic cycle', help_to_predict1 = 'Sum of standard deviation values',\n",
    "#                                                          help_to_predict2 = 'Sum of recession/expansion values', plot = 'no', return_predictions = 'no',\n",
    "#                                                          return_dataframe = 'yes', smooth_filter = 'yes', svgal_window=50, polyorder=3, smooth_both_columns = 'no',\n",
    "#                                                          index_which_column_to_smooth = 1, smooth_economic_cycle = 'no', svgal_eco_cycle = 75, eco_poly = 6,\n",
    "#                                                          smooth_fed = 'yes', fed_as_pctg = 'yes', svgal_fed_cycle = 150, fed_poly = 3, sp_as_pctg = 'yes',\n",
    "#                                                          smooth_sp500 = 'no', msci_as_pctg = 'yes', smooth_msci = 'yes', nasdaq_as_pctg = 'yes', smooth_nasdaq = 'yes',\n",
    "#                                                          indices_window = 500, indices_order = 5, freq = 'W', sp_sub_ind_as_pctg = 'yes', smooth_sp_sub_ind = 'yes'),\n",
    "#                              dont_plot_series='yes', plot_both_series='yes', normalize_series='no', smooth_filter='no',\n",
    "#                              plot_economic_cycle='yes', plot_predictions='yes',\n",
    "#                              plot_sp500='yes', sp_as_pctg='no', smooth_sp500='no',\n",
    "#                              plot_msci='no', msci_as_pctg='yes', smooth_msci='yes',\n",
    "#                              plot_nasdaq='no', smooth_nasdaq='yes', nasdaq_as_pctg='yes',\n",
    "#                              nbins = 35, plot_ci= \"no\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {},
   "outputs": [],
   "source": [
    "#plot_combined_cycles_and_stds(df = predict_values_model_2(new_model='yes', what_to_predict = 'Economic cycle', help_to_predict1 = 'Sum of standard deviation values',\n",
    "#                                                          help_to_predict2 = 'Sum of recession/expansion values', plot = 'no', return_predictions = 'no', \n",
    "#                                                          return_dataframe = 'yes', smooth_filter = 'yes', svgal_window=50, polyorder=3, smooth_both_columns = 'no', \n",
    "#                                                          index_which_column_to_smooth = 1, smooth_economic_cycle = 'no', svgal_eco_cycle = 75, eco_poly = 6, \n",
    "#                                                          smooth_fed = 'yes', fed_as_pctg = 'yes', svgal_fed_cycle = 150, fed_poly = 3, sp_as_pctg = 'yes', \n",
    "#                                                          smooth_sp500 = 'yes', msci_as_pctg = 'yes', smooth_msci = 'yes', nasdaq_as_pctg = 'yes', smooth_nasdaq = 'yes', \n",
    "#                                                          indices_window = 500, indices_order = 5, freq = 'W', sp_sub_ind_as_pctg = 'yes', smooth_sp_sub_ind = 'yes',\n",
    "#                                                          confidence_level=0.8),\n",
    "#                              dont_plot_series='yes', plot_both_series='yes', normalize_series='yes', smooth_filter='no',\n",
    "#                              plot_economic_cycle='no', plot_predictions='yes',\n",
    "#                              plot_sp500='no', sp_as_pctg='yes', smooth_sp500='yes',\n",
    "#                              plot_msci='no', msci_as_pctg='yes', smooth_msci='yes',\n",
    "#                              plot_nasdaq='no', smooth_nasdaq='yes', nasdaq_as_pctg='yes',\n",
    "#                              plot_ci='no', nbins=35)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## To be hidden for now"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [],
   "source": [
    "#for y in l1:\n",
    "#    if y not in pivot_values_dict:\n",
    "#        print(y)"
   ]
  }


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
  }
