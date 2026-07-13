
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
