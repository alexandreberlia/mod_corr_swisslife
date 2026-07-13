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
    
