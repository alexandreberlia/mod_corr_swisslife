
 
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Find the dataframes which match our needs"
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
 
