"def count_dfs(df):\n",
    "    x = df.iloc[1, ::2].nunique()\n",
    "    return x\n",
    "\n",
    "count_dfs(df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**THERE ARE {count_dfs() + 1} DATAFRAMES TO BE MADE**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Make a copy of the dataframe for later"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "#df_copy = df.copy()\n",
    "#df_copy.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Makes the dataframes by grouping the values by dates"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Dataframe Creation Loop"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"for x in len(df): #we want x to be a column in our dataframe, and len(df) to refer to how many columns it has\n",
    "    if x = #our chosen column:\n",
    "        #we add the next column to a new dataframe df1\n",
    "    else\n",
    "        #we go on to the next column\n",
    "                \"\"\"\n",
    "def create_df(df):\n",
    "    chosen_column = df.columns[0]\n",
    "    new_df = pd.DataFrame()\n",
    "    new_df[chosen_column] = df[chosen_column]\n",
    "\n",
    "    cols_to_drop = []\n",
    "\n",
    "    for i in range(len(df.columns)):\n",
    "        if df[df.columns[i]].equals(df[chosen_column]) == True:\n",
    "            if i+1 < len(df.columns):\n",
    "                next_column = df.columns[i + 1]\n",
    "                new_df[next_column] = df[next_column]\n",
    "                current_column = df.columns[i]\n",
    "                cols_to_drop.extend([current_column, next_column])\n",
    "        else:\n",
    "            continue\n",
    "    df.drop(columns= cols_to_drop, inplace=True)\n",
    "    new_df.set_index(chosen_column, inplace=True)\n",
    "    return new_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Isolate the dataframes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_dataframes(df):\n",
    "    num_dataframes = count_dfs(df)\n",
    "\n",
    "    for i in range(1, num_dataframes + 3):\n",
    "        new_df = create_df(df)\n",
    "        # Drop NaN and NaT values\n",
    "        new_df = new_df.dropna()\n",
    "        var_name = f'df{i}'\n",
    "        globals()[var_name] = new_df\n",
    "        if new_df.empty:\n",
    "            break\n",
    "\n",
    "generate_dataframes(df)"
