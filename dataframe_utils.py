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
  }
