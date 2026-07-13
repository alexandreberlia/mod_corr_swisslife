current_directory = os.getcwd()

def load_and_preview_csv(file_name):
    # Get the current working directory

    # Construct the full file path
    file_path = os.path.join(current_directory, file_name)

    # Read the CSV file with the specified delimiter
    df = pd.read_csv(file_path, delimiter=";", low_memory=False)

    # Return the dataframe
    return df

global df
df = load_and_preview_csv("Data_in_value.csv")
