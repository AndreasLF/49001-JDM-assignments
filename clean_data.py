import pandas as pd
import os
import pdb
import numpy as np

def process_fat_column(value):
    # if nan return 0
    if pd.isnull(value):
        return 0

    try:
        # Check if the value contains a sum (e.g., "65+900")
        if "+" in str(value):
            # Evaluate the sum
            return int(sum(map(float, value.split("+"))))
        else:
            # Convert float strings to integer
            return int(float(value))
    except ValueError:
        # Handle any unexpected values that cannot be converted
        print(f"Warning: Could not convert {value} to an integer.")
        return None
    
def clean_date_column(value):
    try:
        # Try to parse the date using pandas
        date = pd.to_datetime(value, errors='raise')
        return date
    except (ValueError, TypeError):
        # If parsing fails, try to extract the month and year manually
        parts = str(value).split()
        if len(parts) == 3 and parts[0].lower() == 'xx':
            # If the format is "xx mm yyyy", construct a date with just month and year
            try:
                return pd.to_datetime(f"{parts[1]} {parts[2]}", format="%b %Y")
            except ValueError:
                return np.nan
        else:
            return np.nan


if __name__ == "__main__":

    # Load all csv files from the data folder and concatenate them into a single DataFrame
    all_data = pd.concat([pd.read_csv(f"data/flight_models/{file}") for file in os.listdir("data/flight_models") if file.endswith(".csv")])

    # remove duplicate rows where every column is the same
    all_data = all_data.drop_duplicates()

    # make fat. column into int type
    all_data["fat."] = all_data["fat."].apply(process_fat_column)

    # sum    fat. columm
    all_fatalities = all_data["fat."].sum()

    # make acc. date into datetime type 
    # Clean and convert the "acc. date" column to datetime
    all_data["acc. date"] = all_data["acc. date"].apply(clean_date_column)

    # Drop rows where "acc. date" could not be converted
    all_data = all_data.dropna(subset=["acc. date"])
    
    # Save the cleaned data to a new CSV file
    all_data.to_csv("data/all_accidents.csv", index=False)