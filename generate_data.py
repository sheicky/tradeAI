# This script is used to generate data for the project
import numpy as np
import pandas as pd


def generate_similar_values(original_values):
    """Generate values similar to original ones with some random variation"""
    mean = np.mean(original_values)
    std = np.std(original_values)
    # Generate new values with similar distribution but some randomness
    new_values = np.random.normal(mean, std, len(original_values))
    # Ensure values stay within reasonable bounds (±3 standard deviations)
    new_values = np.clip(new_values, mean - 3 * std, mean + 3 * std)
    return new_values


def data_generator(data):
    """Generate new dataset while keeping dates and column structure"""
    # Create a copy of original data
    new_data = pd.DataFrame()

    # Keep original dates
    new_data["Data"] = data["Data"]

    # Generate new values for each numeric column while preserving the original structure
    for column in data.columns:
        if column != "Data":  # Skip the date column
            original_values = data[column].values
            new_values = generate_similar_values(original_values)
            new_data[column] = new_values

            # For binary columns like 'Y', round to 0 or 1
            if column == "Y":
                new_data[column] = np.round(new_values).astype(int)

    return new_data


if __name__ == "__main__":
    # Read original dataset
    original_data = pd.read_csv("market_dataset.csv")

    # Generate new dataset
    generated_data = data_generator(original_data)

    # Save to new file
    generated_data.to_csv("generated_dataset.csv", index=False)
    print("New dataset generated and saved as 'generated_dataset.csv'")
