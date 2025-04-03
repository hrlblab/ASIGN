import os
import pandas as pd
import numpy as np
from tqdm import tqdm

"""
Step 6.3: gene expression normalization.
Input: Selected raw genes (csv file)
Output: Normalized gene expression (csv_file)

Note: We follow the data preprocessing step proposed by ST-Net (He et.al. Nature Nature Biomedical Engineering (2020)):
'We pre-processed the gene counts with two transformations: 
First, we normalized the total expression in each spot after adding a pseudo count of one.
Second, we took the log value of the normalized counts to bring the values into a reasonable range.'

We apply same strategy for all 3 resolutions.
"""

# Define the input and output directories for CSV files
input_directory = './raw_data/ST_Breast/gene_csv'
output_directory = './raw_data/ST_Breast/gene_expression_st'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Iterate through all CSV files in the directory
for filename in tqdm(os.listdir(input_directory)):
    if filename.endswith('.csv'):
        # Read the CSV file
        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path, index_col=0)

        # Process each column: add 1 to each element, divide by the column sum, then take the logarithm.
        # Note: each column represents a spatial spot.
        df = df.add(1)  # Add 1 to each element
        df = df.div(df.sum(axis=0), axis=1)  # Normalize each column by its sum
        df = np.log(df)  # Take the natural logarithm

        # Save the result to a new CSV file
        output_file_path = os.path.join(output_directory, filename)
        df.to_csv(output_file_path)

        print(f"File '{filename}' has been processed and saved to '{output_file_path}'")


