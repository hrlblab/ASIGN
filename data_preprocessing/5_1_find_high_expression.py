import os
import pandas as pd
from tqdm import tqdm

"""
Step 6.1: find genes of top 250 highest expressions
Input: Dir of original ST data information (csv/tsv files)
Output: Select highest gene expressions, saved in csv file.

Note: please modify to fit your data format
"""


# Define the input CSV directory and the output directory
input_directory = './raw_data/ST_Breast/stdata'
output_directory = './raw_data/ST_Breast/gene_expression'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Initialize a set to store the row names (gene names) common to all files
common_genes = None

# Initialize an empty DataFrame to store mean values from all files
all_means = pd.DataFrame()

# Step 1: Find the gene names shared across all files
for filename in tqdm(os.listdir(input_directory)):
    if filename.endswith('.tsv'):
        # Read the CSV file
        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path, index_col=0, sep='\t').transpose()

        # If this is the first file, initialize common_genes with its row names
        if common_genes is None:
            common_genes = set(df.index)
        else:
            # Otherwise, take the intersection with current file's row names
            common_genes &= set(df.index)

# Step 2: For each file, calculate the mean expression of common genes
# and keep the top 250 genes with the highest average expression
for filename in tqdm(os.listdir(input_directory)):
    if filename.endswith('.tsv'):
        # Read the CSV file
        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path, index_col=0, sep='\t').transpose()

        # Keep only the rows in common_genes
        df = df.loc[list(common_genes)]

        # Calculate the mean for each row (gene)
        row_means = df.mean(axis=1)

        # Append the row means to the all_means DataFrame
        all_means = pd.concat([all_means, row_means], axis=1)

# Calculate the overall mean expression across all files
overall_means = all_means.mean(axis=1)

# Get the row names of the top 250 genes with the highest mean expression
top_250_row_names = overall_means.nlargest(250).index.tolist()

# Step 3: For each CSV file, retain only these 250 rows and save to a new CSV
for filename in tqdm(os.listdir(input_directory)):
    if filename.endswith('.tsv'):
        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path, index_col=0, sep='\t').transpose()

        # Keep only the rows in top_250_row_names
        filtered_df = df.loc[top_250_row_names]

        # Save to the new CSV file
        output_file_path = os.path.join(output_directory, filename)
        filtered_df.to_csv(output_file_path)

        print(f"File '{filename}' has been processed and saved to '{output_file_path}'")
