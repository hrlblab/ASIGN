import os
import pandas as pd

"""
Step 6.2: calculate the gene expression for region/global level
Input: pair information for spot and region or region and global level (csv file, information got from step 3 and 4)
       selected genes of spot level
    
Output: selected gene expression of region/global level (saved as csv file)

Note: please modify this code to get the two resolution results respectively. 
This example is to get global-level label from region level.
"""


def process_csv_files(csv_file_path, label_file_path, output_file_path):
    """
    Process two CSV files. Based on the 'points' (or 'overlapping_512_patches') column
    in the first file, compute and extract data from the second file, and generate a new CSV.

    Parameters:
    - csv_file_path (str): Path to the first CSV file, containing 'points' and 'patch_filename' columns.
    - label_file_path (str): Path to the second CSV file, where column names match the 'points'.
    - output_file_path (str): Path to the output CSV file (default: 'new_file.csv').
    """

    # 1. Read the CSV files
    df = pd.read_csv(csv_file_path)
    label_df = pd.read_csv(label_file_path)

    # 2. Create an empty DataFrame to store the results
    df_new = pd.DataFrame()

    # 3. Iterate over each row in the 'overlapping_512_patches' column
    for i in range(len(df['overlapping_512_patches'])):
        label = float()
        for j in df['overlapping_512_patches'][i].split(', '):
            if j in label_df.columns:
                print(j)
                label += label_df[j]
        df_new[df['patch_filename'][i]] = label

    # 4. Save the result as a new CSV file
    df_new.to_csv(output_file_path, index=False) 

    print(f"New CSV file created: {output_file_path}")


root_dir = './ST_Breast/gene_expression/512'

for sample in os.listdir(root_dir):
    csv_file_dir = f'./ST_Breast/patches_csv/patches_1024/{sample}'
    label_file_dir = f'./ST_Breast/gene_expression/512/{sample}'
    save_dir = f'./ST_Breast/gene_expression/1024/{sample}'

    # Create output directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for csv_file in os.listdir(csv_file_dir):
        print(csv_file)
        csv_file_p = os.path.join(csv_file_dir, csv_file)
        label_file_p = os.path.join(label_file_dir, '{}512.csv'.format(csv_file[:-8]))
        output_file_p = os.path.join(save_dir, csv_file)
        process_csv_files(csv_file_p, label_file_p, output_file_p)
