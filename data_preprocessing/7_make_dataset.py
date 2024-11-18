import numpy as np
import pandas as pd
import os


def find_region(x, y, region_size=1024):
    # Calculate the region index for point (x, y)
    region_x = x // region_size
    region_y = y // region_size
    return region_x, region_y


root_dir = './ST_Breast/patches_csv/patches_224'
for sample_dir in os.listdir(root_dir):
    img_file_dir = f'./ST_Breast/patches_csv/patches_224/{sample_dir}'
    gene_expression_dir = f'./ST_Breast/gene_expression/224/{sample_dir}'
    feature_1024_path = f'./extracted_feature/1024/{sample_dir}'
    feature_512_path = f'./ST_Breast/extracted_feature/512/{sample_dir}'
    gene_expression_dir_1024 = f'./ST_Breast/gene_expression/1024/{sample_dir}'
    gene_expression_dir_512 = f'./ST_Breast/gene_expression/512/{sample_dir}'
    output_path = f'./ST_Breast/npy_information/{sample_dir}'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for label_file in os.listdir(gene_expression_dir):
        img_name = label_file[:-4]
        tmp_label_file_224 = pd.read_csv(os.path.join(gene_expression_dir, label_file))
        position_file_224 = pd.read_csv(os.path.join(img_file_dir, label_file))
        tmp_feature_1024 = np.load(os.path.join(feature_1024_path, f'{img_name}.npy'), allow_pickle=True).item()
        tmp_feature_512 = np.load(os.path.join(feature_512_path, f'{img_name}.npy'), allow_pickle=True).item()
        tmp_label_1024 = pd.read_csv(os.path.join(gene_expression_dir_1024, f'{img_name}_patches_1024.csv'))
        tmp_label_512 = pd.read_csv(os.path.join(gene_expression_dir_512, f'{img_name}_patches_512.csv'))

        information_list = []
        for spot in tmp_label_file_224.columns[1:]:
            information_dict = {}
            img_path = f'cropped_imgs/patch_224/{img_name}/{spot}.png'

            label = np.array(tmp_label_file_224[spot])

            # position = [int(position_file_224.loc[position_file_224.iloc[:, 0] == spot]['x']),
            #             int(position_file_224.loc[position_file_224.iloc[:, 0] == spot]['y'])]

            x_series = position_file_224.loc[position_file_224.iloc[:, 0] == spot]['i']
            y_series = position_file_224.loc[position_file_224.iloc[:, 0] == spot]['j']

            # Check if Series is empty
            if not x_series.empty and not y_series.empty:
                try:
                    position = [int(x_series.iloc[0]), int(y_series.iloc[0])]
                except (ValueError, TypeError) as e:
                    print(f"Skipping due to conversion error: {e}")
                    # You can choose to log this issue or continue execution
                    continue
            else:
                print(f"Skipping because x or y Series is empty for spot {spot}")
                continue

            true_position = [int(position_file_224.loc[position_file_224.iloc[:, 0] == spot]['X']),
                             int(position_file_224.loc[position_file_224.iloc[:, 0] == spot]['Y'])]
            pair_1024 = find_region(true_position[0], true_position[1])
            pair_512 = find_region(true_position[0], true_position[1], region_size=512)

            pair_1024_img_name = f'{img_name}_size_1024_{int(pair_1024[0]*1024)}_{int(pair_1024[1]*1024)}.png'
            pair_512_img_name = f'{img_name}_size_512_{int(pair_512[0] * 512)}_{int(pair_512[1] * 512)}.png'

            if pair_1024_img_name in tmp_label_1024.columns:
                label_1024 = tmp_label_1024[pair_1024_img_name]
                feature_1024 = tmp_feature_1024[pair_1024_img_name[:-4]]
            else:
                pair_1024_img_name = f'{img_name}_size_1024_{max(2048, int(pair_1024[0]*1024-1024))}_{max(2048, int(pair_1024[1]*1024-1024))}.png'
                label_1024 = tmp_label_1024[pair_1024_img_name]
                feature_1024 = tmp_feature_1024[pair_1024_img_name[:-4]]

            if pair_512_img_name in tmp_label_512.columns:
                label_512 = tmp_label_512[pair_512_img_name]
                feature_512 = tmp_feature_512[pair_512_img_name[:-4]]
            else:
                pair_512_img_name = f'{img_name}_size_512_{max(512, int(pair_512[0] * 512-512))}_{max(512, int(pair_512[1] * 512-512))}.png'
                label_512 = tmp_label_512[pair_512_img_name]
                feature_512 = tmp_feature_512[pair_512_img_name[:-4]]

            information_dict['img_path'] = img_path
            information_dict['label'] = label
            information_dict['position'] = position
            information_dict['feature_1024'] = feature_1024
            information_dict['feature_512'] = feature_512
            information_dict['label_1024'] = np.array(label_1024)
            information_dict['label_512'] = np.array(label_512)

            information_list.append(information_dict)

        save_path = os.path.join(output_path, f'{img_name}.npy')
        print(save_path)
        print(len(information_list))
        np.save(save_path, information_list)
