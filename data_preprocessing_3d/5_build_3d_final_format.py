import shutil
import os
import numpy as np


"""
Step 5: final step to get combined sample level information
Input: npy files of each WSI
Output: npy file for samples, add layer number to each spot

"""
file_dir = './Human_dorsolateral_3D/npy_infor'
save_dir = './Human_dorsolateral_3D/3D_npy_information'


for sample in os.listdir(file_dir):
    new_sample = []

    for file in os.listdir(os.path.join(file_dir, sample)):
        samples = [int(a[:-4]) for a in os.listdir(os.path.join(file_dir, sample))]
        level = min(samples)
        tmp_level = 1+int(file[:-4])-level
        tmp_file = np.load(os.path.join(file_dir, sample, file), allow_pickle=True)

        for i in range(tmp_file.shape[0]):
            tmp_spot = tmp_file[i]
            tmp_spot['level'] = tmp_level
            new_sample.append(tmp_spot)

    np.save(os.path.join(save_dir, f'{sample}.npy'), new_sample)
    print(os.path.join(save_dir, f'{sample}.npy'))




