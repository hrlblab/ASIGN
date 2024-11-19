# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
import sys
import os
import time

import matplotlib.pyplot as plt
import torch as th
import glob
import numpy as np
import math

from skimage.transform import resize

# sys.path.append('/Data/3D_Vessel/airlab')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2 as cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import glob

import re
from skimage.transform import warp

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text) ]


if __name__ == '__main__':
    case_folder = './Human_dorsolateral'

    cases = glob.glob(os.path.join(case_folder, '*'))
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        image_output_dir = os.path.join(case, 'registration_ANTs')
        # image_badcase_dir = os.path.join(case, 'registration_badcase_2048')
        reconstruction_dir = os.path.join(case, 'registration_3D_ANTs')

        if not os.path.exists(reconstruction_dir):
            os.makedirs(reconstruction_dir)

        images = glob.glob(os.path.join(image_input_dir, '*.png'))
        images.sort(key=natural_keys)

        middle_idx = int(len(images) / 2)

        print(middle_idx)
        middle_image = plt.imread(images[middle_idx])[:,:,:3]

        for ii in range(len(images)):
            now_idx = ii

            if now_idx < middle_idx:
                M_new = np.zeros((3, 3))
                M_new[2, 2] = 1.

                for ri in range(now_idx, middle_idx):
                    print(os.path.basename(images[ri+1]))

                    moving_index = int(os.path.basename(images[ri+1]).split('.')[0])
                    fixed_index = int(os.path.basename(images[ri]).split('.')[0])
                    # moving_index = int(os.path.basename(images[ri+1])[0])
                    # fixed_index = int(os.path.basename(images[ri])[0])

                    affine_root = os.path.join(image_output_dir, '%d_to_%d' % (moving_index, fixed_index), 'step2_run_ants_reg', 'output0GenericAffine.mat')
                    m = sitk.ReadTransform(affine_root)
                    FixP = m.GetFixedParameters()
                    M_vec = m.GetParameters()
                    M_inv = np.zeros((2, 3))

                    M_inv[0, 0] = M_vec[0]
                    M_inv[0, 1] = M_vec[1]
                    M_inv[1, 0] = M_vec[2]
                    M_inv[1, 1] = M_vec[3]
                    M_inv[0, 2] = M_vec[4] - (FixP[0] * M_inv[0, 0] + FixP[1] * M_inv[0, 1] - FixP[0])
                    M_inv[1, 2] = M_vec[5] - (FixP[0] * M_inv[1, 0] + FixP[1] * M_inv[1, 1] - FixP[1])

                    M1 = np.zeros((3, 3))
                    M1[2, 2] = 1.
                    M1[:2, :] = cv2.invertAffineTransform(M_inv)

                    if ri == now_idx:
                        M_new = M1.copy()
                    else:
                        M_new = M_new.dot(M1)

                now_image = plt.imread(images[now_idx])[:,:,:3]
                img1_affine = warp(now_image, M_new, output_shape=(middle_image.shape[0], middle_image.shape[1]))
                new_root = images[now_idx].replace(image_input_dir, reconstruction_dir)
                plt.imsave(new_root, img1_affine)

            elif now_idx > middle_idx:
                M_new = np.zeros((3, 3))
                M_new[2, 2] = 1.

                for ri in range(middle_idx, now_idx):

                    moving_index = int(os.path.basename(images[ri+1]).split('.')[0])
                    fixed_index = int(os.path.basename(images[ri]).split('.')[0])

                    # moving_index = int(os.path.basename(images[ri+1])[0])
                    # fixed_index = int(os.path.basename(images[ri])[0])

                    affine_root = os.path.join(image_output_dir, '%d_to_%d' % (moving_index, fixed_index), 'step2_run_ants_reg', 'output0GenericAffine.mat')

                    m = sitk.ReadTransform(affine_root)
                    FixP = m.GetFixedParameters()
                    M_vec = m.GetParameters()
                    M_inv = np.zeros((2, 3))

                    M_inv[0, 0] = M_vec[0]
                    M_inv[0, 1] = M_vec[1]
                    M_inv[1, 0] = M_vec[2]
                    M_inv[1, 1] = M_vec[3]
                    M_inv[0, 2] = M_vec[4] - (FixP[0] * M_inv[0, 0] + FixP[1] * M_inv[0, 1] - FixP[0])
                    M_inv[1, 2] = M_vec[5] - (FixP[0] * M_inv[1, 0] + FixP[1] * M_inv[1, 1] - FixP[1])

                    M1 = np.zeros((3, 3))
                    M1[2, 2] = 1.
                    M1[:2, :] = cv2.invertAffineTransform(M_inv)

                    if ri == middle_idx:
                        M_new = M1.copy()
                    else:
                        M_new = M_new.dot(M1)

                affine_matrix_inv = cv2.invertAffineTransform(M_new[:2, :])
                M_new[:2, :] = affine_matrix_inv

                now_image = plt.imread(images[now_idx])[:, :, :3]
                img1_affine = warp(now_image, M_new, output_shape=(middle_image.shape[0], middle_image.shape[1]))
                new_root = images[now_idx].replace(image_input_dir, reconstruction_dir)
                plt.imsave(new_root, img1_affine)

            else:
                new_root = images[middle_idx].replace(image_input_dir, reconstruction_dir)
                plt.imsave(new_root, middle_image)