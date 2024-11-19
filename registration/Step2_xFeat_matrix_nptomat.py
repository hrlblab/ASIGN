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
import matplotlib


import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import glob

import re
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


if __name__ == '__main__':
    case_folder = './Human_dorsolateral'

    cases = glob.glob(os.path.join(case_folder, '*'))
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        sg_folder = os.path.join(case, 'registration_xfeat')
        ANTs_folder = os.path.join(case, 'registration_ANTs')

        images = glob.glob(os.path.join(image_input_dir, '*.png'))
        images.sort(key=natural_keys)

        for ii in range(len(images)-1):
            moving_root = images[ii+1]
            moving_output_root = moving_root.replace(image_input_dir, sg_folder)
            displacement_root = moving_output_root.replace('.png', '.npy')

            fixed_root = images[ii]
            print('now case is %s to %s.' % (os.path.basename(moving_root), os.path.basename(fixed_root)))

            img_highres = np.array(Image.open(fixed_root))

            overlay_dir = os.path.join(sg_folder,'%s_to_%s' % (os.path.basename(moving_root).split('.')[0],os.path.basename(fixed_root).split('.')[0]))
            affine_root = os.path.join(overlay_dir, 'sg_affine_init.npy')

            affine = np.load(affine_root)
            M_inv = cv2.invertAffineTransform(affine)

            M_vector = np.zeros([6])

            # world coordinate to itk
            FixParameters = [img_highres.shape[1] / 2.0, img_highres.shape[0] / 2.0]
            M_vector[0] = M_inv[0, 0]
            M_vector[1] = M_inv[0, 1]
            M_vector[2] = M_inv[1, 0]
            M_vector[3] = M_inv[1, 1]
            M_vector[4] = (FixParameters[0] * M_inv[0, 0] + FixParameters[1] * M_inv[0, 1] - FixParameters[0]) + M_inv[
                0, 2]
            M_vector[5] = (FixParameters[0] * M_inv[1, 0] + FixParameters[1] * M_inv[1, 1] - FixParameters[1]) + M_inv[
                1, 2]

            m = sitk.ReadTransform('test.mat')

            m.SetParameters(M_vector)
            m.SetFixedParameters(FixParameters)

            save_root = os.path.dirname(affine_root.replace(sg_folder, ANTs_folder))
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            sitk.WriteTransform(m, os.path.join(save_root, 'sg_affine_init.mat'))

