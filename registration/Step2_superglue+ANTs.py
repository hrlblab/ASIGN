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
from PIL import Image
import matplotlib.pyplot as plt
import torch as th
import glob
import numpy as np
import math

from skimage.transform import resize

sys.path.append('./ST_registration')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import matplotlib.pyplot as plt
import glob
from superglue_ants_registration_onpair_new_RGB import register_a_pair
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
    case_folder = './Human_dorsolateral/human'

    cases = glob.glob(os.path.join(case_folder, '*'))
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        ANTs_folder = os.path.join(case, 'registration_ANTs')

        images = glob.glob(os.path.join(image_input_dir, '*.png'))
        images.sort(key=natural_keys)

        for ii in range(len(images)-1):
            moving_root = images[ii+1]
            fixed_root = images[ii]
            print('now case is %s to %s.' % (os.path.basename(moving_root), os.path.basename(fixed_root)))

            # img_highres = np.array(Image.open(fixed_root))
            img_highres = plt.imread(fixed_root)[:, :,3]

            working_dir = os.path.join(ANTs_folder,'%s_to_%s' % (os.path.basename(moving_root).split('.')[0],os.path.basename(fixed_root).split('.')[0]))
            if os.path.exists(os.path.join(working_dir, 'step2_run_ants_reg', 'output0GenericAffine.mat')):
                continue

            register_a_pair(moving_root, fixed_root, working_dir)
