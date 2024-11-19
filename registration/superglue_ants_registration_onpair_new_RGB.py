import os
import pandas as pd
import numpy as np
import cv2
# from SURF_robust_match import surf_affine
# from Superglue_robust_match import surf_affine
import nibabel as nib
from sklearn.metrics import mutual_info_score, mean_squared_error
from numpy import corrcoef
import csv
from PIL import Image
# based on https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images,-pointsets-and-Jacobians
# and
import matplotlib.pyplot as plt


def count_score(original_fname, backward_fname):
    img1 = nib.load(original_fname)
    img2 = nib.load(backward_fname)
    # print('shape')
    # print(img1.shape)
    # print(img2.shape)

    img1 = np.array(img1.dataobj).reshape(-1)
    img2 = np.array(img2.dataobj).reshape(-1)
    # img1 = cv2.resize(img1, (img1.shape[0]*img1.shape[1]))
    # img2 = cv2.resize(img2, (img1.shape[0]*img1.shape[1]))

    # print(img1.shape)
    # print(img2.shape)
    mutual_info = mutual_info_score(img1, img2)
    mse = mean_squared_error(img1, img2)
    cross_cor = corrcoef(img1, img2)

    print(mutual_info)
    print(mse)
    print(cross_cor)

    return mutual_info, mse, cross_cor


def write_bbox_to_csv(df, fixed_bbox_csv):
    if os.path.exists(fixed_bbox_csv):
        return
    df.to_csv(fixed_bbox_csv, index=False)


def show_bounding_box(fixed_jpg, fixed_bbox_csv, output_img, seg_nii=None, image_FOV='local', t=None):
    csf_data = pd.read_csv(fixed_bbox_csv)
    if t != None:
        csf_data = csf_data[csf_data['t'] == t]
    csf_size = csf_data.shape[0]
    bbox_num = int(csf_size / 4)

    cimg = cv2.imread(fixed_jpg)
    if not csf_data['x'].dtype == 'int':
        csf_data = csf_data.round().astype(np.int)

    for bi in range(bbox_num):
        row_start_ind = bi * 4
        cnt = np.zeros((4, 1, 2))
        x_list = csf_data['x'].to_list()
        y_list = csf_data['y'].to_list()
        cnt[0, 0, 0] = int(x_list[row_start_ind])
        cnt[0, 0, 1] = int(y_list[row_start_ind])
        cnt[1, 0, 0] = int(x_list[row_start_ind + 1])
        cnt[1, 0, 1] = int(y_list[row_start_ind + 1])
        cnt[2, 0, 0] = int(x_list[row_start_ind + 2])
        cnt[2, 0, 1] = int(y_list[row_start_ind + 2])
        cnt[3, 0, 0] = int(x_list[row_start_ind + 3])
        cnt[3, 0, 1] = int(y_list[row_start_ind + 3])

        cv2.drawContours(cimg, [cnt.astype(int)], -1, (0, 255, 0), 3)

        cropped_x_min = np.max([np.min(x_list[row_start_ind:row_start_ind + 3]) - 100, 0])
        cropped_x_max = np.min([np.max(x_list[row_start_ind:row_start_ind + 3]) + 100, cimg.shape[1]])
        cropped_y_min = np.max([np.min(y_list[row_start_ind:row_start_ind + 3]) - 100, 0])
        cropped_y_max = np.min([np.max(y_list[row_start_ind:row_start_ind + 3]) + 100, cimg.shape[0]])

        # cropped_x_min = csf_data['x'].min() - 100
        # cropped_x_max = csf_data['x'].max() + 100
        # cropped_y_min = csf_data['y'].min() - 100
        # cropped_y_max = csf_data['y'].max() + 100

        if seg_nii != None and os.path.exists(seg_nii):
            nii = nib.load(seg_nii)
            mask = nii.get_data()
            try:

                if len(mask.shape) > 2:
                    mask = mask[:, :, 0]
            except:
                aaa = 1
            mask = mask * 255
            mask = np.rot90(mask, -1)
            backtorgb = np.clip(mask, 0, 255)
            backtorgb = np.array(backtorgb, np.uint8)
            img = cv2.merge([backtorgb, backtorgb, backtorgb])
            # img = cv2.flip(img, 0, dst=None)
            img = cv2.flip(img, 1, dst=None)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find contours:
            (thresh, im_bw) = cv2.threshold(img, 50, 255, 0)
            im, contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(cimg, contours, -1, (0, 0, 255), 3)

        cimg_crop = cimg[cropped_y_min:cropped_y_max, cropped_x_min:cropped_x_max, :]

        if bi == 0:
            output_img_print = output_img
        else:
            output_img_print = output_img.replace('.jpg', '_%d.jpg' % bi)

        if image_FOV == 'local':
            # Image.fromarray(cimg_crop).show()
            cv2.imwrite(output_img_print, cimg_crop)
        elif image_FOV == 'global':
            cv2.imwrite(output_img_print, cimg)


def register_a_pair(moving_jpg: object, fixed_jpg: object, working_dir: object) -> object:
    ants_bin_dir = './ANTS/install/bin'
    os.environ["ANTSPATH"] = ants_bin_dir

    output_name = 'output'

    # step 1 ============================================================================================================
    step_1_dir = os.path.join(working_dir, 'step1_prepare_data')
    if not os.path.exists(step_1_dir):
        os.makedirs(step_1_dir)

    # #copy raw data
    moving_fname = os.path.basename(moving_jpg)
    fix_fname = os.path.basename(fixed_jpg)
    moving_jpg_file = os.path.join(step_1_dir, moving_fname)
    fix_jpg_file = os.path.join(step_1_dir, fix_fname)
    print(moving_jpg, fixed_jpg)

    if not os.path.exists(moving_jpg_file):
        os.system("cp %s %s" % (moving_jpg, moving_jpg_file))

    if not os.path.exists(fix_jpg_file):
        os.system("cp %s %s" % (fixed_jpg, fix_jpg_file))

    # convert jpg to nifti
    ConvertCMD = os.path.join(ants_bin_dir, 'ConvertImagePixelType')
    moving_nii = os.path.join(step_1_dir, 'moving.nii.gz')  #########   moving image
    fix_nii = os.path.join(step_1_dir, 'fix.nii.gz')  #########   fixed image
    if not os.path.exists(moving_nii):
        os.system("%s %s %s 1" % (ConvertCMD, moving_jpg, moving_nii))
    if not os.path.exists(fix_nii):
        os.system("%s %s %s 1" % (ConvertCMD, fixed_jpg, fix_nii))


    sg_affine_file = os.path.join(working_dir, 'sg_affine_init.mat')

    # antsApplyTransformsCMD = os.path.join(ants_bin_dir, 'antsApplyTransforms')
    #
    # pwd_dir = os.getcwd()
    # os.chdir(surf_affine_dir)
    # os.system("%s -d 2 -i %s -r %s -t %s -o %s " % (antsApplyTransformsCMD, moving_nii, fix_nii, sg_affine_file, affine_init_file))
    # os.chdir(pwd_dir)

    # step 2 ============================================================================================================
    final_fname = '%sWarped.nii.gz' % (output_name)
    invfinal_fname = '%sInverseWarped.nii.gz' % (output_name)
    warp_fname = '%s1Warp.nii.gz' % (output_name)
    invwarp_fname = '%s1InverseWarp.nii.gz' % (output_name)
    aff_mtx_fname = '%s0GenericAffine.mat' % (output_name)

    pwd_dir = os.getcwd()
    step_2_dir = os.path.join(working_dir, 'step2_run_ants_reg')
    if not os.path.exists(step_2_dir):
        os.makedirs(step_2_dir)
    antsRegistrationSyNCMD = os.path.join(ants_bin_dir, 'antsRegistrationSyN.sh')
    output_file = os.path.join(step_2_dir, output_name)  ########The different between two files
    final_file = os.path.join(step_2_dir, final_fname)

    print(moving_nii)
    print(fix_nii)
    print(sg_affine_file)
    print(final_file)
    os.chdir(step_2_dir)

    os.system("%s -d 2 -m %s -f %s -j 1 -t s -n 8 -i %s - o %s " % (
    antsRegistrationSyNCMD, moving_nii, fix_nii, sg_affine_file, output_file))
    os.chdir(pwd_dir)


def register_a_pair_intensitychange(moving_jpg: object, fixed_jpg: object, working_dir: object,
                                    resolution: object) -> object:
    ants_bin_dir = './ANTS/install/bin'
    os.environ["ANTSPATH"] = ants_bin_dir

    output_name = 'output'

    # step 1 ============================================================================================================
    step_1_dir = os.path.join(working_dir, 'step1_prepare_data')
    if not os.path.exists(step_1_dir):
        os.makedirs(step_1_dir)

    # #copy raw data
    moving_fname = os.path.basename(moving_jpg)
    fix_fname = os.path.basename(fixed_jpg)
    moving_jpg_file = os.path.join(step_1_dir, moving_fname)
    fix_jpg_file = os.path.join(step_1_dir, fix_fname)
    print(moving_jpg, fixed_jpg)
    # if not os.path.exists(moving_jpg_file):
    #     os.system("cp %s %s" % (moving_jpg, moving_jpg_file))

    # reverse the protein image:
    # if not os.path.exists(moving_jpg_file):
    #    moving_img = 1 - plt.imread(moving_jpg)[:,:,:3]
    #    plt.imsave(moving_jpg_file, moving_img)

    # power the intensity of protein:
    if not os.path.exists(moving_jpg_file):
        moving_img = plt.imread(moving_jpg)[:, :, :3]
        # moving_img[:, :, 0] = moving_img[:, :, 2]
        # moving_img[:, :, 1] = moving_img[:, :, 2]

        # img2 = 255 - ((255 - cv2.imread(img2_file, 0)) ** 2 / 2).astype(np.uint8)
        moving_img = 1 - (1 - moving_img) ** 2
        moving_img[moving_img > 1.] = 1.
        moving_img[moving_img < 0.] = 0.

        # moving_img[:,:,2] = moving_img[:,:,0]
        plt.imsave(moving_jpg_file, moving_img)

    # if not os.path.exists(moving_jpg_file):
    #     os.system("cp %s %s" % (moving_jpg, moving_jpg_file))

    if not os.path.exists(fix_jpg_file):
        os.system("cp %s %s" % (fixed_jpg, fix_jpg_file))

    # convert jpg to nifti
    ConvertCMD = os.path.join(ants_bin_dir, 'ConvertImagePixelType')
    moving_nii = os.path.join(step_1_dir, 'moving.nii.gz')  #########   moving image
    fix_nii = os.path.join(step_1_dir, 'fix.nii.gz')  #########   fixed image
    if not os.path.exists(moving_nii):
        os.system("%s %s %s 1" % (ConvertCMD, moving_jpg, moving_nii))
    if not os.path.exists(fix_nii):
        os.system("%s %s %s 1" % (ConvertCMD, fixed_jpg, fix_nii))

    if resolution == 5:
        sg_affine_file = os.path.join(working_dir, 'sg_affine_init_5X.mat')
    elif resolution == 10:
        sg_affine_file = os.path.join(working_dir, 'sg_affine_init_10X.mat')
    else:
        sg_affine_file = os.path.join(working_dir, 'sg_affine_init_20X.mat')

    # affine_init_file = os.path.join(working_dir, 'sg_affine_file.nii.gz')
    #
    #
    # antsApplyTransformsCMD = os.path.join(ants_bin_dir, 'antsApplyTransforms')
    #
    # pwd_dir = os.getcwd()
    # os.chdir(surf_affine_dir)
    # os.system("%s -d 2 -i %s -r %s -t %s -o %s " % (antsApplyTransformsCMD, moving_nii, fix_nii, sg_affine_file, affine_init_file))
    # os.chdir(pwd_dir)

    # step 2 ============================================================================================================
    final_fname = '%sWarped.nii.gz' % (output_name)
    invfinal_fname = '%sInverseWarped.nii.gz' % (output_name)
    warp_fname = '%s1Warp.nii.gz' % (output_name)
    invwarp_fname = '%s1InverseWarp.nii.gz' % (output_name)
    aff_mtx_fname = '%s0GenericAffine.mat' % (output_name)

    pwd_dir = os.getcwd()
    step_2_dir = os.path.join(working_dir, 'step2_run_ants_reg')
    if not os.path.exists(step_2_dir):
        os.makedirs(step_2_dir)
    antsRegistrationSyNCMD = os.path.join(ants_bin_dir, 'antsRegistrationSyN.sh')
    output_file = os.path.join(step_2_dir, output_name)  ########The different between two files
    final_file = os.path.join(step_2_dir, final_fname)

    print(moving_nii)
    print(fix_nii)
    print(sg_affine_file)
    print(final_file)
    os.chdir(step_2_dir)

    os.system("%s -d 2 -m %s -f %s -j 1 -t s -n 8 -i %s - o %s " % (
        antsRegistrationSyNCMD, moving_nii, fix_nii, sg_affine_file, output_file))
    os.chdir(pwd_dir)


def register_3D(original_image_index: object, middle_image_index: object, slice_roi_list: object,
                image_output_dir: object):
    ants_bin_dir = '/home/dengr/Tools/ANTS/install/bin'
    os.environ["ANTSPATH"] = ants_bin_dir

    original_name = 'output'
    output_name = '3Doutput'
    pwd_dir = os.getcwd()
    output_dir = os.path.join(image_output_dir, str(slice_roi_list[original_image_index]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    antsApplyTransformsCMD = os.path.join(ants_bin_dir, 'antsApplyTransforms')
    antsApplyTransformsToPointsCMD = os.path.join(ants_bin_dir, 'antsApplyTransformsToPoints')

    moving_roi_curr = slice_roi_list[middle_image_index - 1]
    fixed_roi_curr = slice_roi_list[middle_image_index]
    destination_dir = os.path.join(image_output_dir, '%d-to-%d' % (moving_roi_curr, fixed_roi_curr))
    destination_3D_folder = os.path.join(destination_dir, 'step1_prepare_data')
    destination_3D_file = os.path.join(destination_3D_folder, 'fix.nii.gz')  # tranfer the image to this nii
    destination_3D_csv = os.path.join(destination_3D_folder, 'fixed_bbox_coordinates.csv')

    if original_image_index < middle_image_index:
        for bi in range(len(slice_roi_list[original_image_index:middle_image_index])):
            moving_roi_curr = slice_roi_list[original_image_index + bi]
            fixed_roi_curr = slice_roi_list[original_image_index + bi + 1]
            # print('yes')
            # print(slice_roi_list[original_image_index])
            # print(slice_roi_list[middle_image_index])
            # print(moving_roi_curr)
            # print(fixed_roi_curr)
            working_dir = os.path.join(image_output_dir, '%d-to-%d' % (moving_roi_curr, fixed_roi_curr))
            if moving_roi_curr == slice_roi_list[original_image_index]:

                original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                original_csv_folder = os.path.join(working_dir, 'step3_inv_reg_points')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

                # print(original_3D_file)
                # print(output_3D_file)
                if not os.path.exists(output_3D_file):
                    os.system("cp %s %s" % (original_3D_file, output_3D_file))
                if not os.path.exists(output_csv_file):
                    os.system("cp %s %s" % (original_csv_file, output_csv_file))
                # img_nii = nib.load(output_3D_file)
                # img = img_nii.get_fdata()
                # print(img.shape)
                print(original_csv_file)
                print(output_csv_file)

            else:
                original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                original_csv_folder = os.path.join(working_dir, 'step3_inv_reg_points')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

                begin_image_folder = os.path.join(working_dir, 'step1_prepare_data')
                begin_image = os.path.join(begin_image_folder, 'fix.nii.gz')

                matrix_dir = os.path.join(working_dir, 'step2_run_ants_reg')
                # sift_affine_dir = os.path.join(begin_image_folder, 'sift_affine')
                surf_affine_file = os.path.join(matrix_dir, 'output0GenericAffine.mat')
                # output_affine_file = os.path.join(sift_affine_dir, 'sift_affine_init.mat')
                warp_file = os.path.join(matrix_dir, 'output1Warp.nii.gz')
                invwarp_file = os.path.join(matrix_dir, 'output1InverseWarp.nii.gz')

                os.chdir(output_dir)
                # os.system("%s -d 2 -i %s -r %s -t [%s, 1] -t %s -n GenericLabel -o %s" % (
                #    antsApplyTransformsCMD, output_3D_file, destination_3D_file, sift_affine_file, warp_file, output_3D_file))
                os.system("%s -d 2 -i %s -r %s -t %s -o %s " % (
                    # os.system("%s -d 2 -m %s -f %s -j 1 -t s -n 8 -i %s - o %s " % (
                    antsApplyTransformsCMD, output_3D_file, original_3D_file, surf_affine_file,
                    output_3D_file))  ##second one shoulf destination before
                # os.system("%s -d 2 -i %s -o %s -t %s -t [%s,1]" % (
                os.system("%s -d 2 -i %s -o %s -t %s -t [%s ,1]" % (
                    #    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, invwarp_file, sift_affine_file))
                    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, invwarp_file, surf_affine_file))
                # os.system("%s -d 2 -i %s -o %s -t %s -t %s" % (
                #    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, sift_affine_file, sift_affine_file))
                os.chdir(pwd_dir)
                # img_nii = nib.load(output_3D_file)
                # img = img_nii.get_fdata()
                # print(img.shape)
    elif original_image_index == middle_image_index:
        working_dir = os.path.join(image_output_dir, '%d-to-%d' % (moving_roi_curr, fixed_roi_curr))
        output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))
        original_csv_folder = os.path.join(working_dir, 'step3_inv_reg_points')
        original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
        output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')
        # if not os.path.exists(output_3D_file):
        os.system("cp %s %s" % (destination_3D_file, output_3D_file))
        os.system("cp %s %s" % (destination_3D_csv, output_csv_file))
        print(destination_3D_csv)
        print(output_csv_file)

    else:
        for bi in range(len(slice_roi_list[middle_image_index:original_image_index])):
            moving_roi_curr = slice_roi_list[original_image_index - bi]
            fixed_roi_curr = slice_roi_list[original_image_index - bi - 1]

            # print('yes')
            # print(slice_roi_list[original_image_index])
            # print(slice_roi_list[middle_image_index])
            # print(moving_roi_curr)
            # print(fixed_roi_curr)

            working_dir = os.path.join(image_output_dir, '%d-to-%d' % (moving_roi_curr, fixed_roi_curr))
            if moving_roi_curr == slice_roi_list[original_image_index]:

                original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                original_csv_folder = os.path.join(working_dir, 'step3_inv_reg_points')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

                # print(original_3D_file)
                # print(output_3D_file)
                if not os.path.exists(output_3D_file):
                    os.system("cp %s %s" % (original_3D_file, output_3D_file))
                if not os.path.exists(output_csv_file):
                    os.system("cp %s %s" % (original_csv_file, output_csv_file))

                print(original_csv_file)
                print(output_csv_file)

            else:
                original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                original_csv_folder = os.path.join(working_dir, 'step3_inv_reg_points')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

                begin_image_folder = os.path.join(working_dir, 'step1_prepare_data')
                begin_image = os.path.join(begin_image_folder, 'fix.nii.gz')

                matrix_dir = os.path.join(working_dir, 'step2_run_ants_reg')
                # sift_affine_dir = os.path.join(matrix_dir, 'sift_affine')
                surf_affine_file = os.path.join(matrix_dir, 'output0GenericAffine.mat')
                warp_file = os.path.join(matrix_dir, 'output1Warp.nii.gz')
                invwarp_file = os.path.join(matrix_dir, 'output1InverseWarp.nii.gz')
                # sift_affine_file = os.path.join(sift_affine_dir, 'sift_affine_init.mat')
                # print(original_3D_file)
                # print(output_3D_file)
                # print(sift_affine_file)
                # print(destination_3D_file)

                os.chdir(output_dir)
                # os.system("%s -d 2 -i %s -r %s -t [%s, 1] -t %s -n GenericLabel -o %s" % (
                #    antsApplyTransformsCMD,  output_3D_file, destination_3D_file, sift_affine_file, warp_file,
                #    output_3D_file))
                # os.system("%s -d 2 -m %s -f %s -j 1 -t s -n 8 -i %s - o %s " % (
                os.system("%s -d 2 -i %s -r %s -t %s -o %s " % (
                    antsApplyTransformsCMD, output_3D_file, original_3D_file, surf_affine_file, output_3D_file))
                # antsApplyTransformsCMD, output_3D_file, destination_3D_file, sift_affine_file, output_3D_file))  ##second one shoulf destination before
                # os.system("%s -d 2 -i %s -o %s -t %s -t [%s, 1]" % (
                os.system("%s -d 2 -i %s -o %s -t %s -t [%s ,1]" % (
                    #    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, invwarp_file, sift_affine_file))
                    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, invwarp_file, surf_affine_file))
                # os.system("%s -d 2 -i %s -o %s -t %s -t %s" % (
                #    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, sift_affine_file, sift_affine_file))
                os.chdir(pwd_dir)

                # os.chdir(pwd_dir)


def return_bbox(original_image_index: object, middle_image_index: object, slice_roi_list: object,
                image_output_dir: object):
    ants_bin_dir = '/home/dengr/Tools/ANTS/install/bin'
    os.environ["ANTSPATH"] = ants_bin_dir

    original_name = 'output'
    output_name = '3Doutput'
    pwd_dir = os.getcwd()
    output_dir = os.path.join(image_output_dir, str(slice_roi_list[original_image_index]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    antsApplyTransformsCMD = os.path.join(ants_bin_dir, 'antsApplyTransforms')
    antsApplyTransformsToPointsCMD = os.path.join(ants_bin_dir, 'antsApplyTransformsToPoints')

    # image = nib.load(os.path.join(image_output_dir, '3D_reconstruction.nii'))
    # image = image.get_fdata()

    '''
    #slice = slice.swapaxes(1, 0)
    if original_image_index == 0:
        for pi in range(len(slice_roi_list)):
            if pi < middle_image_index:
                original_dir = os.path.join(image_output_dir, '%d-to-%d' % (
                    slice_roi_list[pi], slice_roi_list[pi+1]))
            elif pi == middle_image_index:
                continue
            else:
                original_dir = os.path.join(image_output_dir, '%d-to-%d' % (
                    slice_roi_list[pi], slice_roi_list[pi-1]))


            original_3D_folder = os.path.join(original_dir, 'step2_run_ants_reg')
            piece = image[:, :, slice_roi_list[pi]]
            piece = nib.Nifti1Image(piece, np.eye(4))
            piece.header.get_xyzt_units()
            nib.save(piece, os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name)))
            #os.system("cp %s %s" % (original_3D_file, output_3D_file))

    #slice = image[:, :, slice_roi_list[original_image_index]]
    '''

    moving_roi_curr = slice_roi_list[middle_image_index - 1]
    fixed_roi_curr = slice_roi_list[middle_image_index]
    destination_dir = os.path.join(image_output_dir, '%d-to-%d' % (moving_roi_curr, fixed_roi_curr))
    destination_3D_folder = os.path.join(destination_dir, 'step1_prepare_data')
    # destination_3D_file = os.path.join(destination_3D_folder, 'fix.nii.gz')   #tranfer the image to this nii
    destination_3D_csv = os.path.join(destination_3D_folder, 'fixed_bbox_coordinates.csv')
    print('slices: ' + str(slice_roi_list[original_image_index]))

    if original_image_index < middle_image_index:
        for bi in reversed(range(len(slice_roi_list[original_image_index:middle_image_index]))):
            moving_roi_curr = slice_roi_list[original_image_index + bi]
            print('moving_roi_curr: ' + str(moving_roi_curr))
            fixed_roi_curr = slice_roi_list[original_image_index + bi + 1]
            print('fixed_roi_curr: ' + str(fixed_roi_curr))
            working_dir = os.path.join(image_output_dir, '%d-to-%d' % (moving_roi_curr, fixed_roi_curr))
            if fixed_roi_curr == slice_roi_list[middle_image_index]:
                original_dir = os.path.join(image_output_dir, '%d-to-%d' % (
                    slice_roi_list[original_image_index], slice_roi_list[original_image_index + 1]))
                original_3D_folder = os.path.join(original_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                original_csv_folder = os.path.join(original_dir, 'step1_prepare_data')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')
                print('copy')
                print(original_csv_file)
                print(output_csv_file)
                # print(original_3D_file)
                # print(output_3D_file)
                # if not os.path.exists(output_3D_file):
                # slice = nib.Nifti1Image(slice, np.eye(4))
                # slice.header.get_xyzt_units()
                # nib.save(slice, os.path.join(original_3D_folder,'%sWarped.nii.gz' % (original_name)))
                #    os.system("cp %s %s" % (original_3D_file, output_3D_file))
                if not os.path.exists(output_csv_file):
                    print('yes')
                    os.system("cp %s %s" % (original_csv_file, output_csv_file))

                original_csv_folder = os.path.join(original_dir, 'step1_prepare_data')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

                begin_image_folder = os.path.join(working_dir, 'step1_prepare_data')
                begin_image = os.path.join(begin_image_folder, 'fix.nii.gz')

                matrix_dir = os.path.join(working_dir, 'step2_run_ants_reg')
                # sift_affine_dir = os.path.join(matrix_dir, 'sift_affine')
                surf_affine_file = os.path.join(matrix_dir, 'output0GenericAffine.mat')
                warp_file = os.path.join(matrix_dir, 'output1Warp.nii.gz')
                invwarp_file = os.path.join(matrix_dir, 'output1InverseWarp.nii.gz')

                original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                os.chdir(output_dir)

                print(output_3D_file)
                print(original_3D_file)

                # os.system("%s -d 2 -i %s -r %s -t [%s,1] -o %s " % (
                #   antsApplyTransformsCMD, output_3D_file, original_3D_file, sift_affine_file, output_3D_file))
                # os.system("%s -d 2 -i %s -o %s -t %s -t [%s ,1]" % (

                # print(warp_file)
                # print(sift_affine_file)
                os.system("%s -d 2 -i %s -o %s -t %s " % (
                    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, surf_affine_file))
                os.chdir(pwd_dir)

                # img_nii = nib.load(output_3D_file)
                # img = img_nii.get_fdata()
                # print(img.shape)
                # print(original_csv_file)
                # print(output_csv_file)

            else:
                # original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                # original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                original_dir = os.path.join(image_output_dir, '%d-to-%d' % (
                    slice_roi_list[original_image_index], slice_roi_list[original_image_index + 1]))
                original_3D_folder = os.path.join(original_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))

                original_csv_folder = os.path.join(original_dir, 'step1_prepare_data')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

                # begin_image_folder = os.path.join(working_dir, 'step1_prepare_data')
                # begin_image = os.path.join(begin_image_folder, 'fix.nii.gz')

                matrix_dir = os.path.join(working_dir, 'step2_run_ants_reg')
                # sift_affine_dir = os.path.join(matrix_dir, 'sift_affine')
                surf_affine_file = os.path.join(matrix_dir, 'output0GenericAffine.mat')
                # output_affine_file = os.path.join(sift_affine_dir, 'sift_affine_init.mat')
                warp_file = os.path.join(matrix_dir, 'output1Warp.nii.gz')
                invwarp_file = os.path.join(matrix_dir, 'output1InverseWarp.nii.gz')

                original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                print(output_3D_file)
                print(original_3D_file)

                os.chdir(output_dir)
                # os.system("%s -d 2 -i %s -r %s -t [%s,1] -o %s " % (
                #    antsApplyTransformsCMD, output_3D_file, original_3D_file, sift_affine_file, output_3D_file))##second one shoulf destination before
                # os.system("%s -d 2 -i %s -o %s -t %s -t [%s ,1]" % (

                os.system("%s -d 2 -i %s -o %s -t %s " % (
                    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, surf_affine_file))

                os.chdir(pwd_dir)

    elif original_image_index == middle_image_index:
        working_dir = os.path.join(image_output_dir, '%d-to-%d' % (
        slice_roi_list[original_image_index], slice_roi_list[original_image_index + 1]))
        output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))
        original_csv_folder = os.path.join(working_dir, 'step1_prepare_data')
        original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
        output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

        # slice = nib.Nifti1Image(slice, np.eye(4))
        # slice.header.get_xyzt_units()
        print('output123123')
        print(os.path.join(output_dir, '%sWarped.nii.gz' % (original_name)))
        # nib.save(slice, os.path.join(output_dir, '%sWarped.nii.gz' % (original_name)))

        # os.system("cp %s %s" % (destination_3D_file, output_3D_file))
        os.system("cp %s %s" % (destination_3D_csv, output_csv_file))
        print('copy')
        print(destination_3D_csv)
        print(output_csv_file)

    else:
        for bi in reversed(range(len(slice_roi_list[middle_image_index:original_image_index]))):
            moving_roi_curr = slice_roi_list[original_image_index - bi]
            fixed_roi_curr = slice_roi_list[original_image_index - bi - 1]
            print('moving_roi_curr: ' + str(moving_roi_curr))
            print('fixed_roi_curr: ' + str(fixed_roi_curr))

            working_dir = os.path.join(image_output_dir, '%d-to-%d' % (moving_roi_curr, fixed_roi_curr))
            if fixed_roi_curr == slice_roi_list[middle_image_index]:
                original_dir = os.path.join(image_output_dir, '%d-to-%d' % (
                    slice_roi_list[original_image_index], slice_roi_list[original_image_index - 1]))
                original_3D_folder = os.path.join(original_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                original_csv_folder = os.path.join(original_dir, 'step1_prepare_data')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')
                print('copy')
                print(os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name)))
                print(output_3D_file)
                # if not os.path.exists(output_3D_file):
                # slice = nib.Nifti1Image(slice, np.eye(4))
                # slice.header.get_xyzt_units()
                # nib.save(slice, os.path.join(original_3D_folder,'%sWarped.nii.gz' % (original_name)))
                # os.system("cp %s %s" % (original_3D_file, output_3D_file))
                if not os.path.exists(output_csv_file):
                    os.system("cp %s %s" % (original_csv_file, output_csv_file))

                original_csv_folder = os.path.join(original_dir, 'step1_prepare_data')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

                begin_image_folder = os.path.join(working_dir, 'step1_prepare_data')
                begin_image = os.path.join(begin_image_folder, 'fix.nii.gz')

                matrix_dir = os.path.join(working_dir, 'step2_run_ants_reg')
                # sift_affine_dir = os.path.join(matrix_dir, 'sift_affine')
                surf_affine_file = os.path.join(matrix_dir, 'output0GenericAffine.mat')
                warp_file = os.path.join(matrix_dir, 'output1Warp.nii.gz')
                invwarp_file = os.path.join(matrix_dir, 'output1InverseWarp.nii.gz')

                original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                os.chdir(output_dir)

                print(output_3D_file)
                print(original_3D_file)
                # os.system("%s -d 2 -i %s -r %s -t [%s,1] -o %s " % (
                #   antsApplyTransformsCMD, output_3D_file, original_3D_file, sift_affine_file, output_3D_file))
                # os.system("%s -d 2 -i %s -o %s -t %s -t [%s ,1]" % (
                # print(warp_file)
                # print(sift_affine_file)

                os.system("%s -d 2 -i %s -o %s -t %s " % (
                    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, surf_affine_file))
                os.chdir(pwd_dir)


            else:
                # original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                # original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                original_dir = os.path.join(image_output_dir, '%d-to-%d' % (
                    slice_roi_list[original_image_index], slice_roi_list[original_image_index - 1]))
                original_3D_folder = os.path.join(original_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))

                original_csv_folder = os.path.join(original_dir, 'step1_prepare_data')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

                begin_image_folder = os.path.join(working_dir, 'step1_prepare_data')
                begin_image = os.path.join(begin_image_folder, 'fix.nii.gz')

                matrix_dir = os.path.join(working_dir, 'step2_run_ants_reg')
                # sift_affine_dir = os.path.join(matrix_dir, 'sift_affine')
                surf_affine_file = os.path.join(matrix_dir, 'output0GenericAffine.mat')
                warp_file = os.path.join(matrix_dir, 'output1Warp.nii.gz')
                invwarp_file = os.path.join(matrix_dir, 'output1InverseWarp.nii.gz')
                original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                os.chdir(output_dir)
                print(output_3D_file)
                print(original_3D_file)

                # os.system("%s -d 2 -i %s -r %s -t [%s,1] -o %s " % (
                #    antsApplyTransformsCMD, output_3D_file, original_3D_file, sift_affine_file, output_3D_file))
                # os.system("%s -d 2 -i %s -o %s -t %s -t [%s ,1]" % (
                # print(output_csv_file)
                # print(warp_file)
                # print(sift_affine_file)
                os.system("%s -d 2 -i %s -o %s -t %s  " % (
                    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, surf_affine_file))
                os.chdir(pwd_dir)


def forward_bbox(original_image_index: object, middle_image_index: object, slice_roi_list: object,
                 image_output_dir: object):
    ants_bin_dir = '/home/dengr/Tools/ANTS/install/bin'
    os.environ["ANTSPATH"] = ants_bin_dir

    original_name = 'output'
    output_name = '3Doutput'
    pwd_dir = os.getcwd()
    output_dir = os.path.join(image_output_dir, str(slice_roi_list[original_image_index]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    antsApplyTransformsCMD = os.path.join(ants_bin_dir, 'antsApplyTransforms')
    antsApplyTransformsToPointsCMD = os.path.join(ants_bin_dir, 'antsApplyTransformsToPoints')

    # slice = slice.swapaxes(1, 0)
    '''
    if original_image_index == 0:
        for pi in range(len(slice_roi_list)):
            if pi < middle_image_index:
                original_dir = os.path.join(image_output_dir, '%d-to-%d' % (
                    slice_roi_list[pi], slice_roi_list[pi+1]))
            elif pi == middle_image_index:
                continue
            else:
                original_dir = os.path.join(image_output_dir, '%d-to-%d' % (
                    slice_roi_list[pi], slice_roi_list[pi-1]))


            original_3D_folder = os.path.join(original_dir, 'step2_run_ants_reg')
            piece = image[:, :, slice_roi_list[pi]]
            piece = nib.Nifti1Image(piece, np.eye(4))
            piece.header.get_xyzt_units()
            nib.save(piece, os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name)))
            #os.system("cp %s %s" % (original_3D_file, output_3D_file))
    '''
    # slice = image[:, :, slice_roi_list[original_image_index]]

    moving_roi_curr = slice_roi_list[middle_image_index - 1]
    fixed_roi_curr = slice_roi_list[middle_image_index]
    destination_dir = os.path.join(image_output_dir, '%d-to-%d' % (moving_roi_curr, fixed_roi_curr))
    destination_3D_folder = os.path.join(destination_dir, 'step1_prepare_data')
    # destination_3D_file = os.path.join(destination_3D_folder, 'fix.nii.gz')   #tranfer the image to this nii
    destination_3D_csv = os.path.join(destination_3D_folder, 'fixed_bbox_coordinates.csv')
    print('slices: ' + str(slice_roi_list[original_image_index]))

    if original_image_index < middle_image_index:
        for bi in reversed(range(len(slice_roi_list[original_image_index:middle_image_index]))):
            moving_roi_curr = slice_roi_list[original_image_index + bi]
            print('moving_roi_curr: ' + str(moving_roi_curr))
            fixed_roi_curr = slice_roi_list[original_image_index + bi + 1]
            print('fixed_roi_curr: ' + str(fixed_roi_curr))
            working_dir = os.path.join(image_output_dir, '%d-to-%d' % (moving_roi_curr, fixed_roi_curr))
            if fixed_roi_curr == slice_roi_list[middle_image_index]:
                original_dir = os.path.join(image_output_dir, '%d-to-%d' % (
                    slice_roi_list[original_image_index], slice_roi_list[original_image_index + 1]))
                original_3D_folder = os.path.join(original_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                original_csv_folder = os.path.join(original_dir, 'step1_prepare_data')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')
                print('copy')
                print(original_csv_file)
                print(output_csv_file)
                # print(original_3D_file)
                # print(output_3D_file)
                # if not os.path.exists(output_3D_file):
                # slice = nib.Nifti1Image(slice, np.eye(4))
                # slice.header.get_xyzt_units()
                # nib.save(slice, os.path.join(original_3D_folder,'%sWarped.nii.gz' % (original_name)))
                #    os.system("cp %s %s" % (original_3D_file, output_3D_file))
                if not os.path.exists(output_csv_file):
                    print('yes')
                    os.system("cp %s %s" % (original_csv_file, output_csv_file))

                original_csv_folder = os.path.join(original_dir, 'step1_prepare_data')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

                begin_image_folder = os.path.join(working_dir, 'step1_prepare_data')
                begin_image = os.path.join(begin_image_folder, 'fix.nii.gz')

                matrix_dir = os.path.join(working_dir, 'step2_run_ants_reg')
                # sift_affine_dir = os.path.join(matrix_dir, 'sift_affine')
                surf_affine_file = os.path.join(matrix_dir, 'output0GenericAffine.mat')
                warp_file = os.path.join(matrix_dir, 'output1Warp.nii.gz')
                invwarp_file = os.path.join(matrix_dir, 'output1InverseWarp.nii.gz')

                original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                os.chdir(output_dir)

                print(output_3D_file)
                print(original_3D_file)

                # os.system("%s -d 2 -i %s -r %s -t [%s,1] -o %s " % (
                #   antsApplyTransformsCMD, output_3D_file, original_3D_file, sift_affine_file, output_3D_file))
                # os.system("%s -d 2 -i %s -o %s -t %s -t [%s ,1]" % (

                # print(warp_file)
                # print(sift_affine_file)
                os.system("%s -d 2 -i %s -o %s -t [%s,1] " % (
                    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, surf_affine_file))
                os.chdir(pwd_dir)

                # img_nii = nib.load(output_3D_file)
                # img = img_nii.get_fdata()
                # print(img.shape)
                # print(original_csv_file)
                # print(output_csv_file)

            else:
                # original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                # original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                original_dir = os.path.join(image_output_dir, '%d-to-%d' % (
                    slice_roi_list[original_image_index], slice_roi_list[original_image_index + 1]))
                original_3D_folder = os.path.join(original_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))

                original_csv_folder = os.path.join(original_dir, 'step1_prepare_data')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

                # begin_image_folder = os.path.join(working_dir, 'step1_prepare_data')
                # begin_image = os.path.join(begin_image_folder, 'fix.nii.gz')

                matrix_dir = os.path.join(working_dir, 'step2_run_ants_reg')
                # sift_affine_dir = os.path.join(matrix_dir, 'sift_affine')
                surf_affine_file = os.path.join(matrix_dir, 'output0GenericAffine.mat')
                # output_affine_file = os.path.join(sift_affine_dir, 'sift_affine_init.mat')
                warp_file = os.path.join(matrix_dir, 'output1Warp.nii.gz')
                invwarp_file = os.path.join(matrix_dir, 'output1InverseWarp.nii.gz')

                original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                print(output_3D_file)
                print(original_3D_file)

                os.chdir(output_dir)
                # os.system("%s -d 2 -i %s -r %s -t [%s,1] -o %s " % (
                #    antsApplyTransformsCMD, output_3D_file, original_3D_file, sift_affine_file, output_3D_file))##second one shoulf destination before
                # os.system("%s -d 2 -i %s -o %s -t %s -t [%s ,1]" % (

                os.system("%s -d 2 -i %s -o %s -t [%s,1] " % (
                    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, surf_affine_file))

                os.chdir(pwd_dir)

    elif original_image_index == middle_image_index:
        working_dir = os.path.join(image_output_dir, '%d-to-%d' % (
        slice_roi_list[original_image_index], slice_roi_list[original_image_index + 1]))
        output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))
        original_csv_folder = os.path.join(working_dir, 'step1_prepare_data')
        original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
        output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

        # slice = nib.Nifti1Image(slice, np.eye(4))
        # slice.header.get_xyzt_units()
        print('output123123')
        print(os.path.join(output_dir, '%sWarped.nii.gz' % (original_name)))
        # nib.save(slice, os.path.join(output_dir, '%sWarped.nii.gz' % (original_name)))

        # os.system("cp %s %s" % (destination_3D_file, output_3D_file))
        os.system("cp %s %s" % (destination_3D_csv, output_csv_file))
        print('copy')
        print(destination_3D_csv)
        print(output_csv_file)

    else:
        for bi in reversed(range(len(slice_roi_list[middle_image_index:original_image_index]))):
            moving_roi_curr = slice_roi_list[original_image_index - bi]
            fixed_roi_curr = slice_roi_list[original_image_index - bi - 1]
            print('moving_roi_curr: ' + str(moving_roi_curr))
            print('fixed_roi_curr: ' + str(fixed_roi_curr))

            working_dir = os.path.join(image_output_dir, '%d-to-%d' % (moving_roi_curr, fixed_roi_curr))
            if fixed_roi_curr == slice_roi_list[middle_image_index]:
                original_dir = os.path.join(image_output_dir, '%d-to-%d' % (
                    slice_roi_list[original_image_index], slice_roi_list[original_image_index - 1]))
                original_3D_folder = os.path.join(original_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                original_csv_folder = os.path.join(original_dir, 'step1_prepare_data')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')
                print('copy')
                print(os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name)))
                print(output_3D_file)
                # if not os.path.exists(output_3D_file):
                # slice = nib.Nifti1Image(slice, np.eye(4))
                # slice.header.get_xyzt_units()
                # nib.save(slice, os.path.join(original_3D_folder,'%sWarped.nii.gz' % (original_name)))
                # os.system("cp %s %s" % (original_3D_file, output_3D_file))
                if not os.path.exists(output_csv_file):
                    os.system("cp %s %s" % (original_csv_file, output_csv_file))

                original_csv_folder = os.path.join(original_dir, 'step1_prepare_data')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

                begin_image_folder = os.path.join(working_dir, 'step1_prepare_data')
                begin_image = os.path.join(begin_image_folder, 'fix.nii.gz')

                matrix_dir = os.path.join(working_dir, 'step2_run_ants_reg')
                # sift_affine_dir = os.path.join(matrix_dir, 'sift_affine')
                surf_affine_file = os.path.join(matrix_dir, 'output0GenericAffine.mat')
                warp_file = os.path.join(matrix_dir, 'output1Warp.nii.gz')
                invwarp_file = os.path.join(matrix_dir, 'output1InverseWarp.nii.gz')

                original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                os.chdir(output_dir)

                print(output_3D_file)
                print(original_3D_file)
                # os.system("%s -d 2 -i %s -r %s -t [%s,1] -o %s " % (
                #   antsApplyTransformsCMD, output_3D_file, original_3D_file, sift_affine_file, output_3D_file))
                # os.system("%s -d 2 -i %s -o %s -t %s -t [%s ,1]" % (
                # print(warp_file)
                # print(sift_affine_file)

                os.system("%s -d 2 -i %s -o %s -t [%s,1] " % (
                    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, surf_affine_file))
                os.chdir(pwd_dir)


            else:
                # original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                # original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                original_dir = os.path.join(image_output_dir, '%d-to-%d' % (
                    slice_roi_list[original_image_index], slice_roi_list[original_image_index - 1]))
                original_3D_folder = os.path.join(original_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))

                original_csv_folder = os.path.join(original_dir, 'step1_prepare_data')
                original_csv_file = os.path.join(original_csv_folder, 'moving_bbox_coordinates.csv')
                output_csv_file = os.path.join(output_dir, 'moving_bbox_coordinates.csv')

                begin_image_folder = os.path.join(working_dir, 'step1_prepare_data')
                begin_image = os.path.join(begin_image_folder, 'fix.nii.gz')

                matrix_dir = os.path.join(working_dir, 'step2_run_ants_reg')
                # sift_affine_dir = os.path.join(matrix_dir, 'sift_affine')
                surf_affine_file = os.path.join(matrix_dir, 'output0GenericAffine.mat')
                warp_file = os.path.join(matrix_dir, 'output1Warp.nii.gz')
                invwarp_file = os.path.join(matrix_dir, 'output1InverseWarp.nii.gz')
                original_3D_folder = os.path.join(working_dir, 'step2_run_ants_reg')
                original_3D_file = os.path.join(original_3D_folder, '%sWarped.nii.gz' % (original_name))
                output_3D_file = os.path.join(output_dir, '%sWarped.nii.gz' % (original_name))

                os.chdir(output_dir)
                print(output_3D_file)
                print(original_3D_file)

                # os.system("%s -d 2 -i %s -r %s -t [%s,1] -o %s " % (
                #    antsApplyTransformsCMD, output_3D_file, original_3D_file, sift_affine_file, output_3D_file))
                # os.system("%s -d 2 -i %s -o %s -t %s -t [%s ,1]" % (
                # print(output_csv_file)
                # print(warp_file)
                # print(sift_affine_file)
                os.system("%s -d 2 -i %s -o %s -t [%s,1]  " % (
                    antsApplyTransformsToPointsCMD, output_csv_file, output_csv_file, surf_affine_file))
                os.chdir(pwd_dir)


def register_a_pair_back(moving_jpg: object, fixed_jpg: object, working_dir: object, df: object,
                         fix_seg_nii: object = None, df_manual: object = None) -> object:
    ants_bin_dir = '/home/dengr/Tools/ANTS/install/bin'
    os.environ["ANTSPATH"] = ants_bin_dir

    output_name = 'output'

    visulization = True

    # step 1 ============================================================================================================
    step_1_dir = os.path.join(working_dir, 'step1_prepare_data')
    if not os.path.exists(step_1_dir):
        os.makedirs(step_1_dir)

    # #copy raw data
    moving_fname = os.path.basename(moving_jpg)
    fix_fname = os.path.basename(fixed_jpg)
    moving_jpg_file = os.path.join(step_1_dir, moving_fname)
    fix_jpg_file = os.path.join(step_1_dir, fix_fname)
    if not os.path.exists(moving_jpg_file):
        os.system("cp %s %s" % (moving_jpg, moving_jpg_file))
    if not os.path.exists(fix_jpg_file):
        os.system("cp %s %s" % (fixed_jpg, fix_jpg_file))

    # convert jpg to nifti
    ConvertCMD = os.path.join(ants_bin_dir, 'ConvertImagePixelType')
    moving_nii = os.path.join(step_1_dir, 'moving.nii.gz')
    fix_nii = os.path.join(step_1_dir, 'fix.nii.gz')
    if not os.path.exists(moving_nii):
        os.system("%s %s %s 1" % (ConvertCMD, moving_jpg, moving_nii))
    if not os.path.exists(fix_nii):
        os.system("%s %s %s 1" % (ConvertCMD, fixed_jpg, fix_nii))

    # prepare label csv file
    fixed_bbox_csv = os.path.join(step_1_dir, 'fixed_bbox_coordinates.csv')
    if not os.path.exists(fixed_bbox_csv):
        write_bbox_to_csv(df, fixed_bbox_csv)

    if df_manual is not None:
        manual_bbox_csv = os.path.join(step_1_dir, 'manual_bbox_coordinates.csv')
        if not os.path.exists(manual_bbox_csv):
            write_bbox_to_csv(df_manual, manual_bbox_csv)

    # prepare initial affine matrix
    surf_affine_dir = os.path.join(step_1_dir, 'surf_affine')
    surf_affine_file = os.path.join(surf_affine_dir, 'surf_affine_init.mat')
    if not os.path.exists(surf_affine_file):
        surf_affine(moving_jpg_file, fix_jpg_file, surf_affine_dir)
    affine_init_file = os.path.join(surf_affine_dir, 'surf_affine_file.nii.gz')
    antsApplyTransformsCMD = os.path.join(ants_bin_dir, 'antsApplyTransforms')
    if not os.path.exists(affine_init_file):
        pwd_dir = os.getcwd()
        os.chdir(surf_affine_dir)
        os.system("%s -d 2 -i %s -r %s -t %s -o %s " % (
        antsApplyTransformsCMD, moving_nii, fix_nii, surf_affine_file, affine_init_file))
        os.chdir(pwd_dir)

    # step 2 ============================================================================================================
    final_fname = '%sWarped.nii.gz' % (output_name)
    invfinal_fname = '%sInverseWarped.nii.gz' % (output_name)
    warp_fname = '%s1Warp.nii.gz' % (output_name)
    invwarp_fname = '%s1InverseWarp.nii.gz' % (output_name)
    aff_mtx_fname = '%s0GenericAffine.mat' % (output_name)

    pwd_dir = os.getcwd()
    step_2_dir = os.path.join(working_dir, 'step2_run_ants_reg')
    if not os.path.exists(step_2_dir):
        os.makedirs(step_2_dir)
    antsRegistrationSyNCMD = os.path.join(ants_bin_dir, 'antsRegistrationSyN.sh')
    output_file = os.path.join(step_2_dir, output_name)
    final_file = os.path.join(step_2_dir, final_fname)
    if not os.path.exists(final_file):
        os.chdir(step_2_dir)
        if os.path.exists(surf_affine_file):
            os.system("%s -d 2 -m %s -f %s -j 1 -t s -n 8 -i %s - o %s " % (
            antsRegistrationSyNCMD, moving_nii, fix_nii, surf_affine_file, output_file))
        else:
            os.system("%s -d 2 -m %s -f %s -j 1 -t s -n 8 - o %s " % (
            antsRegistrationSyNCMD, moving_nii, fix_nii, output_file))
        os.chdir(pwd_dir)

    # step 3 get inverse registration points ============================================================================
    step_3_dir = os.path.join(working_dir, 'step3_inv_reg_points')
    if not os.path.exists(step_3_dir):
        os.makedirs(step_3_dir)

    antsApplyTransformsToPointsCMD = os.path.join(ants_bin_dir, 'antsApplyTransformsToPoints')
    moving_bbox_csv = os.path.join(step_3_dir, 'moving_bbox_coordinates.csv')
    warp_file = os.path.join(step_2_dir, warp_fname)
    aff_mtx_file = os.path.join(step_2_dir, aff_mtx_fname)
    os.chdir(step_3_dir)
    if not os.path.exists(moving_bbox_csv):
        os.system("%s -d 2 -i %s -o %s -t %s -t %s" % (
            antsApplyTransformsToPointsCMD, fixed_bbox_csv, moving_bbox_csv, warp_file, aff_mtx_file))

    if fix_seg_nii != None and os.path.exists(fix_seg_nii):
        moving_seg_nii = os.path.join(step_3_dir, 'moving_seg.nii.gz')
        if not os.path.exists(moving_seg_nii):
            os.system("%s -d 2 -i %s -r %s -t [%s, 1] -t %s -n GenericLabel -o %s" % (
                antsApplyTransformsCMD, fix_seg_nii, moving_nii, aff_mtx_file, warp_file, moving_seg_nii))

    os.chdir(pwd_dir)
    df_new = pd.read_csv(moving_bbox_csv)

    # step 4 visulization ===============================================================================================
    step_4_dir = os.path.join(working_dir, 'step4_visulization')
    if not os.path.exists(step_4_dir):
        os.makedirs(step_4_dir)
    if visulization:
        fixed_vis_jpg_file = os.path.join(step_4_dir, 'fixed_bbox.jpg')
        moving_vis_jpg_file = os.path.join(step_4_dir, 'moving_bbox.jpg')
        fixed_seg_jpg_file = os.path.join(step_4_dir, 'fixed_seg.jpg')
        moving_seg_jpg_file = os.path.join(step_4_dir, 'moving_seg.jpg')

        if not os.path.exists(fixed_vis_jpg_file):
            show_bounding_box(fixed_jpg, fixed_bbox_csv, fixed_vis_jpg_file)
            if fix_seg_nii != None and os.path.exists(fix_seg_nii):
                show_bounding_box(fixed_jpg, fixed_bbox_csv, fixed_seg_jpg_file, fix_seg_nii)

        if not os.path.exists(moving_vis_jpg_file):
            show_bounding_box(moving_jpg, moving_bbox_csv, moving_vis_jpg_file)
            if fix_seg_nii != None and os.path.exists(moving_seg_nii):
                show_bounding_box(moving_jpg, moving_bbox_csv, moving_seg_jpg_file, moving_seg_nii)

        if df_manual is not None:
            fixed_vis_jpg_file = os.path.join(step_4_dir, 'fixed_manual_bbox.jpg')
            moving_vis_jpg_file = os.path.join(step_4_dir, 'moving_manual_bbox.jpg')
            if not os.path.exists(fixed_vis_jpg_file):
                show_bounding_box(fixed_jpg, manual_bbox_csv, fixed_vis_jpg_file, t=0)
            if not os.path.exists(moving_vis_jpg_file):
                show_bounding_box(moving_jpg, manual_bbox_csv, moving_vis_jpg_file, t=1)

    if fix_seg_nii != None and os.path.exists(moving_seg_nii):
        return df_new, moving_seg_nii
    else:
        return df_new


if __name__ == "__main__":

    fixed_bbox = [[2383, 2827, 2451, 2910]]

    # moving_jpg = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/output_overlay/13-261-x-ROI_3-x-37515-x-32775-x-8861-x-10612.jpg'
    # fixed_jpg = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/output_overlay/13-261-x-ROI_5-x-57097-x-34733-x-9015-x-10767.jpg'
    # working_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/output/10612_to_10767'

    moving_jpg = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/output_overlay/13-261-x-ROI_7-x-14894-x-18862-x-9736-x-10200.jpg'
    fixed_jpg = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/output/test/2.png'
    working_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/output/test/ROI6'

    # moving_jpg = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/output/test/4.png'
    # fixed_jpg = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/output/test/5.png'
    # working_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/registration/data/output/test/ROI6_trans'

    df = pd.DataFrame(columns=['x', 'y', 't', 'label'])
    row = 0
    for fi in range(len(fixed_bbox)):
        bbox = fixed_bbox[fi]
        assert len(bbox) == 4
        # data = [bbox[0], bbox[1], 0, 0]
        # df.append(data)

        df.loc[row] = [bbox[0], bbox[1], 0, 0]
        row = row + 1
        df.loc[row] = [bbox[2], bbox[1], 0, 0]
        row = row + 1
        df.loc[row] = [bbox[2], bbox[3], 0, 0]
        row = row + 1
        df.loc[row] = [bbox[0], bbox[3], 0, 0]
        row = row + 1

    register_a_pair(moving_jpg, fixed_jpg, working_dir, df)