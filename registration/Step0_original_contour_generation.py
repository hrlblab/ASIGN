import os
import pandas as pd
import numpy as np


def generate_circle_contour(cx, cy, radius, num_points=100):
    # Generate angles for circle contour points
    angles = np.linspace(0, 2 * np.pi, num_points)

    # Calculate the x and y coordinates of the contour points
    contour = np.array([
        [int(cx + radius * np.cos(angle)), int(cy + radius * np.sin(angle))]
        for angle in angles
    ])

    # Reshape the contour to fit the format used by cv2.drawContours
    contour = contour.reshape((-1, 1, 2))

    return contour


# Directories for input and output data
position_dir = './ST_Breast/location'
output_dir = './ST_Breast/WSI_png_test_contour'
for sample in os.listdir(position_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(os.path.join(output_dir, sample)):
        os.makedirs(os.path.join(output_dir, sample))

    for file in os.listdir(os.path.join(position_dir, sample)):
        # Initialize real_x and real_y (assumed to be 0)
        real_x = 0
        real_y = 0
        # Read the CSV file containing position data
        position = pd.read_csv(os.path.join(position_dir, sample, file))
        column_names = ['barcode', 'pixel_x', 'pixel_y', 'i', 'j']
        position.columns = column_names

        # Create save path for output CSV
        save_path = os.path.join(os.path.join(output_dir, sample), file[:-4] + '_%d_%d_spot.csv' % (real_x, real_y))
        position_intissue = position

        # Create array to store point data
        point_array = np.zeros((len(position_intissue), 4))
        point_array[:, 0] = np.array(position_intissue['pixel_x'].tolist()).astype('int64')
        point_array[:, 1] = np.array(position_intissue['pixel_y'].tolist()).astype('int64')

        # point_array_onsample uses extracted coordinates
        point_array_onsample = point_array

        # Generate a circle contour
        radius = 112
        num_points = 100

        # Generate a single circle contour
        single_circle_contour = generate_circle_contour(0, 0, radius, num_points=num_points)

        # Create batch of circular contours, initialized with zeros
        batch_circle_contour = np.zeros((len(point_array_onsample), num_points, 1, 2))

        # Copy single circle contour to each point in the batch
        batch_circle_contour[:] = single_circle_contour

        # Translate the contours to center them on each point
        batch_circle_contour[:, :, 0, 0] = batch_circle_contour[:, :, 0, 0] + np.round(
            (point_array_onsample[:, 0] - real_x)).astype(int)[:, np.newaxis]
        batch_circle_contour[:, :, 0, 1] = batch_circle_contour[:, :, 0, 1] + np.round(
            (point_array_onsample[:, 1] - real_y)).astype(int)[:, np.newaxis]

        # Prepare columns for output DataFrame
        columns = ['barcode'] + \
                  ['now_x_%d' % ki for ki in range(num_points)] + \
                  ['now_y_%d' % ki for ki in range(num_points)]

        # Create DataFrame to store contours
        contours = pd.DataFrame(columns=columns)
        contours['barcode'] = position_intissue['barcode']

        # Populate DataFrame with contour coordinates
        for ni in range(num_points):
            contours['now_x_%d' % (ni)] = np.round(batch_circle_contour[:, ni, 0, 1]).astype(int).tolist()
            contours['now_y_%d' % (ni)] = np.round(batch_circle_contour[:, ni, 0, 0]).astype(int).tolist()

        # Save the contours to CSV
        contours.to_csv(save_path, index=False)
