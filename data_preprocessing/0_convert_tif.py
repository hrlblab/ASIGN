from PIL import Image
import os

"""
This is the first step in data preprocessing.
If your original image is in tif format, you can use this file to process the raw WSI.
Please note that for some png or svs format WSIs, the barcode coordinates in ST data may be flipped, such as x = y and y = x or x = -y.
During preprocessing, please make sure to first check whether the spot coordinates match the loaded image correctly.
"""


def convert_tif_to_png(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.tif') or file_name.lower().endswith('.tiff'):
            input_path = os.path.join(input_folder, file_name)
            output_file_name = os.path.splitext(file_name)[0] + '.png'
            output_path = os.path.join(output_folder, output_file_name)

            try:
                with Image.open(input_path) as img:
                    img.save(output_path, format='PNG')
                    print(f"Converted {file_name} to {output_file_name}")
            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")


input_folder = "./raw_data"
output_folder = "./img_png"
convert_tif_to_png(input_folder, output_folder)
