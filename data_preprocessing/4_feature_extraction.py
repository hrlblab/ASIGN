import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import numpy as np
from tqdm import tqdm


def extract_image_features(image_folder, output_file, target_dim=1024):
    """
    Load images from a given folder, extract image features using a pre-trained ResNet50 model,
    reduce the feature dimensions to 1024, and save the results as a .npy file.

    Parameters:
    image_folder (str): Path to the folder containing images.
    output_file (str): Path to save the extracted features, default is 'image_features.npy'.
    target_dim (int): Dimension of the final feature, default is 1024.
    """

    # 1. Define image preprocessing steps (without CenterCrop)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Load pre-trained ResNet50 model and remove the last classification layer
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer
    model.eval()  # Set to evaluation mode

    # 3. Add a linear layer to reduce 2048-dimensional features to 1024 dimensions
    linear_layer = torch.nn.Linear(2048, target_dim)
    linear_layer = linear_layer.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 4. Check if GPU is available, and use it if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 5. Define a dictionary to store image names and their features
    features_dict = {}

    # 6. Iterate over all images in the folder
    for img_name in tqdm(os.listdir(image_folder)):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
            # 7. Load the image
            img_path = os.path.join(image_folder, img_name)
            img = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format
            img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
            img_tensor = img_tensor.to(device)

            # 8. Extract 2048-dimensional features using ResNet50
            with torch.no_grad():
                feature_2048 = model(img_tensor).squeeze()

                # 9. Reduce the dimension to 1024 using the linear layer
                feature_1024 = linear_layer(feature_2048).cpu().numpy()

            # 10. Store image name and features
            features_dict[img_name[:-4]] = feature_1024

    # 11. Save the features dictionary as a .npy file
    np.save(output_file, features_dict)

    print(f"Feature extraction complete and saved to '{output_file}'")


# Example call:
img_roots = ['./ST_Breast/cropped_imgs/patches_1024',
             './ST_Breast/cropped_imgs/patches_512',
             './ST_Breast/cropped_img']

save_roots = ['./ST_Breast/extracted_feature/1024',
              './ST_Breast/extracted_feature/512',
              './ST_Breast/extracted_feature/224']

for i in range(3):
    sample_folders = img_roots[i]
    save_sample_folders = save_roots[i]

    for tmp_sample in os.listdir(sample_folders):
        img_folders = os.path.join(sample_folders, tmp_sample)
        save_folders = os.path.join(save_sample_folders, tmp_sample)
        if not os.path.exists(save_folders):
            os.mkdir(save_folders)

        for save_img in os.listdir(img_folders):
            save_dirs = os.path.join(save_folders, save_img)
            img_dirs = os.path.join(img_folders, save_img)

            save_paths = os.path.join(save_folders, f'{save_img}.npy')
            extract_image_features(img_dirs, save_paths)
