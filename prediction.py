import torch
from dataloader import ImageGraphDataset, collate_fn, create_dataloaders_for_each_file
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.graph_construction import edge_index_to_adj_matrix
from models.model import ST_GTC
import warnings
import os
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import random
import torchvision
import logging
import sys
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import time
from models.losses import calculate_pcc, pcc_loss
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, message="Creating a tensor from a list of numpy.ndarrays is extremely slow")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Our_data_format/HER2',
                    help='Name of Experiment')
parser.add_argument('--max_iterations', type=int, default=25000, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float, default=0.001, help='maximum epoch number to train')
parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--batch_size', type=int, default=128, help='repeat')
parser.add_argument('--rate', type=int, default=10, help='downsample rate')

args = parser.parse_args()

root = args.root_path
label_root = os.path.join(root, 'npy_information')
image_folder_paths = sorted([f for f in os.listdir(label_root) if f.endswith('.npy')])
data_name = args.root_path.split('/')[-1]

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
max_iterations = args.max_iterations
base_lr = args.base_lr
lr_decay = args.lr_decay
loss_record = 0
batch_size = args.batch_size
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    save_path = './weight/Her2'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.basicConfig(filename=save_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = ST_GTC().cuda()

    img_transform = transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                        torchvision.transforms.RandomVerticalFlip(),
                                        torchvision.transforms.RandomApply(
                                            [torchvision.transforms.RandomRotation((90, 90))]),
                                        torchvision.transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
    mse_loss_fn = nn.MSELoss()

    mse_metrics = []
    mae_metrics = []
    pcc_metrics = []
    save_dict = {}

    image_dict = {}

    for image in image_folder_paths:
        sample_name = image[0]
        if sample_name not in image_dict:
            image_dict[sample_name] = []
        image_dict[sample_name].append(image)

    samples = {
        'E': 3, 'F': 3, 'G': 3, 'H': 3,
        'A': 6, 'B': 6, 'C': 6, 'D': 6
    }

    fold_splits = [
        {
            'E': list(range(1, 3))
        },
    ]

    for fold_index, fold_split in enumerate(fold_splits):
        train_labels_path = []
        val_labels_path = []

        print(f"\nFold {fold_index + 1}:")

        # 遍历每个样本
        for sample, images in image_dict.items():
            if sample not in fold_split:
                print(f"Training set paths: {images}")
                train_labels_path.extend(os.path.join(label_root, images[idx]) for idx in range(len(images)))
            else:
                print(f"Validation set paths: {images}")
                val_indices = fold_split[sample]
                val_labels = [os.path.join(label_root, images[idx]) for idx in val_indices]
                val_labels_path.extend(val_labels)
                val_label = os.path.join(label_root, images[0])

        val_loader = create_dataloaders_for_each_file(val_labels_path, batch_size=128, transform=img_transform)

        checkpoint_path = f'./weight/Her2/model_best_{0}.pth'
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        model.cuda()

        with torch.no_grad():
            for file_name, tmp_loader in val_loader.items():
                print(f"Processing file: {file_name}")

                tmp_name = file_name.split('/')[-1][:-4]
                mae = 0
                mean_pcc_per_patch = 0
                val_loss = 0
                total_samples = 0
                final_save = {}
                output_prediction = []
                position = []
                gt = []
                tmp_prediction = []

                for images, labels, adj_sub, positions, idx_1024, idx_512, features_1024, features_512, graph_torch in tmp_loader:
                    graph_512 = graph_torch['layer_512']
                    graph_1024 = graph_torch['layer_1024']

                    batch_data = [
                        images, adj_sub, features_512, features_1024,
                        graph_512.x, graph_1024.x,
                        edge_index_to_adj_matrix(graph_512), edge_index_to_adj_matrix(graph_1024),
                        idx_1024, idx_512
                    ]

                    batch_data = [data.to(device) for data in batch_data]
                    outputs, b, c, = model(*batch_data)
                    tmp_prediction.append(outputs)
                tmp_prediction = torch.cat(tmp_prediction, dim=0).cpu().numpy()
                print(tmp_prediction.shape)
                save_name = f'./final/{tmp_name}.npy'
                np.save(save_name, np.array(tmp_prediction))