import torch
from dataloader_3d import ImageGraphDataset, collate_fn, create_dataloaders_for_train_file, \
    create_dataloaders_for_test_file, create_dataloaders_for_result_file
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.graph_construction import edge_index_to_adj_matrix
from models.model_3d import ST_GTC_3d
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import pandas as pd


def split_and_calculate_metrics(predictions_list, ground_truths_list, indices_list, position_list, max_level=None):
    if max_level is None:
        max_level = max(torch.max(indices).item() for indices in indices_list)

    metrics_per_level = {}
    saved_predictions = {}

    for level in range(1, max_level + 1):
        level_predictions = []
        level_ground_truths = []
        level_position = []

        for predictions, ground_truths, indice, position in zip(predictions_list, ground_truths_list, indices_list, position_list):
            mask = indice == level
            level_preds = predictions[mask]
            level_gts = ground_truths[mask]
            level_pos = position[mask]

            if level_preds.numel() > 0:
                level_predictions.append(level_preds)
                level_ground_truths.append(level_gts)
                level_position.append(level_pos)

        if level_predictions:
            level_predictions = torch.cat(level_predictions)
            level_ground_truths = torch.cat(level_ground_truths)
            level_position = torch.cat(level_position)

            if level > 1:
                saved_predictions[level] = {
                    "predictions": level_predictions.tolist(),
                    "ground_truths": level_ground_truths.tolist(),
                    'position': level_position.tolist()
                }

                pred_array = level_predictions.cpu().numpy()
                gt_array = level_ground_truths.cpu().numpy()

                mse = mean_squared_error(gt_array, pred_array)
                mae = mean_absolute_error(gt_array, pred_array)
                pcc = calculate_pcc(torch.tensor(pred_array), torch.tensor(gt_array))

                metrics_per_level[level] = {
                    "MSE": mse,
                    "MAE": mae,
                    "PCC": np.array(pcc)
                }

    return metrics_per_level, saved_predictions


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./Our_data_format/HER2_3D',
                    help='Name of Experiment')
parser.add_argument('--max_iterations', type=int, default=25000, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float, default=0.0005, help='maximum epoch number to train')
parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--batch_size', type=int, default=128, help='repeat')
parser.add_argument('--rate', type=int, default=10, help='downsample rate')

args = parser.parse_args()

root = args.root_path
label_root = os.path.join(root, '3D_npy_information')
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

    model = ST_GTC_3d().cuda()

    img_transform = transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                        torchvision.transforms.RandomVerticalFlip(),
                                        torchvision.transforms.RandomApply(
                                            [torchvision.transforms.RandomRotation((90, 90))]),
                                        torchvision.transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])

    mse_loss_fn = nn.MSELoss()

    cross_mse, cross_mae, cross_pcc_patch, = [], [], []

    image_dict = {}

    for image in image_folder_paths:
        sample_name = image[0]
        if sample_name not in image_dict:
            image_dict[sample_name] = []
        image_dict[sample_name].append(image)

    fold_splits = [
        {'A', 'E'},
        {'B', 'F'},
        {'C', 'G'},
        {'D', 'H'},
    ]

    results = []
    for fold_index, fold_split in enumerate(fold_splits):
        train_labels_path = []
        val_labels_path = []

        print(f"\nFold {fold_index + 1}:")

        for sample, images in image_dict.items():
            print(sample, images)
            if sample not in fold_split:
                train_labels_path.append(os.path.join(label_root, images[0]))
            else:
                val_labels = os.path.join(label_root, images[0])
                val_labels_path.append(val_labels)

        part_trainloader, test_dataloader = create_dataloaders_for_result_file(val_labels_path, batch_size=128,
                                                                               transform=img_transform)

        checkpoint_path = f'./weight/Her2/model_best_{fold_index}.pth'  # 你的权重文件路径
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        model.cuda()

        model.eval()

        with torch.no_grad():
            for file_name, tmp_loader in test_dataloader.items():
                print(f"Processing file: {file_name}")
                tmp_name = file_name.split('/')[-1][:-4]

                tmp_prediction = []
                tmp_gt = []
                tmp_position = []
                tmp_level = []

                for images, labels, adj_sub, positions, idx_1024, idx_512, features_1024, features_512, graph_torch, known_label, indices, levels in tmp_loader:
                    graph_512 = graph_torch['layer_512']
                    graph_1024 = graph_torch['layer_1024']

                    batch_data = [
                        images, adj_sub, features_512, features_1024,
                        graph_512.x, graph_1024.x,
                        edge_index_to_adj_matrix(graph_512), edge_index_to_adj_matrix(graph_1024),
                        known_label
                    ]

                    batch_data = [data.to(device) for data in batch_data]

                    pred, b, c, = model(*batch_data)

                    tmp_prediction.append(pred)
                    tmp_gt.append(labels)
                    tmp_level.append(levels)
                    tmp_position.append(positions)

                metrics_per_level, saved_predictions = split_and_calculate_metrics(tmp_prediction, tmp_gt, tmp_level,
                                                                                   tmp_position)
                np.save(f'./weight/Her2/{tmp_name}.npy', np.array(saved_predictions))
                results.append(metrics_per_level)
    summary_data = []

