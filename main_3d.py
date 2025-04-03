import torch
from dataloader_3d import ImageGraphDataset, collate_fn, create_dataloaders_for_train_file, \
    create_dataloaders_for_result_file
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
import pandas as pd


warnings.filterwarnings("ignore", category=UserWarning,
                        message="Creating a tensor from a list of numpy.ndarrays is extremely slow")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message="You are using `torch.load` with `weights_only=False`")

# ./HER2_3D
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./ST_Breast_3D',
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


    save_path = './weight/ours'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.basicConfig(filename=save_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    img_transform = transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                        torchvision.transforms.RandomVerticalFlip(),
                                        torchvision.transforms.RandomApply(
                                            [torchvision.transforms.RandomRotation((90, 90))]),
                                        torchvision.transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])

    mse_loss_fn = nn.MSELoss()

    cross_mse, cross_mae, cross_pcc_patch, = [], [], []

    mse_metrics = []
    mae_metrics = []
    pcc_metrics = []
    save_dict = {}

    # Here is cross validation code for ST data
    all_folders = sorted([os.path.join(label_root, folder) for folder in os.listdir(label_root)])
    num_folders = len(all_folders)
    fold_size = num_folders // 4

    for fold in range(4):
        start = fold * fold_size
        end = start + fold_size if fold < 4 - 1 else num_folders  # Handle remaining folders in the last fold

        # Divide train set and test set
        val_labels_path = all_folders[start:end]
        train_labels_path = [folder for folder in all_folders if folder not in val_labels_path]
        print(train_labels_path, val_labels_path)

        train_dataloaders = create_dataloaders_for_train_file(train_labels_path, batch_size=128,
                                                              transform=img_transform)
        part_trainloader, test_dataloader = create_dataloaders_for_result_file(val_labels_path, batch_size=128,
                                                                             transform=img_transform)
        train_dataloaders.update(part_trainloader)

    # # Here is cross validation code for HER2
    # image_dict = {}
    #
    # # According to image naming conventions (e.g. A1.npy, A2.npy, B1.npy...) Parse the picture
    # for image in image_folder_paths:
    #     sample_name = image[0]
    #     if sample_name not in image_dict:
    #         image_dict[sample_name] = []
    #     image_dict[sample_name].append(image)
    # fold_splits = [
    #     {'A', 'E'},
    #     {'B', 'F'},
    #     {'C', 'G'},
    #     {'D', 'H'},
    # ]
    # results = []
    # for fold_index, fold_split in enumerate(fold_splits):
    #
    #     train_labels_path = []
    #     val_labels_path = []
    #
    #     print(f"\nFold {fold_index + 1}:")
    #
    #     for sample, images in image_dict.items():
    #         print(sample, images)
    #         if sample not in fold_split:
    #             train_labels_path.append(os.path.join(label_root, images[0]))
    #         else:
    #             val_labels = os.path.join(label_root, images[0])
    #             val_labels_path.append(val_labels)
    #
    #     train_dataloaders = create_dataloaders_for_train_file(train_labels_path, batch_size=128,
    #                                                           transform=img_transform)
    #     part_trainloader, test_dataloader = create_dataloaders_for_result_file(val_labels_path, batch_size=128,
    #                                                                            transform=img_transform)
    #     train_dataloaders.update(part_trainloader)

        iter_num = 0
        max_epoch = 51
        lr_ = base_lr
        model = ST_GTC_3d().cuda()

        writer = SummaryWriter(save_path + '/log')

        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
        model.cuda()

        for epoch_num in tqdm(range(max_epoch), ncols=70):
            model.train()

            file_names = list(train_dataloaders.keys())
            random.shuffle(file_names)

            for file_name in file_names:
                tmp_loader = train_dataloaders[file_name]

                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr * (1 - float(iter_num) / max_iterations) ** lr_decay

                print(f"Processing file: {file_name}")
                tmp_name = file_name.split('/')[-1][:-4]

                for images, labels, adj_sub, adj_propagate, positions, idx_1024, idx_512, features_1024, features_512, graph_torch, known_label in tmp_loader:
                    graph_512 = graph_torch['layer_512']
                    graph_1024 = graph_torch['layer_1024']

                    batch_data = [
                        images, adj_sub, adj_propagate, features_512, features_1024,
                        graph_512.x, graph_1024.x,
                        edge_index_to_adj_matrix(graph_512), edge_index_to_adj_matrix(graph_1024),
                        known_label
                    ]

                    batch_data = [data.to(device) for data in batch_data]

                    pred, pred_512, pred_1024 = model(*batch_data)
                    pred_512_con = pred_512.cpu()[idx_512.cpu()]
                    pred_1024_con = pred_1024.cpu()[idx_1024.cpu()]

                    loss_other_layer_predict = pcc_loss(graph_512.y.cuda(), pred_512) + pcc_loss(graph_1024.y.cuda(), pred_1024)
                    loss_consistency = pcc_loss(pred_512_con.cuda(), labels.cuda()) + pcc_loss(pred_1024_con.cuda(), labels.cuda())
                    loss_spot_predict = 0.75 * mse_loss_fn(pred, labels.cuda()) + 0.25 * pcc_loss(pred, labels.cuda())

                    loss = loss_spot_predict + 0.125 * loss_other_layer_predict + 0.125 * loss_consistency

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    iter_num = iter_num + 1
                    writer.add_scalar('lr', lr_, iter_num)
                    writer.add_scalar('loss/loss', loss, iter_num)

                    logging.info(
                        'iteration %d :loss: %5f, lr: %f5' %
                        (iter_num, loss,
                         optimizer.param_groups[0]['lr']))

                    if iter_num >= max_iterations:
                        break
                    time1 = time.time()

            if epoch_num % 10 == 0:
                model.eval()
                with torch.no_grad():
                    for file_name, tmp_loader in test_dataloader.items():
                        print(f"Processing file: {file_name}")
                        tmp_name = file_name.split('/')[-1][:-4]
                        val_loss = 0.0
                        val_pcc = 0.0
                        val_mae = 0.0
                        total_samples = 0

                        for images, labels, adj_sub, adj_propagate, positions, idx_1024, idx_512, features_1024, features_512, graph_torch, known_label, indices, _ in tmp_loader:
                            graph_512 = graph_torch['layer_512']
                            graph_1024 = graph_torch['layer_1024']

                            batch_data = [
                                images, adj_sub, adj_propagate, features_512, features_1024,
                                graph_512.x, graph_1024.x,
                                edge_index_to_adj_matrix(graph_512), edge_index_to_adj_matrix(graph_1024),
                                known_label
                            ]
                            batch_data = [data.to(device) for data in batch_data]
                            pred, b, c, = model(*batch_data)
                            loss = mse_loss_fn(pred[indices], labels[indices].cuda())
                            val_loss += loss.item() * len(indices)
                            total_samples += len(indices)
                            val_mae += np.mean(
                                np.abs(pred[indices].cpu().numpy() - labels[indices].cpu().numpy())) * len(
                                indices)
                            val_pcc += calculate_pcc(pred[indices], labels[indices].cuda()) * len(indices)
                            print(f'mse {loss}, pcc {calculate_pcc(pred[indices], labels[indices].cuda())}')

            if (epoch_num+1) % max_epoch == 0:
                model.eval()
                with torch.no_grad():
                    for file_name, tmp_loader in test_dataloader.items():
                        print(f"Processing file: {file_name}")
                        tmp_name = file_name.split('/')[-1][:-4]
                        val_loss = 0.0
                        val_pcc = 0.0
                        val_mae = 0.0
                        total_samples = 0

                        for images, labels, adj_sub, adj_propagate, positions, idx_1024, idx_512, features_1024, features_512, graph_torch, known_label, indices, _ in tmp_loader:
                            graph_512 = graph_torch['layer_512']
                            graph_1024 = graph_torch['layer_1024']

                            batch_data = [
                                images, adj_sub, adj_propagate, features_512, features_1024,
                                graph_512.x, graph_1024.x,
                                edge_index_to_adj_matrix(graph_512), edge_index_to_adj_matrix(graph_1024),
                                known_label
                            ]

                            batch_data = [data.to(device) for data in batch_data]

                            pred, b, c, = model(*batch_data)

                            loss = mse_loss_fn(pred[indices], labels[indices].cuda())
                            val_loss += loss.item() * len(indices)
                            total_samples += len(indices)
                            val_mae += np.mean(
                                np.abs(pred[indices].cpu().numpy() - labels[indices].cpu().numpy())) * len(
                                indices)
                            val_pcc += calculate_pcc(pred[indices], labels[indices].cuda()) * len(indices)
                            print(f'mse {loss}, pcc {calculate_pcc(pred[indices], labels[indices].cuda())}')

                        mse_metrics.append(val_loss / total_samples)
                        mae_metrics.append(val_mae / total_samples)
                        pcc_metrics.append(val_pcc.cpu().numpy() / total_samples)
                        torch.save(model.state_dict(), os.path.join(save_path, 'model_weights.pth'))

            save_dict['mse'] = mse_metrics
            save_dict['mae'] = mae_metrics
            save_dict['pcc'] = pcc_metrics
            print(save_dict)

            df = pd.DataFrame(save_dict)
            output_file = os.path.join(save_path, f'ours_metrics.xlsx')
            df.to_excel(output_file, index=False)

            print(f"Results have been saved to {output_file}")


