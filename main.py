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


warnings.filterwarnings("ignore", category=UserWarning, message="Creating a tensor from a list of numpy.ndarrays is extremely slow")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Our_data_format/ST_Breast',
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

    save_path = './weight/ours_st_breast_3'

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

    cross_mse, cross_mae, cross_pcc_patch, = [], [], []

    image_dict = {}

    all_folders = sorted([folder for folder in os.listdir(label_root) if os.path.isdir(os.path.join(label_root, folder))])
    num_folders = len(all_folders)
    fold_size = num_folders // 4
    print(all_folders)

    for fold in range(4):
        start = fold * fold_size
        end = start + fold_size if fold < 4 - 1 else num_folders  # Handle remaining folders in the last fold

        val_folders = all_folders[start:end]
        train_folders = [folder for folder in all_folders if folder not in val_folders]

        train_files = []

        for folder in train_folders:
            folder_path = os.path.join(label_root, folder)
            train_files.extend([os.path.join(folder_path, f'{i}.npy') for i in range(1, 4)])

        val_files = []
        for folder in val_folders:
            folder_path = os.path.join(label_root, folder)
            val_files.extend([os.path.join(folder_path, f'{i}.npy') for i in range(2, 4)])
            train_files.append(os.path.join(folder_path, '1.npy'))

        train_dataloaders = create_dataloaders_for_each_file(train_files, batch_size=130, transform=img_transform)
        test_dataloader = create_dataloaders_for_each_file(val_files, batch_size=130, transform=img_transform)

        iter_num = 0
        max_epoch = 100
        lr_ = base_lr

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

                    pred, pred_512, pred_1024 = model(*batch_data)
                    pred_512_con = pred_512.cpu()[idx_512.cpu()]
                    pred_1024_con = pred_1024.cpu()[idx_1024.cpu()]

                    loss_other_layer = pcc_loss(graph_512.y.cuda(), pred_512) + pcc_loss(graph_1024.y.cuda(), pred_1024)

                    loss_consistency = pcc_loss(pred_512_con.cuda(), labels.cuda()) + pcc_loss(pred_1024_con.cuda(), labels.cuda())

                    loss = 0.75*mse_loss_fn(pred, labels.cuda()) + 0.1*pcc_loss(pred, labels.cuda()) + 0.1 * loss_other_layer + 0.1 * loss_consistency

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

            if epoch_num % 1 == 0:
                model.eval()
                val_loss = 0.0
                val_pcc = 0.0
                val_mae = 0.0
                total_samples = 0

                best_mse = 10000

                pcc_per_row, pcc_per_row_h50, pcc_per_row_v50 = 0, 0, 0
                mean_pcc_per_patch, mean_pcc_per_patch_h50, mean_pcc_per_patch_v50 = 0, 0, 0
                mae = 0

                with torch.no_grad():
                    for file_name, tmp_loader in test_dataloader.items():
                        print(f"Processing file: {file_name}")
                        tmp_name = file_name.split('/')[-1][:-4]

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

                            pred, b, c, = model(*batch_data)

                            loss = mse_loss_fn(pred, labels.cuda())
                            val_loss += loss.item() * images.size(0)
                            total_samples += images.size(0)

                            """
                            Here is pcc for all, 50H, 50V                 
                            """
                            # mean_pcc_per_patch += calculate_overall_pcc(outputs.cpu().numpy(), labels)
                            mae += np.mean(np.abs(pred.cpu().numpy() - labels.cpu().numpy())) * images.size(0)
                            val_pcc += calculate_pcc(pred, labels.cuda()) * images.size(0)
                            print(f'mse {loss}, pcc {calculate_pcc(pred, labels.cuda())}')

                    if val_loss / total_samples < best_mse:
                        best_mse = val_loss / total_samples
                        best_mae = mae / total_samples
                        best_pcc = val_pcc / total_samples

                        torch.save(model.state_dict(), os.path.join(save_path, f"model_best_{fold}.pth"))
                        print(f'Best model saved to {os.path.join(save_path, f"model_best_{fold}.pth")}')

                    print(f"Validation MSE Loss: {val_loss / total_samples}")
                    print(f"Mean Absolute Error (MAE) with NumPy:{mae / total_samples}")
                    print(f"Validation mean pcc per patch: {val_pcc / total_samples}")

        cross_mse.append(best_mse)
        cross_mae.append(best_mae)
        cross_pcc_patch.append(best_pcc)

        metrics_result = []

        validation_metrics = {
            'cross_mse': cross_mse,
            'cross_mae': cross_mae,
            'cross_pcc_patch': cross_pcc_patch,
        }
        metrics_result.append(validation_metrics)

        np.save(os.path.join(save_path, '{}_validation_metrics.npy'.format(data_name)),
                np.array(metrics_result))  # Save as a binary file in NumPy .npy format

        with open(os.path.join(save_path, '{}_validation_metrics.txt'.format(data_name)), 'w') as file:
            for key, values in validation_metrics.items():
                file.write(f"{key}: {values}\n")