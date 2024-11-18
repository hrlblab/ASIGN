import torch
from dataloader_3d import ImageGraphDataset, collate_fn, create_dataloaders_for_train_file, \
    create_dataloaders_for_test_file
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

warnings.filterwarnings("ignore", category=UserWarning,
                        message="Creating a tensor from a list of numpy.ndarrays is extremely slow")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message="You are using `torch.load` with `weights_only=False`")

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Our_data_format/HER2_3D',
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


    save_path = './weight/ours'

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

        train_dataloaders = create_dataloaders_for_train_file(train_labels_path, batch_size=128,
                                                              transform=img_transform)
        part_trainloader, test_dataloader = create_dataloaders_for_test_file(val_labels_path, batch_size=128,
                                                                             transform=img_transform)
        train_dataloaders.update(part_trainloader)

        iter_num = 0
        max_epoch = 40
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

                for images, labels, adj_sub, positions, idx_1024, idx_512, features_1024, features_512, graph_torch, known_label in tmp_loader:
                    graph_512 = graph_torch['layer_512']
                    graph_1024 = graph_torch['layer_1024']

                    batch_data = [
                        images, adj_sub, features_512, features_1024,
                        graph_512.x, graph_1024.x,
                        edge_index_to_adj_matrix(graph_512), edge_index_to_adj_matrix(graph_1024),
                        known_label
                    ]

                    batch_data = [data.to(device) for data in batch_data]

                    pred, pred_512, pred_1024 = model(*batch_data)
                    pred_512_con = pred_512.cpu()[idx_512.cpu()]
                    pred_1024_con = pred_1024.cpu()[idx_1024.cpu()]

                    loss_other_layer = pcc_loss(graph_512.y.cuda(), pred_512) + pcc_loss(graph_1024.y.cuda(), pred_1024)

                    loss_consistency = pcc_loss(pred_512_con.cuda(), labels.cuda()) + pcc_loss(pred_1024_con.cuda(),
                                                                                               labels.cuda())

                    loss = 1.5 * mse_loss_fn(pred, labels.cuda()) + 0.25 * pcc_loss(pred, labels.cuda()) + 0 * loss_other_layer + 0 * loss_consistency

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

                        for images, labels, adj_sub, positions, idx_1024, idx_512, features_1024, features_512, graph_torch, known_label, indices in tmp_loader:
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

                            loss = mse_loss_fn(pred[indices], labels[indices].cuda())
                            val_loss += loss.item() * len(indices)
                            total_samples += len(indices)

                            """
                            Here is pcc for all, 50H, 50V                 
                            """
                            # mean_pcc_per_patch += calculate_overall_pcc(outputs.cpu().numpy(), labels)
                            mae += np.mean(np.abs(pred[indices].cpu().numpy() - labels[indices].cpu().numpy())) * len(
                                indices)
                            val_pcc += calculate_pcc(pred[indices], labels[indices].cuda()) * len(indices)
                            print(f'mse {loss}, pcc {calculate_pcc(pred[indices], labels[indices].cuda())}')

                    if val_loss / total_samples < best_mse:
                        best_mse = val_loss / total_samples
                        best_mae = mae / total_samples
                        best_pcc = val_pcc / total_samples

                        torch.save(model.state_dict(), os.path.join(save_path, f"model_best_{fold_index}.pth"))
                        print(f'Best model saved to {os.path.join(save_path, f"model_best_{fold_index}.pth")}')

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
                np.array(metrics_result))

        with open(os.path.join(save_path, '{}_validation_metrics.txt'.format(data_name)), 'w') as file:
            for key, values in validation_metrics.items():
                file.write(f"{key}: {values}\n")
