import argparse
import os.path
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from scipy import ndimage
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader_land_fetal import FetalDataset
from dataloader_land_knee import KneeDataset
from dataloader_seg_fetal import FetalSegData
from dataloader_seg_knee import KneeSegDataset

from networks.sdmt import SDMT
from networks.TransDoD.TransDoDNet import TransDoDNet
from networks.UniSeg.Prompt_generic_UNet import UniSeg_model
from networks.dodnet import UNet3D
from networks.mtnet import MTNetGN
from networks.network import ResUnetMT
from networks.swinunetr_3d import SwinUNETR_multitask


def extract_boundary_3d(seg_mask, kernel_size=3, erosion_kernel_size=2):
    # Ensure the mask is a PyTorch tensor
    if not isinstance(seg_mask, torch.Tensor):
        seg_mask = torch.from_numpy(seg_mask)

    # Convert to float and binarize the mask
    seg_mask = (seg_mask > 0).float()

    # Create a cuboid kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size))
    if seg_mask.is_cuda:
        kernel = kernel.cuda()

    # Perform dilation
    dilated = F.conv3d(seg_mask, kernel, padding=kernel_size // 2) > 0

    # Perform erosion
    eroded = F.conv3d(1.0 - seg_mask, kernel, padding=kernel_size // 2) > 0

    # Subtract the eroded mask from the dilated mask to get the boundary
    boundary = dilated.int() - (1 - eroded.int())

    # Create a cuboid kernel for erosion
    erosion_kernel = torch.ones((1, 1, erosion_kernel_size, erosion_kernel_size, erosion_kernel_size))
    if seg_mask.is_cuda:
        erosion_kernel = erosion_kernel.cuda()

    # Perform erosion on the boundary to make it thinner
    eroded_boundary = F.conv3d(1.0 - boundary.float(), erosion_kernel, padding=erosion_kernel_size // 2) > 0

    # Convert the eroded boundary back to the original binary form
    thin_boundary = 1 - eroded_boundary.int()

    # Slice the tensor to match the original shape
    thin_boundary = thin_boundary[..., :seg_mask.shape[2], :seg_mask.shape[3], :seg_mask.shape[4]]

    return thin_boundary


def compute_distance_map(boundary, max_dist=10):
    # Create an initial distance map filled with max_dist where boundary is not present
    dist_map = torch.full_like(boundary, max_dist)
    dist_map[boundary > 0] = 0

    # Create a cubic dilation kernel
    kernel = torch.ones((1, 1, 3, 3, 3), device=boundary.device)

    # Iteratively dilate the boundary and update the distance map
    for distance in range(1, max_dist + 1):
        dilated_boundary = F.conv3d(boundary.float(), kernel, padding=1) > 0
        # Update the distance map where the dilated boundary "touches" it and it's less than current distance
        update_mask = (dilated_boundary > 0) & (dist_map > distance)
        dist_map[update_mask] = distance

        # Update boundary for the next dilation
        boundary = dilated_boundary.float()
    dist_map = (max_dist - dist_map) / max_dist
    return dist_map


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def distance_points(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2))


landmark_relations_feta = {
    0: {"relation": 'e', "segment": [1, 5]},
    1: {"relation": 'e', "segment": [1, 5]},
    2: {"relation": 'e', "segment": [1, 5]},
    3: {"relation": 'e', "segment": [1, 5]},
    4: {"relation": 'e', "segment": [4, 5]},
    5: {"relation": 'e', "segment": [1, 5]}
}

# landmark_relations_feta = {
#     0: {"relation": 'e', "segment": [5]},
#     1: {"relation": 'e', "segment": [5]},
#     2: {"relation": 'e', "segment": [5]},
#     3: {"relation": 'e', "segment": [5]},
#     4: {"relation": 'e', "segment": [5]},
#     5: {"relation": 'e', "segment": [5]}
# }

landmark_relations_knee = {
    0: {"relation": 'e', "segment": 1},
    1: {"relation": 'e', "segment": 3},
    2: {"relation": 'e', "segment": 1},
    3: {"relation": 'e', "segment": 1},
    4: {"relation": 'e', "segment": 3},
    5: {"relation": 'e', "segment": 1},
    6: {"relation": 'e', "segment": 3},
    7: {"relation": 'e', "segment": [1, 2]},
    8: {"relation": 'e', "segment": [2]},
    9: {"relation": 'e', "segment": [1, 2]},
    10: {"relation": 'e', "segment": [2]},
    11: {"relation": 'e', "segment": [3, 4]},
    12: {"relation": 'e', "segment": [3, 5]}
}


def generate_gaussian_heatmap(a_shape, points, sigma=5.0, scaling=255):
    batch_size, numbers, x, y, z = a_shape
    grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(x), torch.arange(y), torch.arange(z), indexing='ij')
    heatmap = torch.zeros((batch_size, numbers, x, y, z))

    for b in range(batch_size):
        for n in range(numbers):
            point = points[b, n]
            distance = (grid_x - point[0]) ** 2 + (grid_y - point[1]) ** 2 + (grid_z - point[2]) ** 2
            heatmap[b, n] = torch.exp(-distance / (2 * sigma ** 2))
    heatmap = (heatmap - torch.min(heatmap)) / (torch.max(heatmap) - torch.min(heatmap)) * scaling

    return heatmap


def get_points_from_heatmap_torch(heatmap):
    max_indices = torch.argmax(heatmap.view(heatmap.size(0), heatmap.size(1), -1), dim=2)
    z_indices = max_indices % heatmap.size(4)
    y_indices = (max_indices // heatmap.size(4)) % heatmap.size(3)
    x_indices = max_indices // (heatmap.size(3) * heatmap.size(4))
    return torch.stack([x_indices, y_indices, z_indices], dim=2)


def bab_loss(out_seg, out_landmark, landmark_relations, max_dist=10):
    consistency_loss = torch.tensor(0, dtype=torch.float32).cuda()
    for b in range(out_seg.shape[0]):
        dist_maps = []
        # pre-calculate the dist_maps for each segment
        for seg_idx in range(out_seg.shape[1]):
            # print(out_seg.grad)
            out_seg_on_segment = out_seg[b, seg_idx, :, :, :]
            # start_time = time.time()
            out_seg_no_grad = out_seg_on_segment.clone()
            binary_seg = (out_seg_no_grad.detach().cpu().numpy())
            labels, num = ndimage.label(binary_seg)
            # print(time.time()-start_time)

            if num != 0:
                max_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
                # max_region = out_seg_on_segment[(labels == max_label)]
                binary_mask = torch.zeros_like(out_seg_on_segment).cuda()
                binary_mask[(labels == max_label)] = 1

                max_region = out_seg_on_segment * binary_mask
                seg_mask = max_region[None, None]
                # Calculate the boundary
                boundary = extract_boundary_3d(seg_mask)

                # Calculate the distance map
                dist_map_nodiff = compute_distance_map(boundary, max_dist).cuda().squeeze(0).squeeze(0)
                # return
                dist_map = dist_map_nodiff * out_seg_on_segment * binary_mask + dist_map_nodiff * (
                        1 - out_seg_on_segment) * (1 - binary_mask)
            else:
                dist_map = None
            dist_maps.append(dist_map)

        for land_idx in range(len(landmark_relations.keys())):
            seg_indices = landmark_relations[land_idx]["segment"]
            if type(seg_indices) is not list:
                seg_indices = [seg_indices]
            for seg_idx in seg_indices:
                dist_map = dist_maps[seg_idx]
                if dist_map is not None:
                    current_land = out_landmark[b, land_idx, :, :, :] / 255
                    simi = torch.sum(current_land * dist_map)
                    consistency_loss += 1 - simi / torch.count_nonzero(current_land)

    consistency_loss /= out_seg.shape[0]
    consistency_loss /= out_landmark.shape[1]
    return consistency_loss


def dyn_mt_loss(index, loss_heatmap, loss_seg, lambda_weight, avg_cost, T=2):
    if index == 0:
        lambda_weight[:, index] = 1.0
    else:
        w_1 = loss_heatmap.item() / avg_cost[index - 1, 0]
        w_2 = loss_seg.item() / avg_cost[index - 1, 1]
        lambda_weight[0, index] = np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
        lambda_weight[1, index] = np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))

    avg_cost[index, 0] = loss_heatmap.item()
    avg_cost[index, 1] = loss_seg.item()

    loss = lambda_weight[0, index] * loss_heatmap + lambda_weight[1, index] * loss_seg
    return loss, lambda_weight, avg_cost


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate_model(args, model, valid_loader, eval_flag, num_points, num_segments):
    """
    eval flag: l: landmark, s: segmentation, mt: both l and s
    """

    distance_losses = []
    model.eval()

    if eval_flag == 'l':
        for data in valid_loader:
            voxel, gt, case_info = data
            factor = case_info['factor'].unsqueeze(1).float()
            spacing = case_info['spacing'].unsqueeze(1).float()
            voxel = voxel.cuda().unsqueeze(1).float()
            if "mt" in args.model_name or 'SDMT' in args.model_name:
                pseudo_seg, heatmap = model(voxel)
            elif 'dodnet' in args.model_name:
                cond = torch.ones((voxel.shape[0], 1)).long().cuda()
                heatmap = model(voxel, cond)
            elif 'uniseg' in args.model_name:
                cond = torch.ones((voxel.shape[0], 1)).long().cuda()
                heatmap = model(voxel, cond)
            elif 'transdod' in args.model_name:
                cond = torch.ones((voxel.shape[0], 1)).long().cuda()
                heatmap = model(voxel, cond)
                N_b, N_q, N_c, N_d, N_h, N_w = heatmap.shape
                preds_convert = torch.zeros(size=(N_b, N_c, N_d, N_h, N_w)).cuda()
                for i_b in range(N_b):
                    preds_convert[i_b] = heatmap[i_b, int(cond[i_b])]
                heatmap = preds_convert
            else:
                heatmap = model(voxel)

            pred_3d_coor = get_points_from_heatmap_torch(heatmap[:, :num_points]).float().detach().cpu().numpy()
            gt = gt.float()

            pred_3d_coor /= (factor / spacing)
            gt /= (factor / spacing)

            distance_losses_tensor = torch.sqrt(torch.sum((pred_3d_coor - gt) ** 2, dim=-1))
            distance_losses.append(distance_losses_tensor)

        distance_array = torch.cat(distance_losses, dim=0)
        distance_array = np.array(distance_array)
        # print(distance_array)
        # with open('best_test.csv', 'w') as f:
        #     np.savetxt(f, distance_array, delimiter=',', fmt='%d')
        distance_category = np.mean(distance_array, axis=1)
        distance_all = np.mean(distance_category).item()

        print('dist_c:', distance_category)
        print('dist_all:', distance_all)
        return round(distance_all, 2), distance_category
    elif eval_flag == 's':
        metric_list = 0.0
        metric_list1 = 0.0
        metric_list2 = 0.0
        for i_batch, sampled_batch in enumerate(valid_loader):
            volume_batch, label = sampled_batch['image'], sampled_batch['mask']
            label = label.squeeze(1).cuda()
            with torch.no_grad():
                if "mt" in args.model_name or 'SDMT' in args.model_name:
                    out, heatmap = model(volume_batch.cuda())
                elif 'dodnet' in args.model_name:
                    cond = torch.zeros((volume_batch.cuda().shape[0], 1)).long().cuda()
                    out = model(volume_batch.cuda(), cond)[:, :num_segments]
                elif 'uniseg' in args.model_name:
                    cond = torch.zeros((volume_batch.cuda().shape[0], 1)).long().cuda()
                    out = model(volume_batch.cuda(), cond)[:, :num_segments]
                elif 'transdod' in args.model_name:
                    cond = torch.zeros((volume_batch.cuda().shape[0], 1)).long().cuda()
                    out = model(volume_batch.cuda(), cond)
                    N_b, N_q, N_c, N_d, N_h, N_w = out.shape
                    preds_convert = torch.zeros(size=(N_b, N_c, N_d, N_h, N_w)).cuda()
                    for i_b in range(N_b):
                        preds_convert[i_b] = out[i_b, int(cond[i_b])]
                    out = preds_convert[:, :num_segments]
                else:
                    out = model(volume_batch.cuda())
                num_classes = out.shape[1]
                out = torch.argmax(torch.softmax(out, dim=1), dim=1)
            metric_i = torchmetrics.functional.dice(out, label, average='none', num_classes=num_classes)[1:]

            # with open(test_save_path + '/results.txt', 'a') as f:
            #     f.write('test_case: ' + str(i_batch) + '\n')
            #     f.write('mean_dice: ' + str(metric_i.mean().item()) + '\n')
            # print('test_case %d : mean_dice : %f' % (i_batch, metric_i.mean()))
            metric_list += metric_i
            if i_batch < 15:
                metric_list1 += metric_i
            else:
                metric_list2 += metric_i

        metric_list = metric_list / len(valid_loader)
        performance = metric_list.mean() * 100
        return round(performance.item(), 2), round(performance.item(), 2)


def one_hot_encode(gt_seg):
    """
    One-hot encodes the input segmentation map.

    Parameters:
    - gt_seg (torch.Tensor): Input segmentation map with shape (2, 1, D, H, W),
                             where D, H, W are the depth, height, and width of the volume.

    Returns:
    - torch.Tensor: One-hot encoded segmentation map with shape (2, C, D, H, W),
                    where C is the number of classes (8 in this case).
    """
    # Ensure the input is a long tensor (contains integers)
    gt_seg = gt_seg.long()

    # Number of classes, assuming labels are from 0 to 7
    num_classes = 8

    # One-hot encode the input tensor
    gt_seg_one_hot = torch.nn.functional.one_hot(gt_seg, num_classes=num_classes)

    # Remove the singleton dimension and permute to get the correct shape
    gt_seg_one_hot = gt_seg_one_hot.squeeze(1).permute(0, 4, 1, 2, 3)

    return gt_seg_one_hot


def random_rotate_3d(image, heatmap):
    B, C, D, H, W = image.shape
    angles = np.radians(np.random.uniform(-5, 5, size=3))
    cos_vals, sin_vals = np.cos(angles), np.sin(angles)

    Rx = torch.tensor([[1, 0, 0, 0], [0, cos_vals[0], sin_vals[0], 0], [0, -sin_vals[0], cos_vals[0], 0], [0, 0, 0, 1]],
                      dtype=torch.float32, device=image.device)
    Ry = torch.tensor([[cos_vals[1], 0, -sin_vals[1], 0], [0, 1, 0, 0], [sin_vals[1], 0, cos_vals[1], 0], [0, 0, 0, 1]],
                      dtype=torch.float32, device=image.device)
    Rz = torch.tensor([[cos_vals[2], sin_vals[2], 0, 0], [-sin_vals[2], cos_vals[2], 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                      dtype=torch.float32, device=image.device)

    R = Rx @ Ry @ Rz

    R = R[:3, :4].unsqueeze(0).repeat(B, 1, 1)

    grid = F.affine_grid(R, size=image.size(), align_corners=False)
    rotated_image = F.grid_sample(image, grid, align_corners=False)
    rotated_heatmap = F.grid_sample(heatmap, grid, align_corners=False)

    return rotated_image, rotated_heatmap


# def adjust_learning_rate(optimizer, epoch, total_epochs=50, warmup_epochs=5, initial_lr=0.001, power=2.0, step_size=10):
#     """Adjusts learning rate using polynomial decay with step changes every specified number of epochs after warmup period"""
#     if epoch < warmup_epochs:
#         # Linear warmup of learning rate
#         lr = initial_lr * (epoch + 1) / warmup_epochs
#     else:
#         # Calculate the effective epoch for stepping every 'step_size' epochs
#         effective_epoch = (epoch - warmup_epochs) // step_size * step_size + warmup_epochs
#         # Polynomial decay of learning rate
#         lr = initial_lr * (1 - (effective_epoch - warmup_epochs) / (total_epochs - warmup_epochs)) ** power
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def train(args):
    setup_seed(1234)
    bs_half = int(args.batch_size / 2)
    save_path = 'runs/' + args.dataset + str(args.max_dist) + '-' + args.model_name + ' ' + args.loss_func + '/'
    os.makedirs(save_path, exist_ok=True)
    logger_file = open(save_path + 'log.txt', 'w')
    logger_loss = open(save_path + 'loss.txt', 'w')

    if args.dataset == 'feta':
        # load dataset_fetal_land for landmark detection
        train_set = FetalDataset(mode='train')
        valid_set = FetalDataset(mode='valid')
        test_set = FetalDataset(mode='test')

        train_loader = DataLoader(train_set, batch_size=bs_half, shuffle=True, pin_memory=True, drop_last=True)
        valid_loader_landmark = DataLoader(valid_set, batch_size=1, shuffle=False, pin_memory=True)
        test_loader_landmark = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

        # load dataset_fetal_land for semantic segmentation
        train_seg = FetalSegData(data_dir='./dataset_fetal_seg/atlas', mode="train", list_name='train.list')
        valid_seg = FetalSegData(data_dir='./dataset_fetal_seg/feta', mode="test", list_name='train.list')
        test_seg = FetalSegData(data_dir='./dataset_fetal_seg/feta', mode="test", list_name='test.list')

        train_loader_seg = DataLoader(train_seg, batch_size=bs_half, shuffle=True, pin_memory=True, drop_last=True)
        valid_loader_seg = DataLoader(valid_seg, batch_size=1, shuffle=False, pin_memory=True)
        test_loader_seg = DataLoader(test_seg, batch_size=1, shuffle=False, pin_memory=True)

        landmark_relations = landmark_relations_feta
        num_epochs = 80
        num_segments = 8
        num_points = 6
    else:
        train_set = KneeDataset(mode='train')
        valid_set = KneeDataset(mode='valid')
        test_set = KneeDataset(mode='test')

        train_loader = DataLoader(train_set, batch_size=bs_half, shuffle=True, drop_last=True)
        valid_loader_landmark = DataLoader(valid_set, batch_size=1, shuffle=False, drop_last=True)
        test_loader_landmark = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True)

        # load dataset_fetal_land for semantic segmentation
        train_seg = KneeSegDataset(mode='train')
        valid_seg = KneeSegDataset(mode="validation")
        test_seg = KneeSegDataset(mode="test")

        train_loader_seg = DataLoader(train_seg, batch_size=bs_half, shuffle=True, pin_memory=True, drop_last=True)
        valid_loader_seg = DataLoader(valid_seg, batch_size=1, shuffle=True, pin_memory=True)
        test_loader_seg = DataLoader(test_seg, batch_size=1, shuffle=False)

        landmark_relations = landmark_relations_knee
        num_epochs = 40
        num_segments = 6
        num_points = 13

    print('Length of land loader:', len(train_loader) * bs_half)
    print('Length of seg loader:', len(train_loader_seg) * bs_half)

    # multi-task
    if args.model_name == 'swinunetr-mt':
        model = SwinUNETR_multitask(out_channels_seg=num_segments, out_channels_landmark=num_points).cuda()
    elif args.model_name == 'mtnet-gn':
        model = MTNetGN(seg_out_ch=num_segments, landmark_out_ch=num_points).cuda()
    elif args.model_name == 'mtnet':
        model = ResUnetMT(points=num_points, segments=num_segments).cuda()
        if 'bab-d' in args.loss_func:
            model.load_state_dict(
                torch.load('./runs/feta-mtnet/39-valid-0.6526-test-0.6581-valid-2.3312-test-2.2199.pth'))
    elif args.model_name == 'dodnet':
        model = UNet3D(num_classes=max(num_points, num_segments)).cuda()
    elif args.model_name == 'transdod':
        dep, wid = map(int, args.dyn_head_dep_wid.split(','))
        dyn_head_dep_wid = (dep, wid)
        model = TransDoDNet(args, norm_cfg='IN', activation_cfg='relu', num_classes=max(num_points, num_segments),
                            weight_std=False, deep_supervision=False, res_depth=args.res_depth,
                            dyn_head_dep_wid=dyn_head_dep_wid).cuda()
    elif args.model_name == 'uniseg':
        base_num_features = 32
        patch_size = [128, 128, 128]
        model = UniSeg_model(patch_size, 2, [1, 2, 4], base_num_features, max(num_segments, num_points)).cuda()
    elif args.model_name == 'SDMT':
        model = SDMT(num_segment=num_segments, num_landmark=num_points).cuda()
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    # optimizer = torch.optim.AdamW(model.parameters())
    # Set up the StepLR scheduler to decay the learning rate by a factor of 0.5 every 20 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    confidence_threshold = 0.9
    local_best = 0

    if 'dyn' in args.loss_func or 'SDMT' in args.model_name:
        lambda_weight = np.ones([2, num_epochs])
        avg_cost = np.zeros([num_epochs, 2], dtype=np.float32)

    for epoch in range(num_epochs):
        # adjust_learning_rate(optimizer, epoch)
        print('epoch:', epoch)
        total_train_loss = 0
        model.train()
        for i_batch, sampled_batch in tqdm(enumerate(zip(train_loader, train_loader_seg))):
            # print(i_batch)
            data_land, data_seg = sampled_batch[0], sampled_batch[1]
            voxel_landmark, gt, case_info = data_land
            if args.use_sam:
                land_pseudo_mask = case_info['mask'].cuda()
            voxel_seg, gt_seg = data_seg['image'].cuda(), data_seg['mask'].cuda()
            # voxel_seg, gt_seg = random_rotate_3d(voxel_seg, gt_seg.float())

            voxel_landmark = voxel_landmark.cuda().unsqueeze(1)
            voxel_fuse = torch.cat((voxel_landmark, voxel_seg))  # landmark before, seg after
            gt_heatmap = generate_gaussian_heatmap((voxel_landmark.shape[0], num_points, 128, 128, 128), gt).cuda()
            # voxel_landmark, gt_heatmap = random_rotate_3d(voxel_landmark, gt_heatmap.float())

            if 'mt' in args.model_name:
                out_seg, out_landmark = model(voxel_fuse)
                loss_heatmap = mse_loss(out_landmark[:bs_half], gt_heatmap)
                loss_seg = F.cross_entropy(out_seg[bs_half:], gt_seg.squeeze(1).long())

                if args.use_sam:
                    # land_pseudo_mask = (land_pseudo_mask > 0.5).long()
                    # print(out_seg[:bs_half, 5].unsqueeze(1).shape)
                    modified_mask_half = torch.where(land_pseudo_mask == 5, torch.tensor(1), land_pseudo_mask)
                    loss_seg += F.binary_cross_entropy(torch.sigmoid(out_seg[:bs_half, 5]), modified_mask_half.float())
                if 'bab' in args.loss_func:
                    loss_bab_1 = bab_loss(out_seg[bs_half:], gt_heatmap, landmark_relations, args.max_dist)
                    loss_bab_2 = bab_loss(one_hot_encode(gt_seg.squeeze(1)), out_landmark[:bs_half], landmark_relations, args.max_dist)
                    loss_bab = loss_bab_1 + loss_bab_2
                    loss = loss_heatmap + loss_seg + loss_bab
                elif 'bab-p' in args.loss_func:
                    loss_bab_1 = bab_loss(out_seg[bs_half:], gt_heatmap, landmark_relations, args.max_dist)
                    loss_bab_2 = bab_loss(one_hot_encode(gt_seg.squeeze(1)), out_landmark[:bs_half], landmark_relations, args.max_dist)
                    loss_bab_pseudo = bab_loss(out_seg, out_landmark, landmark_relations, args.max_dist)
                    loss_bab = loss_bab_1 + loss_bab_2 + loss_bab_pseudo
                    loss = loss_heatmap + loss_seg + loss_bab
                elif 'dyn' in args.loss_func:
                    loss_heatmap = mse_loss(out_landmark[:bs_half], gt_heatmap)
                    loss_seg = F.cross_entropy(out_seg[bs_half:], gt_seg.squeeze(1).long())
                    loss, lambda_weight, avg_cost = dyn_mt_loss(epoch, loss_heatmap, loss_seg, lambda_weight, avg_cost)
                elif 'st' in args.loss_func:
                    # Calculate pseudo segmentation probability, labels, and mask
                    pseudo_seg_prob = F.softmax(out_seg[:bs_half], dim=1)
                    pseudo_seg_max_value, pseudo_seg_label = torch.max(pseudo_seg_prob, dim=1)
                    mask_seg = pseudo_seg_max_value > confidence_threshold

                    # Apply the mask and calculate unsupervised segmentation loss
                    out_seg_masked = out_seg[:bs_half][mask_seg.unsqueeze(1).expand_as(out_seg[:bs_half])].view(-1, out_seg.size(1))
                    pseudo_seg_label_masked = pseudo_seg_label[mask_seg].detach().view(-1)
                    unsupervised_seg_loss = F.cross_entropy(out_seg_masked, pseudo_seg_label_masked) if out_seg_masked.size(
                        0) > 0 else torch.tensor(0.0).cuda()

                    # Calculate pseudo landmark probability, confidence, and mask
                    pseudo_landmark_prob = out_landmark[bs_half:].detach()
                    confidence = torch.max(pseudo_landmark_prob, dim=1)[0]
                    mask_landmark = confidence > confidence_threshold * 255

                    # Apply the mask and calculate unsupervised landmark loss
                    out_landmark_masked = out_landmark[bs_half:][
                        mask_landmark.unsqueeze(1).expand_as(out_landmark[bs_half:])].view(-1, out_landmark.size(1))
                    pseudo_landmark_prob_masked = pseudo_landmark_prob[
                        mask_landmark.unsqueeze(1).expand_as(out_landmark[bs_half:])].view(-1, out_landmark.size(1))
                    unsupervised_landmark_loss = F.mse_loss(out_landmark_masked, pseudo_landmark_prob_masked) if out_landmark_masked.size(
                        0) > 0 else torch.tensor(0.0).cuda()

                    loss = loss_heatmap + loss_seg + unsupervised_landmark_loss + unsupervised_seg_loss
                else:
                    loss = loss_heatmap + loss_seg
                if 'bab' in args.loss_func:
                    logger_loss.write(
                        str(loss_heatmap.item()) + ' ' + str(loss_seg.item()) + ' ' + str(loss_bab.item()) + '\n')
                else:
                    logger_loss.write(str(loss_heatmap.item()) + ' ' + str(loss_seg.item()) + '\n')
            elif 'dodnet' in args.model_name:
                cond = torch.zeros((voxel_fuse.shape[0], 1)).long().cuda()
                cond[:bs_half] = 1
                out = model(voxel_fuse, cond)
                loss_heatmap = mse_loss(out[:bs_half][:, :num_points], gt_heatmap)
                loss_seg = F.cross_entropy(out[bs_half:][:, :num_segments], gt_seg.squeeze(1).long())
                loss = loss_heatmap + loss_seg
            elif 'transdod' in args.model_name:
                cond = torch.zeros((voxel_fuse.shape[0], 1)).long().cuda()
                cond[:bs_half] = 1
                out = model(voxel_fuse, cond)

                N_b, N_q, N_c, N_d, N_h, N_w = out.shape
                preds_convert = torch.zeros(size=(N_b, N_c, N_d, N_h, N_w)).cuda()
                for i_b in range(N_b):
                    preds_convert[i_b] = out[i_b, int(cond[i_b])]

                loss_heatmap = mse_loss(preds_convert[:bs_half][:, :num_points], gt_heatmap)
                loss_seg = F.cross_entropy(preds_convert[bs_half:][:, :num_segments], gt_seg.squeeze(1).long())
                loss = loss_heatmap + loss_seg
            elif 'uniseg' in args.model_name:
                cond = torch.zeros((voxel_fuse.shape[0], 1)).long().cuda()
                cond[:bs_half] = 1  # 0 for seg, 1 for landmark
                out = model(voxel_fuse, cond)
                loss_heatmap = mse_loss(out[:bs_half][:, :num_points], gt_heatmap)
                loss_seg = F.cross_entropy(out[bs_half:][:, :num_segments], gt_seg.squeeze(1).long())
                loss = loss_heatmap + loss_seg
            elif 'SDMT' in args.model_name:
                out_seg, out_landmark = model(voxel_fuse)
                loss_heatmap = mse_loss(out_landmark[:bs_half], gt_heatmap)
                loss_seg = F.cross_entropy(out_seg[bs_half:], gt_seg.squeeze(1).long())
                loss, lambda_weight, avg_cost = dyn_mt_loss(epoch, loss_heatmap, loss_seg, lambda_weight, avg_cost)
            else:
                raise NotImplementedError

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
        print('loss function:', loss.item())
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{num_epochs}, Current Learning Rate: {current_lr}")
        valid_mre, valid_distances = evaluate_model(args, model, valid_loader_landmark, 'l', num_points, num_segments)
        valid_dice, valid_abn_dice = evaluate_model(args, model, valid_loader_seg, 's', num_points, num_segments)
        print('val_mre:' + str(valid_mre) + ' val_dice:' + str(valid_dice))
        if args.record_train_metric:
            train_mre, train_distances = evaluate_model(args, model, train_loader, 'l', num_points, num_segments)
            train_dice, train_abn_dice = evaluate_model(args, model, train_loader_seg, 's', num_points, num_segments)
            logger_file.write('train_mre:' + str(train_mre) + ' train_dice:' + str(train_dice) + ' val_mre:' + str(valid_mre) + ' val_dice:' + str(valid_dice) + '\n')
        else:
            logger_file.write('val_mre:' + str(valid_mre) + ' val_dice:' + str(valid_dice) + '\n')
        current_best = valid_dice / valid_mre
        if local_best < current_best:
            local_best = current_best
            test_mre, test_distances = evaluate_model(args, model, test_loader_landmark, 'l', num_points, num_segments)
            test_dice, test_abn_dice = evaluate_model(args, model, test_loader_seg, 's', num_points, num_segments)
            weights_path = save_path + str(epoch) + '-val-' + str(valid_dice) + '-test-' + str(
                test_dice) + '-val-' + str(valid_mre) + '-test-' + str(test_mre) + '.pth'
            print('test_mre:' + str(test_mre) + ' test_dice:' + str(test_dice))
            torch.save(model.state_dict(), weights_path)

    return 'finish'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int,
                        default=100, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size per gpu')
    parser.add_argument('--patch_size', type=list, default=[128, 128, 128],
                        help='patch size of network input')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--max_dist', type=int, default=10, help='random seed')

    parser.add_argument('--model_name', type=str, default='mtnet',
                        help='the network structure used for training')
    parser.add_argument('--loss_func', type=str, default='',
                        help='the loss function used for training')
    parser.add_argument('--dataset', type=str, default='feta',
                        help='the args.dataset used for training, feta or knee')
    parser.add_argument('--use_sam', type=bool, default=False,
                        help='the args.dataset used for training, feta or knee')
    parser.add_argument('--record_train_metric', type=bool, default=False,
                        help='the args.dataset used for training, feta or knee')
    # setting for TransDoDNet
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--dec_layers', default=3, type=int)
    parser.add_argument('--dim_feedforward', default=768, type=int)
    parser.add_argument('--hidden_dim', default=192, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=2, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--num_feature_levels', default=3, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--normalize_before', default=False)
    parser.add_argument('--deepnorm', default=True)
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument("--add_memory", type=int, default=2,
                        choices=(0, 1, 2))  # feature fusion: 0: cnn; 1:tr; 2:cnn+tr
    parser.add_argument('--res_depth', default=50, type=int)
    parser.add_argument("--dyn_head_dep_wid", type=str, default='3,8')
    args = parser.parse_args()

    train(args)
