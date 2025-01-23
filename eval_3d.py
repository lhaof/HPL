import argparse
import os
import time

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from monai.metrics import compute_surface_dice
from monai.networks.utils import one_hot
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
from utils.plot_3d_points import plot_3D_points_knee, plot_3D_points_feta


def save_metrics(metric_values, case_names, metric_name, current_root, num_points=None, num_segments=None):
    metric_values = np.array(metric_values)

    # Calculate overall average and std
    overall_avg = np.mean(metric_values)
    overall_std = np.std(metric_values)

    # Prepare data for CSV
    rows = []
    for i, case_name in enumerate(case_names):
        row = [case_name] + [np.round(np.mean(metric_values[i]), 2)] + np.round(metric_values[i], 2).tolist()
        rows.append(row)
    avg_row = ["Avg"] + [np.round(overall_avg, 2)] + np.round(np.mean(metric_values, axis=0), 2).tolist()
    std_row = ["Std"] + [np.round(overall_std, 2)] + np.round(np.std(metric_values, axis=0), 2).tolist()
    rows.insert(0, std_row)
    rows.insert(0, avg_row)

    # Write CSV
    csv_path = os.path.join(current_root, f'{metric_name}_cases.csv')
    with open(csv_path, 'w') as f:
        if metric_name in ['landmark_mre', 'landmark_sdr']:
            header = ['Case'] + [f'Average_{metric_name.upper()}'] + [f'Landmark_{i}' for i in range(num_points)]
        else:
            header = ['Case'] + [f'Average_{metric_name.upper()}'] + [f'Structure_{i}' for i in range(1, num_segments)]
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(map(str, row)) + '\n')


def get_points_from_heatmap_torch(heatmap):
    max_indices = torch.argmax(heatmap.view(heatmap.size(0), heatmap.size(1), -1), dim=2)
    z_indices = max_indices % heatmap.size(4)
    y_indices = (max_indices // heatmap.size(4)) % heatmap.size(3)
    x_indices = max_indices // (heatmap.size(3) * heatmap.size(4))
    return torch.stack([x_indices, y_indices, z_indices], dim=2)


def resample_to_original(tensor, original_shape, mode='trilinear'):
    if tensor.dim() == 4:  # Assuming tensor is in (B, C, H, W) format for 2D data
        resampled_tensor = F.interpolate(tensor, size=original_shape, mode=mode,
                                         align_corners=False if mode != 'nearest' else None)
    elif tensor.dim() == 5:  # Assuming tensor is in (B, C, D, H, W) format for 3D data
        resampled_tensor = F.interpolate(tensor, size=original_shape, mode=mode,
                                         align_corners=False if mode != 'nearest' else None)
    else:
        raise ValueError("Unsupported tensor dimensions. Expected 4D or 5D tensor.")

    return resampled_tensor


def evaluate_model(args):
    start_time = time.time()
    is_knee = 'knee' in args.model_path
    current_root = f'./results_{"knee" if is_knee else "feta"}/' + args.model_path.split('/')[1] + '/'
    image_path, mask_path, landmark_path = [current_root + sub_dir for sub_dir in ['img/', 'mask/', 'landmark/']]
    num_segments, num_points = (6, 13) if is_knee else (8, 6)
    sdr_threshold = 4 if is_knee else 1.5

    os.makedirs(current_root, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(landmark_path, exist_ok=True)

    if args.model_name == 'swinunetr-mt':
        model = SwinUNETR_multitask(out_channels_seg=num_segments, out_channels_landmark=num_points).cuda()
    elif args.model_name == 'mtnet-gn':
        model = MTNetGN(seg_out_ch=num_segments, landmark_out_ch=num_points).cuda()
    elif args.model_name == 'mtnet':
        model = ResUnetMT(points=num_points, segments=num_segments).cuda()
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
    elif args.model_name == 'sdmt':
        model = SDMT(num_segment=num_segments, num_landmark=num_points).cuda()
    else:
        raise NotImplementedError(f"Model {args.model_name} is not implemented.")

    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore unexpected keys
    model.eval()

    if args.eval_flag == 'l':
        test_set = FetalDataset(mode='test') if not is_knee else KneeDataset(mode='test')
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True)
        case_names = []
        mre_per_case = []
        sdr_per_case = []

        for data in tqdm(test_loader):
            voxel, gt, case_info = data
            case_name = case_info['name'][0]
            case_names.append(case_name)
            factor, spacing = case_info['factor'].unsqueeze(1).float(), case_info['spacing'].unsqueeze(1).float()
            origin = case_info['origin'].detach().cpu().numpy()[0]
            voxel = voxel.cuda().unsqueeze(1).float()

            if "mt" in args.model_name or 'SDMT' in args.model_name:
                _, heatmap = model(voxel)
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

            pred_3d_coor = get_points_from_heatmap_torch(heatmap[:, :num_points]).float().cpu().numpy()
            if is_knee:
                plot_3D_points_knee(gt[0].cpu().numpy(), pred_3d_coor[0], save_name=landmark_path + case_name + '.jpg')
            else:
                plot_3D_points_feta(gt[0].cpu().numpy(), pred_3d_coor[0], save_name=landmark_path + case_name + '.jpg')

            pred_3d_coor /= (factor / spacing)
            gt /= (factor / spacing)
            mre_case = torch.sqrt(torch.sum((pred_3d_coor - gt) ** 2, dim=-1)).squeeze(0).numpy()
            mre_per_case.append(mre_case)

            sdr_case = (mre_case <= sdr_threshold).astype(float) * 100
            sdr_per_case.append(sdr_case)

        save_metrics(mre_per_case, case_names, 'landmark_mre', current_root, num_points=num_points)
        save_metrics(sdr_per_case, case_names, 'landmark_sdr', current_root, num_points=num_points)

    elif args.eval_flag == 's':
        test_seg = FetalSegData(data_dir='./dataset_fetal_seg/feta', mode="test",
                                list_name='test.list') if not is_knee else KneeSegDataset(mode="test")
        test_loader_seg = DataLoader(test_seg, batch_size=1, shuffle=False, num_workers=1)

        case_names = []
        dice_scores = []
        nsd_scores = []

        for sampled_batch in tqdm(test_loader_seg):
            volume_batch, label, spacing = sampled_batch['image'], sampled_batch['mask'], sampled_batch['spacing']
            spacing = torch.tensor(spacing).numpy()
            label = label.squeeze(1).cuda().cpu().squeeze(0).numpy()  # Ensure label is in numpy format
            case_name = sampled_batch['idx'][0].split('.')[0]
            case_names.append(case_name)

            with torch.no_grad():
                if "mt" in args.model_name or 'SDMT' in args.model_name:
                    out, _ = model(volume_batch.cuda())
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

                out = torch.argmax(torch.softmax(out, dim=1), dim=1).cpu().squeeze(0).numpy()

            original_shape = label.shape
            volume_batch_resampled = resample_to_original(volume_batch, original_shape)
            label_resampled = resample_to_original(torch.tensor(label).unsqueeze(0).unsqueeze(0), original_shape, mode='nearest')
            out_resampled = resample_to_original(torch.tensor(out).unsqueeze(0).unsqueeze(0).float(), original_shape, mode='nearest')

            # Ensure tensors are in the correct format
            label_resampled = label_resampled.squeeze(0).long()
            out_resampled = out_resampled.squeeze(0).long()

            dice_score = torchmetrics.functional.dice(out_resampled.flatten(), label_resampled.long().flatten(), average='none',
                                                      num_classes=num_segments)[1:].numpy() * 100
            dice_scores.append(dice_score)

            # One-hot encode the labels and predictions
            label_resampled = one_hot(label_resampled.unsqueeze(0), num_segments).squeeze(0)
            out_resampled = one_hot(out_resampled.unsqueeze(0), num_segments).squeeze(0)

            surface_dice = compute_surface_dice(
                out_resampled.unsqueeze(0),
                label_resampled.unsqueeze(0),
                class_thresholds=[1] * (num_segments - 1),
                include_background=False,
                distance_metric="euclidean",
                spacing=[spacing]
            ).squeeze(0).numpy() * 100
            nsd_scores.append(surface_dice)

            # sitk.WriteImage(sitk.GetImageFromArray(volume_batch_resampled.squeeze(0).squeeze(0).numpy() * 255),
            #                 image_path + case_name + '_srr.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(label_resampled.argmax(dim=0).numpy().astype(np.int16)),
                            mask_path + case_name + '_gt.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(out_resampled.argmax(dim=0).numpy().astype(np.int16)),
                            mask_path + case_name + '_pred.nii.gz')

        save_metrics(dice_scores, case_names, 'segmentation_dice', current_root, num_segments=num_segments)
        save_metrics(nsd_scores, case_names, 'segmentation_nsd', current_root, num_segments=num_segments)

    print('Evaluation completed in:', time.time() - start_time, 'seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--model_path', type=str, help='Path to the model file')
    parser.add_argument('--eval_flag', type=str, default='l', help='l for landmark and s for segmentation')

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
    evaluate_model(args)