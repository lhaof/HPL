import json
import os
from multiprocessing import Pool

import torch
from torch.utils.data import Dataset

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import skimage.transform as skTrans
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Define the points, compute the convex hull, and create the 3D grid of points
# points = np.array([
#     [30, 30, 30],
#     [100, 30, 30],
#     [30, 100, 30],
#     [100, 100, 30],
#     [50, 50, 100],
#     [90, 90, 100]
# ])


def get_pseudo_mask(points):
    # print(points)
    hull = ConvexHull(points)
    cube_shape = (128, 128, 128)
    xx, yy, zz = np.meshgrid(np.arange(cube_shape[0]), np.arange(cube_shape[1]), np.arange(cube_shape[2]),
                             indexing='ij')
    grid_points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

    # Check if each grid point is inside the convex hull and create the 3D mask
    delaunay = Delaunay(hull.points[hull.vertices])
    mask = delaunay.find_simplex(grid_points) >= 0
    mask_3d = mask.reshape(cube_shape)
    # print(mask_3d.shape)
    return mask_3d


# # Plot the 3D mask using matplotlib
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.voxels(mask_3d, edgecolor='k')
# plt.show()

def find_symmetric_point_with_midpoint(A, P1, P2):
    B = ((P1[0] + P2[0]) / 2, (P1[1] + P2[1]) / 2, (P1[2] + P2[2]) / 2)
    return 2 * B[0] - A[0], 2 * B[1] - A[1], 2 * B[2] - A[2]


class FetalDataset(Dataset):
    def __init__(self, mode, root_path='./dataset_fetal_land/', img_size=128, pseudo_label=False, aug=None,
                 cere_only=True):
        self.voxel_path = f"{root_path}{mode}/"
        self.anno_path = f"{root_path}{mode}_label/"
        self.pseudo_label = pseudo_label
        case_folders = sorted(os.listdir(self.anno_path), key=int)
        self.data_list = []
        self.img_size = img_size
        self.aug = aug
        self.cere_only = cere_only
        with Pool() as pool:
            results = list(tqdm(pool.imap(self.process_case_folder, case_folders), total=len(case_folders)))
        # results = []
        # for case_folder in case_folders:
        #     results.append(self.process_case_folder(case_folder))
        self.data_list = [result for result in results if result is not None]

    def padding_fetal_mr(self, fetal_mr):
        # Assuming fetal_mr is already read and is a SimpleITK Image
        size_ori = fetal_mr.GetSize()
        longest_side = max(size_ori)

        # Calculate the padding needed for each dimension
        lower_bound_padding = [(longest_side - s) // 2 for s in size_ori]
        upper_bound_padding = [longest_side - (s + lb) for s, lb in zip(size_ori, lower_bound_padding)]

        # If padding is required (i.e., the image is not already cubic), pad the image
        if any(p > 0 for p in lower_bound_padding + upper_bound_padding):
            # Apply the padding with the constant value you want, e.g., 0
            return sitk.ConstantPad(fetal_mr, lower_bound_padding, upper_bound_padding, 0)
        else:
            return fetal_mr

    def process_case_folder(self, case_folder):
        json_paths_ori = os.listdir(f"{self.anno_path}/{case_folder}")
        json_paths = sorted(json_paths_ori, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        if len(json_paths) != 6:
            print(case_folder)
            return None

        fetal_mr, origin, direction, factor, spacing = self.process_image(self.voxel_path, case_folder)
        mask = self.process_mask(self.voxel_path, case_folder)
        norm_landmarks, case, json_path = self.get_anno(json_paths, self.anno_path, case_folder, origin, factor,
                                                        spacing)
        if self.pseudo_label:
            small_brain_points = np.array([norm_landmarks[0], norm_landmarks[1], norm_landmarks[2],
                                           norm_landmarks[3], norm_landmarks[4], norm_landmarks[5]])
            small_brain = get_pseudo_mask(small_brain_points)
            case_info = {'name': case_folder, 'origin': origin, 'factor': factor, 'small_brain': small_brain,
                         'json_path': json_path, 'mask': mask}
        else:
            case_info = {'name': case_folder, 'origin': origin, 'factor': factor, 'json_path': json_path,
                         'spacing': spacing, 'mask': mask}
        return fetal_mr, norm_landmarks, case_info

    def window_image(self, fetal_mr):
        masked_intensities = fetal_mr[fetal_mr > 0]
        mean_intensity = np.mean(masked_intensities)
        std_intensity = np.std(masked_intensities)
        window_level = mean_intensity
        window_width = 4 * std_intensity

        min_intensity = window_level - window_width / 2
        max_intensity = window_level + window_width / 2

        # Clamp intensities
        windowed_array = np.clip(fetal_mr, min_intensity, max_intensity)

        # Normalize the windowed intensities to the range [0, 1]
        normalized = (windowed_array - min_intensity) / window_width
        normalized = normalized.astype(np.float32)
        return normalized

    def process_image(self, voxel_path, case_folder):
        fetal_mr = sitk.ReadImage(f"{voxel_path}{case_folder}.nii.gz")
        # fetal_mr = self.padding_fetal_mr(fetal_mr)
        origin = fetal_mr.GetOrigin()
        direction = fetal_mr.GetDirection()
        spacing = fetal_mr.GetSpacing()

        size = fetal_mr.GetSize()
        fetal_mr = sitk.GetArrayFromImage(fetal_mr)

        fetal_mr = self.window_image(fetal_mr)

        factor = np.divide([self.img_size, self.img_size, self.img_size], size)
        factor[0], factor[2] = factor[2], factor[0]
        fetal_mr = skTrans.resize(fetal_mr, (self.img_size, self.img_size, self.img_size), order=3, preserve_range=True)
        return fetal_mr, np.asarray(origin), direction, factor, np.array(spacing)

    def process_mask(self, voxel_path, case_folder):
        voxel_path = voxel_path.replace('train/', 'train_seg/')
        fetal_mr = sitk.ReadImage(f"{voxel_path}{case_folder}.nii.gz")
        size = fetal_mr.GetSize()
        fetal_mr = sitk.GetArrayFromImage(fetal_mr)

        factor = np.divide([self.img_size, self.img_size, self.img_size], size)
        factor[0], factor[2] = factor[2], factor[0]
        fetal_mr = skTrans.resize(fetal_mr, (self.img_size, self.img_size, self.img_size), order=0, preserve_range=True)
        return fetal_mr

    def get_anno(self, json_paths, anno_path, case_folder, origin, factor, spacing):
        norm_landmarks = []
        annotation_list = []
        metric_list = ['CBD', 'BBD', 'TCD', 'FOD', 'HDV', 'ADV']
        cere_only_idxs = [0, 1, 3] if self.cere_only else []

        metric_idx = 0
        for json_path in json_paths:
            if metric_idx in cere_only_idxs:
                metric_idx += 1
                continue
            path = f"{anno_path}/{case_folder}/{json_path}"
            json_file = json.load(open(path, 'r'))
            length_value = json_file['markups'][0]['measurements'][0]['value']
            start_point, end_point = (np.array(p['position']) for p in json_file['markups'][0]['controlPoints'])
            case = {
                'case_id': case_folder,
                'metric_id': metric_list[metric_idx],
                'length_value': length_value,
                'start_point': start_point,
                'end_point': end_point
            }
            annotation_list.append(case)
            normalized_start_point = (start_point - origin) / (spacing / factor)
            normalized_end_point = (end_point - origin) / (spacing / factor)
            norm_landmarks.append(normalized_start_point)
            norm_landmarks.append(normalized_end_point)
            metric_idx += 1
        return np.asarray(norm_landmarks), annotation_list, path

    def __getitem__(self, idx):
        voxel, gt, case_info = self.data_list[idx]
        if self.aug:
            voxel, gt = self.aug(voxel, gt)
        return voxel, gt, case_info

    def __len__(self):
        return len(self.data_list)
