import json
import os
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
import skimage.transform as skTrans
from torch.utils.data import Dataset
from tqdm import tqdm


class KneeDataset(Dataset):
    def __init__(self, mode, root_path='./dataset_knee_landmark/', img_size=128, aug=None):
        self.voxel_path = f"{root_path}{mode}/"
        self.anno_path = f"{root_path}{mode}_label/"
        case_folders = sorted(os.listdir(self.anno_path), key=int)
        self.data_list = []
        self.img_size = img_size
        self.aug = aug
        # results = []
        # for case_folder in case_folders:
        #     results.append(self.process_case_folder(case_folder))
        with Pool() as pool:
            results = list(tqdm(pool.imap(self.process_case_folder, case_folders), total=len(case_folders)))
        if mode == 'train':
            self.data_list = [result for result in results if result is not None for i in range(3)]
        else:
            self.data_list = [result for result in results]

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
        # case_info = {}
        json_paths_ori = os.listdir(f"{self.anno_path}/{case_folder}")
        json_paths_sort = sorted(json_paths_ori, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        if len(json_paths_sort) != 13:
            print(case_folder)
            return None

        fetal_mr, origin, direction, factor, spacing = self.process_image(self.voxel_path, case_folder)
        normalized_key_points = self.process_annotations(json_paths_sort, self.anno_path, case_folder, origin,
                                                               factor, spacing)

        case_info = {'name': case_folder, 'origin': origin, 'factor': factor, 'spacing':spacing}
        return fetal_mr, normalized_key_points, case_info

    def window_image(self, image):
        masked_intensities = image[image > 0]
        mean_intensity = np.mean(masked_intensities)
        std_intensity = np.std(masked_intensities)
        window_level = mean_intensity
        window_width = 4 * std_intensity
        min_intensity = window_level - window_width / 2
        max_intensity = window_level + window_width / 2

        # Clamp intensities
        windowed_array = np.clip(image, min_intensity, max_intensity)

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
        # factor[1], factor[2] = factor[2], factor[1]

        fetal_mr = skTrans.resize(fetal_mr, (self.img_size, self.img_size, self.img_size), order=3, preserve_range=True)

        return fetal_mr, np.asarray(origin), direction, factor, np.array(spacing)

    def process_annotations(self, json_paths_sort, anno_path, case_folder, origin, factor, spacing):
        normalized_key_points = []
        for json_path in json_paths_sort:
            path = f"{anno_path}/{case_folder}/{json_path}"
            json_file = json.load(open(path, 'r'))
            start_point = np.array(json_file['markups'][0]['controlPoints'][0]['position'])
            normalized_start_point = (start_point - origin) / (spacing / factor)
            normalized_key_points.append(normalized_start_point)
        return np.asarray(normalized_key_points)

    def __getitem__(self, idx):
        voxel, gt, case_info = self.data_list[idx]
        if self.aug:
            voxel, gt = self.aug(voxel, gt)
        return voxel, gt, case_info

    def __len__(self):
        return len(self.data_list)
