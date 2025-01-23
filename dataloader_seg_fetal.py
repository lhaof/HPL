import logging
import os
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm


def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


class FetalSegData(Dataset):
    def __init__(self, data_dir=None, mode='train', preprocess='training', list_name='train.list', transform=None,
                 patch_size=[128, 128, 128], zoom=True, atlas=None, data_dir_s=None):
        self._data_dir = data_dir
        self.sample_list = []
        self.mode = mode
        self.preprocess = preprocess
        self.transform = transform
        self.patch_size = patch_size
        self.zoom = zoom
        self.atlas = atlas
        self.data_dir_s = data_dir_s

        self.illness = [
            'sub-070', 'sub-052', 'sub-066', 'sub-005', 'sub-017', 'sub-016', 'sub-042', 'sub-023',
            'sub-025', 'sub-008', 'sub-020', 'sub-021', 'sub-014', 'sub-080', 'sub-073', 'sub-003',
            'sub-013', 'sub-064', 'sub-048', 'sub-002', 'sub-009', 'sub-075', 'sub-024', 'sub-050'
        ]
        self.illness_t = [
            'sub-022', 'sub-065', 'sub-063', 'sub-071', 'sub-007', 'sub-043', 'sub-074', 'sub-015',
            'sub-078', 'sub-006', 'sub-004', 'sub-012', 'sub-056', 'sub-077', 'sub-055', 'sub-010',
            'sub-069', 'sub-011', 'sub-001', 'sub-047', 'sub-018', 'sub-067', 'sub-019', 'sub-049',
            'sub-054'
        ]
        self.week = [
            27.9, 28.2, 27.4, 25.5, 22.6, 24.9, 22.8, 25.2, 29, 27.3, 27.6, 25.9, 27.5, 26.7, 23.7,
            23.3, 22.8, 28.5, 29.2, 25.8, 26.1, 20, 23.7, 30.4, 24.2, 27.8, 26.5, 31.1, 32.5, 33.4,
            31.4, 32.3, 30, 28.7, 32.8, 22.7, 23.4, 26.9, 24.3, 27.3, 34.8, 23.6, 22.9, 27.9, 24.7,
            23.9, 28.1, 27.9, 31.1, 33.1, 29.6, 21.2, 30.3, 33.1, 27.1, 26.6, 28.2, 29.2, 34.8, 31.7,
            33, 24.4, 21.7, 27.8, 20.9, 21.8, 29, 31.5, 27.4, 20.1, 22.4, 25.9, 27.2, 23.3, 29, 23.2,
            26.9, 24, 29.1, 26.9
        ]

        self.dict = {f'sub-{str(i + 1).zfill(3)}': age for i, age in enumerate(self.week)}

        list_path = os.path.join(self._data_dir, list_name)
        with open(list_path, "r") as f:
            self.sample_list = [line.strip() for line in f.readlines()]
            if "atlas" in self._data_dir:
                self.sample_list *= 3

        logging.info(f'Creating total {self.mode} dataset with {len(self.sample_list)} examples')

        self.image_list = []
        self.mask_list = []
        self.idx_list = []
        self.age_list = []
        self.spacing_list = []
        self.pathological_list = []
        self.original_shape_list = []

        self.read_data()

    def read_data(self):
        with Pool() as pool:
            results = tqdm(pool.map(self.create_data_list, self.sample_list))
        self.image_list, self.mask_list, self.idx_list, self.age_list, self.spacing_list, self.pathological_list, self.original_shape_list = zip(
            *results)

    def __len__(self):
        return len(self.sample_list)

    def padding_fetal_mr(self, fetal_mr):
        size_ori = fetal_mr.GetSize()
        longest_side = max(size_ori)
        lower_bound_padding = [(longest_side - s) // 2 for s in size_ori]
        upper_bound_padding = [longest_side - (s + lb) for s, lb in zip(size_ori, lower_bound_padding)]
        if any(p > 0 for p in lower_bound_padding + upper_bound_padding):
            return sitk.ConstantPad(fetal_mr, lower_bound_padding, upper_bound_padding, 0)
        return fetal_mr

    def create_data_list(self, case):
        if "/feta" in self._data_dir:
            self.age = round(self.dict[case[:7]])
            self.pathological = 1 if case[:7] in self.illness or case[:7] in self.illness_t else 0
            img_np_path = os.path.join(self._data_dir, 'image', case)
            mask_np_path = os.path.join(self._data_dir, 'label', case[:-7] + 'dseg.nii.gz')
        else:
            img_np_path = os.path.join(self._data_dir, 'image', case)
            mask_np_path = os.path.join(self._data_dir, 'label', case[:-7] + 'parcellation.nii.gz')
            self.age = int(case[:-8])
            self.pathological = 1 if 14 < self.age < 30 else 0
            if self.age < 15:
                self.age += 21
            elif self.age < 30:
                self.age += 6 if self.age < 20 else 5
            else:
                self.age -= 9

        vol = sitk.ReadImage(img_np_path, sitk.sitkFloat32)
        image = sitk.GetArrayFromImage(vol)
        spacing = vol.GetSpacing()
        original_shape = image.shape
        image = self.normalization(image)

        mask = sitk.ReadImage(mask_np_path, sitk.sitkUInt8)
        mask = sitk.GetArrayFromImage(mask)
        map = mask > 0
        image = image * map
        image1 = np.expand_dims(image, 0)
        nonzero_mask = create_nonzero_mask(image1)
        bbox = get_bbox_from_mask(nonzero_mask, 0)
        image = crop_to_bbox(image, bbox)
        mask = crop_to_bbox(mask, bbox)
        image = resize(image, output_shape=self.patch_size, order=3, preserve_range=True)
        mask = resize(mask, output_shape=self.patch_size, order=0, preserve_range=True)
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)

        sample = {'image': image.copy(), 'mask': mask.copy(), 'idx': case, 'age': self.age,
                  'pathological': self.pathological}
        if self.transform:
            sample = self.transform(sample)
            sample = {'image': sample['image'].copy(), 'mask': sample['mask'].copy(), 'idx': case, 'age': self.age,
                      'pathological': self.pathological}

        return image, mask, case, self.age, spacing, self.pathological, original_shape

    def __getitem__(self, idx):
        return {
            'image': self.image_list[idx],
            'mask': self.mask_list[idx],
            'idx': self.idx_list[idx],
            'age': self.age_list[idx],
            'pathological': self.pathological_list[idx],
            'spacing': self.spacing_list[idx],
            'shape': self.original_shape_list[idx]
        }

    def normalization(self, image):
        masked_intensities = image[image > 0]
        mean_intensity = np.mean(masked_intensities)
        std_intensity = np.std(masked_intensities)
        window_level = mean_intensity
        window_width = 4 * std_intensity
        min_intensity = window_level - window_width / 2
        max_intensity = window_level + window_width / 2
        windowed_array = np.clip(image, min_intensity, max_intensity)
        normalized = (windowed_array - min_intensity) / window_width
        return normalized.astype(np.float32)
