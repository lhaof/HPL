import glob
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


class KneeSegDataset(Dataset):
    def __init__(self, mode='train', transform=None, patch_size=[128, 128, 128], zoom=True):
        self.mode = mode
        self.transform = transform
        self.patch_size = patch_size
        self.zoom = zoom
        self.oaiseg_dir = 'dataset_knee_seg/OAISEG'

        self.data_path_list, self.label_path_list = self.create_data_list()
        self.data_list, self.label_list = [], []
        self.spacing_list, self.shape_list = [], []  # Lists to store spacing and shape information
        self.name_list = []
        with Pool() as pool:
            tasks = [(img_path, label_path, self.patch_size, self.mode)
                     for img_path, label_path in zip(self.data_path_list, self.label_path_list)]
            results = list(tqdm(pool.imap(self.process_data, tasks), total=len(self.data_path_list)))
            pool.close()
            pool.join()

        self.data_list, self.label_list, self.spacing_list, self.shape_list, self.name_list = zip(*results)

    def normalization(self, image):
        masked_intensities = image[image > 0]
        mean_intensity = np.mean(masked_intensities)
        std_intensity = np.std(masked_intensities)
        window_level = mean_intensity
        window_width = 4 * std_intensity
        min_intensity = window_level - window_width / 2
        max_intensity = window_level + window_width / 2
        windowed_array = np.clip(image, min_intensity, max_intensity)
        return ((windowed_array - min_intensity) / window_width).astype(np.float32)

    def process_data(self, args):
        img_path, label_path, patch_size, mode = args
        idx = img_path.strip().split('/')[-1].split('.')[0]
        vol = sitk.ReadImage(img_path, sitk.sitkFloat32)
        image = sitk.GetArrayFromImage(vol)
        spacing = vol.GetSpacing()
        shape = image.shape
        mask = sitk.GetArrayFromImage(sitk.ReadImage(label_path, sitk.sitkUInt8))
        image = self.normalization(image)
        map = mask > 0
        image = image * map
        image1 = np.expand_dims(image, 0)
        nonzero_mask = create_nonzero_mask(image1)
        bbox = get_bbox_from_mask(nonzero_mask, 0)
        image = crop_to_bbox(image, bbox)
        mask = crop_to_bbox(mask, bbox)
        image = resize(image, output_shape=patch_size, order=3, preserve_range=True)
        mask = resize(mask, output_shape=patch_size, order=0, preserve_range=True)
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        return image, mask, spacing, shape, idx

    def create_data_list(self):
        def get_paths(subdir):
            images = sorted(glob.glob(os.path.join(self.oaiseg_dir, 'image', subdir, '*.nii.gz')))
            labels = sorted(glob.glob(os.path.join(self.oaiseg_dir, 'label', subdir, '*.nii.gz')))
            return [(img, lbl) for img, lbl in zip(images, labels) if os.path.basename(img) == os.path.basename(lbl)]

        if self.mode == 'train':
            return zip(*get_paths('train'))
        elif self.mode == 'validation':
            return zip(*get_paths('val'))
        elif self.mode == 'test':
            return zip(*get_paths('test'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = {
            'image': self.data_list[idx],
            'mask': self.label_list[idx],
            'spacing': self.spacing_list[idx],
            'shape': self.shape_list[idx],
            'idx': self.name_list[idx]
        }
        return sample
