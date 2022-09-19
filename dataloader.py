import os

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import resize_and_pad_image


class StainlessDefectsDataset(Dataset):
    def __init__(self, base_dir, split, input_size=None, data_type='classification'):
        self.base_dir = base_dir
        self.split = split
        self.input_size = input_size
        self.data_type = data_type

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Augmentation setting (Not yet implemented all)
        self.affine_seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Fliplr(0.5)),
            iaa.Sometimes(0.5, iaa.Flipud(0.5)),
        ], random_order=True)
        # self.color_seq = iaa.Sequential([
        #     iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 0.5))),
        #     iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
        #     iaa.Sometimes(0.2, iaa.Multiply((0.8, 1.2), per_channel=0.2)),
        #     iaa.Sometimes(0.2, iaa.MultiplyHue((0.8, 1.2))),
        #     iaa.Sometimes(0.2, iaa.MultiplySaturation((0.8, 1.2))),
        #     iaa.Sometimes(0.2, iaa.LogContrast((0.8, 1.2))),
        # ], random_order=True)

        # Load data
        self.samples = []
        temp_label_num = [0, 0]
        data_dir = os.path.join(self.base_dir, self.split)
        for path, dir, files in os.walk(data_dir):
            for filename in files:
                fn, ext = os.path.splitext(filename)
                if ext not in ('.png', '.jpg', '.jpeg') or 'GT' in fn:
                    continue

                img_path = os.path.join(path, filename)
                gt_path = os.path.join(path, '{}_GT{}'.format(fn, ext))
                if not os.path.isfile(gt_path):
                    continue

                if self.data_type == 'classification':
                    gt = cv2.imread(gt_path)
                    gt = 1. if np.max(gt) > 0 else 0.
                    temp_label_num[int(gt)] += 1
                else:
                    print('Data type {} is not implemented yet!'.format(self.data_type))
                    raise TypeError

                self.samples.append((img_path, gt))

        if self.data_type == 'classification':
            print('Loaded {} total: {} ({})'.format(self.split, len(self.samples), temp_label_num))
        else:
            print('Loaded {} total: {}'.format(self.split, len(self.samples)))

    def __getitem__(self, index):
        img_path, gt = self.samples[index]

        img = cv2.imread(img_path)
        if self.input_size is not None:
            img = resize_and_pad_image(img, target_size=(self.input_size, self.input_size), keep_ratio=True, padding=True)

        if self.split == 'train':
            img = self.affine_seq.augment_image(img)

        img = img.astype(np.float32) / 255.
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        gt = torch.from_numpy(np.array([gt]))

        return img, gt

    def __len__(self):
        return len(self.samples)
