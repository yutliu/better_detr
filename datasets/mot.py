# Copyright (c) Yutliu
"""
MOT dataset which returns image_id for evaluation.
"""
from pathlib import Path
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw

import datasets.transforms as T


class MOTDetection:
    def __init__(self, args, seqs_folder, transforms):
        self.args = args
        self._transforms = transforms
        path = r'/share/home/kaihuatrack/Code/detr/datasets/data_path/mot17.train'
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [osp.join(seqs_folder, x.strip()) for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [
            x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
            for x in self.img_files]

        self.item_num = len(self.img_files)

    def pre_data(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        img = Image.open(img_path)
        targets = {}
        w, h = img._size

        if osp.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized cewh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4]/2)
            labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5]/2)
            labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4]/2)
            labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5]/2)
            # draw = ImageDraw.Draw(img)
            # for label in labels:
            #     draw.rectangle(label[2:6].tolist(), outline=tuple(np.random.randint(0, 255, size=[3])))
            # img.show()

        targets['boxes'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        for label in labels:
            targets['boxes'].append(label[2:6].tolist())
            targets['area'].append(label[4] * label[5])
            targets['iscrowd'].append(0)
            targets['labels'].append(0)

        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['boxes'][:, 0::2].clamp_(min=0, max=w)
        targets['boxes'][:, 1::2].clamp_(min=0, max=h)
        return img, targets

    def __getitem__(self, idx):
        img, target = self.pre_data(idx)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        img_path = self.img_files[idx]
        return img, target

    def __len__(self):
        return self.item_num

class MOTDetection_val:
    def __init__(self, args, seqs_folder, transforms):
        self.args = args
        self._transforms = transforms
        path = r'/share/home/kaihuatrack/Code/detr/datasets/data_path/mot17.train'
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [osp.join(seqs_folder, x.strip()) for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [
            x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
            for x in self.img_files]

        self.item_num = len(self.img_files)

    def pre_data(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        img = Image.open(img_path)
        targets = {}
        w, h = img._size

        if osp.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized cewh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4]/2)
            labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5]/2)
            labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4]/2)
            labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5]/2)
            # draw = ImageDraw.Draw(img)
            # for label in labels:
            #     draw.rectangle(label[2:6].tolist(), outline=tuple(np.random.randint(0, 255, size=[3])))
            # img.show()

        targets['boxes'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        for label in labels:
            targets['boxes'].append(label[2:6].tolist())
            targets['area'].append(label[4] * label[5])
            targets['iscrowd'].append(0)
            targets['labels'].append(0)

        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['boxes'][:, 0::2].clamp_(min=0, max=w)
        targets['boxes'][:, 1::2].clamp_(min=0, max=h)
        return img, targets

    def __getitem__(self, idx):
        img, target = self.pre_data(idx)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        img_path = self.img_files[idx]
        return img, target

    def __len__(self):
        return self.item_num


def make_mot_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    if image_set == 'train':
        dataset = MOTDetection(args, root, transforms=make_mot_transforms(image_set))
    if image_set == 'test':
        dataset = MOTDetection_val(args, root, transforms=make_mot_transforms(image_set))
    return dataset

