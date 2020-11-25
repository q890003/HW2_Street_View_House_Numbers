#!/usr/bin/env python
# coding: utf-8

import os
import glob
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import pickle
from torch.utils.data import DataLoader
import cv2


class SVHNDataset(object):
    def __init__(self, root, trans=None, train=True):
        self.root = root
        self.transforms = trans
        self.train = train

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = glob.glob(os.path.join(root, "*.png"))
        #         if self.train == True:
        self.annotation_path = root.replace("train/", "dummy.pkl")
        with open(self.annotation_path, "rb") as f:
            self.annotations = pickle.load(f)
            self.annotations = self.annotations.set_index("img_name")

    def __getitem__(self, idx):
        # load images ad labels
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        #         img = cv2.imread(img_path)
        img_name = os.path.basename(img_path)

        # get lebels of an image
        origin_targets = self.annotations.loc[img_name]
        labels = torch.LongTensor(origin_targets["label"])

        # get bounding box coordinates for each object in the image[idx].
        boxes = []
        num_objs = len(labels)
        for i in range(num_objs):
            left = origin_targets["left"][i]
            top = origin_targets["top"][i]
            right = origin_targets["right"][i]
            bottom = origin_targets["bottom"][i]
            boxes.append([left, top, right, bottom])  # [xmin, ymin, xmax, ymax]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = labels

        new_targets = {}
        new_targets["boxes"] = boxes
        new_targets["labels"] = labels
        if self.train == False:
            new_img = FT.to_tensor(img)
            return new_img, new_targets

        else:
            # transform
            if self.transforms is not None:
                new_img, new_targets = self.transforms(img, new_targets, "TRAIN")

            return new_img, new_targets

    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = list()
        targets = list()

        for b in batch:
            images.append(b[0])
            targets.append(b[1])

        #         images = torch.stack(images, dim=0)

        return images, targets  # tensor (N, 3, 300, 300), 3 lists of N tensors each


if __name__ == "__main__":
    os.chdir(
        "/home/mbl/Yiyuan/Selected_Topics_in_Visual_Recognition_using_Deep_Learning/CV_hw2"
    )
    get_ipython().system("pwd")
    root = "/home/mbl/Yiyuan/Selected_Topics_in_Visual_Recognition_using_Deep_Learning/CV_hw2/data/train/"
    dataset = SVHNDataset(
        root=root,
        trans=utils.transform,
    )

    a = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )
    for i, label in a:
        print((i, label))
