#!/usr/bin/env python
# coding: utf-8

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import os
import time
import copy
import numpy as np

import utils
import dataset
import config
import model


def train_model(model, dataloaders, optimizer, scheduler, num_epochs=25):

    model.train()  # Set model to training mode

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 9999
    for epoch in range(num_epochs):
        epoch_bgin = time.time()
        print("Epoch %d/%d" % (epoch, num_epochs - 1))
        print("-" * 10)

        running_loss = 0.0
        current_num_data = 0
        all_data = len(dataloaders["train"].dataset)
        # Iterate over data.
        for i, (imgs, _targets) in enumerate(dataloaders["train"]):
            if current_num_data % (32 * 40) == 0:
                batch_bgin = time.time()

            imgs = [img.to(device) for img in imgs]
            targets = [
                {
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device),
                }
                for target in _targets
            ]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            pred = model(imgs, targets)
            loss_classifier = pred["loss_classifier"]
            loss_box_reg = pred["loss_box_reg"]
            loss_objectness = pred["loss_objectness"]
            loss_rpn_box_reg = pred["loss_rpn_box_reg"]
            loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg
            loss.backward(retain_graph=True)
            optimizer.step()

            # statistics
            current_num_data += len(imgs)
            running_loss += loss.item() * len(imgs)
            time_batch = time.time() - batch_bgin
            minute_batch, second_batch = time_batch // 60, time_batch % 60
            if current_num_data % (32 * 40) == 0:
                print(
                    "epoch:{epoch}/{total_epoch} [{learned_data}/{data}], "
                    "time: {minute:.0f}m {sec:.0f}s, {phase}: Loss: {loss:.4f}".format(
                        epoch=epoch,
                        total_epoch=num_epochs,
                        learned_data=current_num_data,
                        data=all_data,
                        minute=minute_batch,
                        sec=second_batch,
                        phase="train",
                        loss=loss.item(),
                    )
                )

        scheduler.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            backbone_name = ["Resnet50", "Resnet101", "ResNext50_32x8d"]
            torch.save(
                best_model_wts,
                config.ckpt_dir
                + config.model_name
                + "op_SGD_lr_cylr_{backbone}_dataaug_shear_rotate_crop_noFlip"
                "__zoomin_loss{epochloss}".format(
                    backbone=backbone_name[1], epochloss=epoch_loss
                ),
            )

        epoch_loss = running_loss / all_data
        print("epoch_loss: {}".format(epoch_loss))

        time_epoch = time.time() - epoch_bgin
        print(
            "One epoch time: {:.0f}m {:.0f}s".format(time_epoch // 60, time_epoch % 60)
        )
    # End of epochs

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Loss: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    state = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sheduler": scheduler.state_dict(),
    }
    return model, state


if __name__ == "__main__":

    # initialization
    for dir_path in config.directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # train on the GPU or on the CPU, if a GPU is not available
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(device)

    # define training and validation data loaders
    train_loader = dataset.dataloader.get_dataloader(
        img_folder_path=config.img_folder, transform=dataset.utils.transform
    )
    checkpoint = False
    with torch.cuda.device(0):
        if checkpoint is True:
            checkpoint = torch.load(checkpoint)
            model = checkpoint["model"]
            start_epoch = checkpoint["epoch"]
            optimizer = checkpoint["optimizer"]

        else:
            # backbone = model.ResnextBackboneWithFPN()
            backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                "resnet101", pretrained=True
            )
            anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),)
            )
            # put the pieces together inside a FasterRCNN model
            model = FasterRCNN(
                backbone,
                num_classes=11,
                min_size=300,
                max_size=300,
            )

            optimizer_ft = torch.optim.SGD(
                model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005
            )
            # and a learning rate scheduler
            exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer_ft, step_size=5, gamma=0.1
            )

        model = model.to(device)

        cudnn.benchmark = True

        # training the model
        model_ft, state = train_model(
            model,
            train_loader,
            optimizer_ft,
            exp_lr_scheduler,
            num_epochs=15,
        )
