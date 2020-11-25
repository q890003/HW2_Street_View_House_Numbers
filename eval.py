import torchvision
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.feature_pyramid_network import (
    LastLevelMaxPool,
    FeaturePyramidNetwork,
)
import torch
import torch.nn as nn

import time
import copy
import os
import json
import numpy as np
from PIL import Image

from collections import OrderedDict

import config
import model


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# load parameters to the model
experiment_name = "fastRCNN_op_SGD_lr_cylr_resnext50_32x8d_dataaug_shear_rotate_crop_noFlip__zoomin_loss8"
# backbone = torchvision.models.detection.backbone_utils. \
#                         resnet_fpn_backbone('resnet101', pretrained=True)
backbone = model.ResnextBackboneWithFPN()
model = FasterRCNN(
    backbone,
    num_classes=11,
    # rpn_anchor_generator=anchor_generator,
    # box_roi_pool=roi_pooler,
    min_size=300,
    max_size=300,
)
model.load_state_dict(torch.load(config.ckpt_dir + experiment_name))
model = model.to(device)
model.eval()

output = []
output_dict = {}
running_time = 0.0
score_threshold = 0.5
prediction_file_name = (
    config.result_pth + "0856148_score_th_{}_"
    "{experiment}.json".format(score_threshold, experiment=experiment_name)
)

print("Start evaluating. Result saved in {}".format(prediction_file_name))

with open(prediction_file_name, "w", encoding="utf-8") as json_f:
    # reorder order of testing image to [1.png, 2.png, ..., n.png]
    allFileList = os.listdir(config.test_img_folder)
    allFileList.sort(key=lambda x: int(x[:-4]))

    for img_name in allFileList:
        # load image
        if os.path.isfile(config.test_img_folder + img_name):
            img = Image.open(config.test_img_folder + img_name).convert("RGB")
            img = transforms.ToTensor()(img)
            img = img.unsqueeze(0)

            with torch.cuda.device(0):
                with torch.no_grad():
                    # predict
                    predict_begin = time.time()
                    pred = model(img.to(device))
                    running_time += time.time() - predict_begin
                    bbox = pred[0]["boxes"].cpu().numpy()
                    label = pred[0]["labels"].cpu().numpy()
                    score = pred[0]["scores"].cpu().numpy()

            # filter out score of bboxs higher than 0.5
            bbox = bbox[score > score_threshold]
            label = label[score > score_threshold]
            score = score[score > score_threshold]

            # reorder coordinate of the boxes in an image.
            # [x1, y1, x2, y2] -> [y1, x1, y2, x2]
            bbox = np.array(
                [
                    [bbox[i][1], bbox[i][0], bbox[i][3], bbox[i][2]]
                    for i in range(bbox.shape[0])
                ]
            )

            # update result and saved to json
            output_dict["bbox"] = bbox.tolist()
            output_dict["label"] = label.tolist()
            output_dict["score"] = score.tolist()
            output.append(output_dict.copy())

    json.dump(output, json_f)

print("Start evaluating. Result saved in {}".format(prediction_file_name))
average_time = running_time / len(allFileList)
m, s = average_time / 60, average_time % 60
print("average detection time: {:.0f}m {:.10f}s.".format(m, s))
