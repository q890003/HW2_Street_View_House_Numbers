#!/usr/bin/python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json
import random
import os
import config


def visualize_detetion_result(image_pth, label):
    """
    image_pth: Path of an image.
    label: Dictionary of bounding boxes and labels. {label: [label1, label2, ...], 'bbox':[bbox1, bbox2, ...]}
           Foramt of each box, bbox_n: [y1, x1, y2, x2]
    """
    fig, ax = plt.subplots(1)
    # Display the image
    im = np.array(Image.open(image_pth), dtype=np.uint8)
    ax.imshow(im)

    # transform label to {num1:(label1,matplotlib.patches(bbox1)),
    #                     num2:(label2,matplotlib.patches(bbox2)),...}
    rectangles = {
        i: (
            num,
            patches.Rectangle(
                (rec[1], rec[0]),
                (rec[3] - rec[1]),
                (rec[2] - rec[0]),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            ),
        )
        for i, (num, rec) in enumerate(zip(label["label"], label["bbox"]))
    }

    # annotate bounding box
    for r in rectangles:
        ax.add_artist(rectangles[r][1])
        rx, ry = rectangles[r][1].get_xy()
        cx = rx + rectangles[r][1].get_width() / 2.0
        ax.annotate(
            rectangles[r][0],
            (cx, ry),
            color="b",
            weight="bold",
            fontsize=20,
            ha="center",
            va="center",
        )

    plt.show()


def visualize_img(image, label):
    """
    image_pth: Path of an image.
    label: Dictionary of bounding boxes and labels. {label: [label1, label2, ...], 'bbox':[bbox1, bbox2, ...]}
           Foramt of each box, bbox_n: [y1, x1, y2, x2]
    """
    fig, ax = plt.subplots(1)
    # Display the image
    im = np.array(image)
    ax.imshow(im)

    # transform label to {num1:(label1,matplotlib.patches(bbox1)),
    #                     num2:(label2,matplotlib.patches(bbox2)),...}
    rectangles = {
        i: (
            num,
            patches.Rectangle(
                (rec[1], rec[0]),
                (rec[3] - rec[1]),
                (rec[2] - rec[0]),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            ),
        )
        for i, (num, rec) in enumerate(zip(label["label"], label["bbox"]))
    }

    # annotate bounding box
    for r in rectangles:
        ax.add_artist(rectangles[r][1])
        rx, ry = rectangles[r][1].get_xy()
        cx = rx + rectangles[r][1].get_width() / 2.0
        ax.annotate(
            rectangles[r][0],
            (cx, ry),
            color="b",
            weight="bold",
            fontsize=20,
            ha="center",
            va="center",
        )

    plt.show()


if __name__ == "__main__":
    result = json.load(open("./results/0856148_score_th_0.5.json"))
    random_num = random.randint(0, len(result))

    label = result[random_num]
    image_path = os.path.join(config.test_img_folder, str(random_num + 1) + ".png")
    visualize_detetion_result(image_path, label)
    # for i in range(len(result)):
    #     image_path = os.path.join(config.test_img_folder,str(i + 1) + '.png')
    #     label = result[i]
    #     visualize_detetion_result(image_path, label)
