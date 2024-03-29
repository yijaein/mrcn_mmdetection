
import os
import cv2
import numpy as np
import colorsys
import random
from skimage.measure import find_contours

from matplotlib import patches
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import torch


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def visualize(result_path, image, predictions, colors=None, mask_display=True, class_names=None):
    image_shape = image.shape

    figsize = ((image_shape[1]) / 100.0, image_shape[0] / 100.0)

    fig, ax = plt.subplots(1, figsize=figsize, frameon=False)
    ax = fig.add_axes([0., 0., 1., 1.])

    line_alpha = 0.7
    linewidth = 3

    frame_dict = predictions['info']
    boxes = predictions['bboxes']
    scores = predictions['scores']
    labels = predictions['labels']

    if mask_display:
        masks =  predictions['masks']

    if class_names:
        N = len(class_names)
    else:
        N = len(boxes)

    # Generate random colors
    colors = colors or random_colors(N)

    # colors = self.compute_colors_for_labels(labels).tolist()

    before_label = -1
    count = 0
    for i, box, label, score in zip(range(len(boxes)), boxes, labels, scores):
        if before_label != -1 and before_label != label:
            count = 0

        objects = frame_dict[label]
        detect_object = objects[count]
        id = detect_object.id

        color = colors[id]

        [x1, y1, x2, y2] = box

        # print('box', box, label, score)

        # Bounding box
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=linewidth,
                              alpha=line_alpha, linestyle="-",
                              # edgecolor=color, facecolor=color)
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        if class_names:
            label_name = class_names[label]
            caption = "{} #{:d} ({:.2f})".format(label_name, id, score)

            ax.text(x1, y1 + 8, caption,
                    color='w', size=11, backgroundcolor="none")

        if mask_display:
            mask = masks[i]
            # mask = mask.reshape((mask.shape[1], mask.shape[2]))
            contours = find_contours(mask, 0.5)

            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor=color, edgecolor=color, alpha=0.5)
                ax.add_patch(p)

        before_label = label
        count += 1

    ax.imshow(image.astype(np.uint8))
    fig.canvas.draw()  # draw the canvas, cache the renderer
    X = np.array(fig.canvas.renderer._renderer)
    # result_image = cv2.cvtColor(X[:, :, :3], cv2.COLOR_BGR2RGB)

    plt.close(fig) # close figure memory

    cv2.imwrite(result_path, X)
