# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np

def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious



pred_file = 'cvf1val_e13.pkl.json'
# gt_file = '/Users/chenbo/Google_Drive/Projects/PoseEst/dataset/images/validf3.json'   # Mac
gt_file = '/home/bochen/google_drive/Projects/PoseEst/dataset/images/validf1.json'  # Linux

with open(pred_file, 'r') as f:
    pred_l = json.load(f)

with open(gt_file, 'r') as f:
    gt_l = json.load(f)

print('aaa',len(pred_l))
print('bbb',len(gt_l))

iou = np.zeros(2000)

for i in range(2000):
    pred = np.array([pred_l[i]['bbox']])
    gt = np.array(gt_l[i]['box'])
    pred[0,2:4] = pred[0,0:2] + pred[0,2:4]
    iou[i] = bbox_overlaps(pred, gt)
    print(i, iou[i])

print(np.mean(iou))
print(np.median(iou))

