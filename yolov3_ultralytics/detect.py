import random
import time
from pathlib import Path
import os

import numpy as np
import torch
import cv2
import imutils
import streamlit as st
import matplotlib.pyplot as plt

from .models import Darknet, load_darknet_weights
from .utils.parse_config import parse_data_cfg
from .utils.utils import load_classes, non_max_suppression, plot_one_box, scale_coords
from .utils.datasets import letterbox

archs = {
    # "YOLOv3-tiny-608-ultralytics": {
    #     "cfg": "my_cfg/yolov3-tiny-corn.cfg",
    #     "weights": "backup/yolov3-tiny-corn-608-ultralytics.pt",
    #     "default_img_size": 608,
    # },
    "YOLOv3-tiny-416": {
        "cfg": "my_cfg/yolov3-tiny-corn.cfg",
        "weights": "backup/yolov3-tiny-corn-416_best.weights",
        "default_img_size": 416,
    },
    "YOLOv3-tiny-608": {
        "cfg": os.path.abspath("my_cfg/yolov3-tiny-corn.cfg"),
        "weights": os.path.abspath("backup/yolov3-tiny-corn-608_best.weights"),
        "default_img_size": 608,
    },
    "YOLOv3-tiny-3l-608": {
        "cfg": "my_cfg/yolov3-tiny_3l-corn.cfg",
        "weights": "backup/yolov3-tiny_3l-corn_best.weights",
        "default_img_size": 608,
    }
}


def load_model(cfg_path, weights_path, img_size, device):
    model = Darknet(cfg_path, img_size)
    # Load weights
    if weights_path.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(
            weights_path, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights_path)
    # Eval mode
    model.to(device).eval()
    return model


def load_image(img_path, img_size):
    original_img = cv2.imread(img_path)
    # Padded resize
    resized_img = letterbox(original_img, new_shape=img_size)[0]
    # Normalize RGB
    resized_img = resized_img[:, :, ::-1]  # BGR to RGB
    resized_img = np.ascontiguousarray(
        resized_img, dtype=np.float32)  # uint8 to fp16/fp32
    resized_img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return original_img, resized_img


def detect_yolo_annotations(model, img_path, img_size, class_list,
                            device='cpu',
                            conf_thresh=0.25, nms_thresh=0.4):
    original_img, resized_img = load_image(img_path, img_size)
    # Inference
    x = torch.from_numpy(resized_img).to(device)
    x = x.permute(2, 0, 1)
    if x.ndimension() == 3:
        x = x.unsqueeze(0)

    pred = model(x)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thresh, nms_thresh, agnostic=True)
    # Process detections
    shapes = []
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            # print(resized_img.shape)
            # print(original_img.shape)
            det[:, :4] = scale_coords(
                resized_img.shape, det[:, :4], original_img.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                label = class_list[int(cls)]
                # print(xyxy)
                xyxy = [int(v) for v in xyxy]
                xmin = min(xyxy[0], xyxy[2])
                xmax = max(xyxy[0], xyxy[2])
                ymin = min(xyxy[1], xyxy[3])
                ymax = max(xyxy[1], xyxy[3])

                points = [(xmin, ymin), (xmax, ymin),
                          (xmax, ymax), (xmin, ymax)]
                shapes.append((label, points, None, None, False))

    return shapes


# Initialize model
if __name__ == "__main__":
    device = "cpu"
    arch = "YOLOv3-tiny-608"
    cfg_path = archs[arch]["cfg"]
    weights_path = archs[arch]["weights"]
    img_size = 608
    model = load_model(cfg_path, weights_path, img_size, device)

    shapes = detect_yolo_annotations(model,
                                    "/media/minhduc0711/Libraries/Codes/telesense/corn-quality/labelImg/dummy_data/IMG_0135.jpg",
                                    img_size,
                                    Path("../dummy_data/classes.txt").read_text().splitlines())
    print(shapes)
