from __future__ import annotations

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
from os.path import isfile, splitext, basename
from PIL import Image, ImageDraw

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from utils.torch_utils import select_device

from mapping import mapped_names, category_map, NO_LABEL


class BoundingBox:
    def __init__(self, x: float = 0, y: float = 0, w: float = 0, h: float = 0):
        self.left: float = x - w / 2
        self.top: float = y - h / 2
        self.right: float = x + w / 2
        self.bottom: float = y + h / 2
        self.label: int = -1
        self.matched: bool = False
        self.score: float = 0

    @staticmethod
    def from_string(string: str) -> BoundingBox:
        """
        Create a rectangle from a darknet label line
        """
        bbox = BoundingBox()
        idx, center_x, center_y, width, height = string.strip().split(" ")
        center_x: float = float(center_x)
        center_y: float = float(center_y)
        width: float = float(width)
        height: float = float(height)
        bbox.left = center_x - width / 2
        bbox.top = center_y - height / 2
        bbox.right = center_x + width / 2
        bbox.bottom = center_y + height / 2
        bbox.label = int(idx)
        return bbox

    def width(self) -> float:
        """
        Compute the width of the rectangle
        """
        return self.right - self.left

    def height(self) -> float:
        """
        Compute the height of the rectangle
        """
        return self.bottom - self.top

    def center(self) -> tuple(float, float):
        """
        Compute the center of the rectangle
        """
        return (self.width() / 2, self.height() / 2)

    def area(self) -> float:
        """
        Compute the area of the rectangle
        """
        return self.width() * self.height()

    def iou(self, other: BoundingBox) -> float:
        """
        Compute the intersection over union between self and other

        Parameters
        ----------------
        other: BoundingBox
        """
        if self.label != other.label:
            return 0
        inter_rect = BoundingBox()
        inter_rect.left = max(self.left, other.left)
        inter_rect.top = max(self.top, other.top)
        inter_rect.right = min(self.right, other.right)
        inter_rect.bottom = min(self.bottom, other.bottom)
        # import pdb

        # pdb.set_trace()
        intersection = max(0, inter_rect.area())
        union = self.area() + other.area() - intersection
        return intersection / union

    def __repr__(self) -> str:
        return "BoundingBox()"

    def __str__(self) -> str:
        return (
                "label: "
                + str(self.label)
                + ", ("
                + str(self.left)
                + ", "
                + str(self.top)
                + "), ("
                + str(self.right)
                + ", "
                + str(self.bottom)
                + "), matched:"
                + str(self.matched)
                + ", score: "
                + str(self.score)
        )


def detect_one_image(model, image_path, thresh=0.25, nms=0.45, imgsz=640, dataset='omnious'):
    detections = []
    image_path = str(Path(image_path))
    image_path = os.path.join('..', dataset, image_path)

    dataset = LoadImages(image_path, img_size=imgsz)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once # TODO: why ???
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, thresh, nms, classes=None, agnostic=False)

        for i, det in enumerate(pred):
            p, s, im0 = path, '', im0s
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size (original image size)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls_ in det:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    cls_ = mapped_names[int(cls_.item())]
                    detections.append((cls_, conf, xywh))

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help='model.yaml path')
    parser.add_argument("--conf", type=float, help="confidence threshold", default=0.25)
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument("--image", type=str, help="image to process")
    parser.add_argument("--iou", type=float, help="IoU threshold", default=0.5)
    parser.add_argument("--list", type=str, help="list with one image per file")
    parser.add_argument("--nms", type=float, help="NMS threshold", default=0.45)
    parser.add_argument("--weights", type=str, help="yolo weights")
    parser.add_argument("--img_size", type=int, default=640, help="image size, e.g. 640, 608, 512")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--dataset', default='', help='dataset, e.g. coco or omnious_210225')
    parser.add_argument("--draw", default=False)
    opt = parser.parse_args()

    # Step 1: Load model
    # Step 2: Open file
    # Step 3: Read file
    # Step 4: Make Prediction
    # Step 5: Measure f1-score
    weights, imgsz = opt.weights, opt.img_size

    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()

    names = model.module.names if hasattr(model, 'module') else model.names

    raw_results = {}
    for i, name in enumerate(mapped_names):
        raw_results[name.strip()] = {"tp": 0, "fp": 0, "fn": 0}

    with open(opt.list, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            image_path = str(Path(line))
            image_path = os.path.join('..', opt.dataset, image_path)
            image_path = os.path.abspath(image_path)
            line = image_path
            if not isfile(line):
                print(line, "does not exist")
                continue
            label_path = splitext(line.replace("images", "labels", 1))[0] + ".txt"
            results = detect_one_image(model, line, thresh=opt.conf, nms=opt.nms, imgsz=imgsz, dataset='omnious_210225')
            pred_bboxes = []
            for r in results:
                bbox = BoundingBox(r[2][0], r[2][1], r[2][2], r[2][3])
                bbox.score = r[1]

                bbox.label = names.index(r[0])
                bbox.label = category_map[bbox.label]
                pred_bboxes.append(bbox)
            if not isfile(label_path):
                print("WARNING: skipping", label_path)
                continue
            target_bboxes = []
            with open(label_path, "r") as label_file:
                labels = label_file.readlines()
                for label in labels:
                    bbox = BoundingBox.from_string(label)
                    bbox.label = category_map[bbox.label]
                    target_bboxes.append(bbox)

            if opt.draw:
                # == IMAGE LOADING == #
                im = Image.open(line).convert("RGB")
                width, height = im.size
                draw = ImageDraw.Draw(im)
                # =================== #

            # True positives: targets matched by predictions
            for target in target_bboxes:
                if target.matched:
                    continue

                # sort predictions by score, then by IOU with current target
                pred_bboxes.sort(key=lambda x: (x.score, target.iou(x)), reverse=True)
                for pred in pred_bboxes:
                    if (
                            not target.matched
                            and not pred.matched
                            and target.iou(pred) > opt.iou
                            and target.label == pred.label
                            and target.label != NO_LABEL
                    ):
                        target.matched = True
                        pred.matched = True
                        raw_results[mapped_names[target.label]]["tp"] += 1

                    if opt.draw:
                        # == DRAW TP PRED BOUNDING BOXES == #
                        draw.rectangle(
                            (
                                (pred.left * width, pred.top * height),
                                (pred.right * width, pred.bottom * height),
                            ),
                            outline="blue",
                            width=7,
                        )
                        # ================================= #

            # False negatives: targets not matched
            for target in target_bboxes:
                if target.label == NO_LABEL:
                    continue
                if not target.matched:
                    raw_results[mapped_names[target.label]]["fn"] += 1

                    if opt.draw:
                        # == DRAW FN PRED BOUNDING BOXES == #
                        draw.rectangle(
                            (
                                (target.left * width, target.top * height),
                                (target.right * width, target.bottom * height),
                            ),
                            outline="yellow",
                            width=3,
                        )
                        # ================================= #

            # False positives: predictions not matched
            for pred in pred_bboxes:
                if pred.label == NO_LABEL:
                    continue
                if not pred.matched:
                    raw_results[mapped_names[pred.label]]["fp"] += 1

                    if opt.draw:
                        # == DRAW FP PRED BOUNDING BOXES == #
                        draw.rectangle(
                            (
                                (pred.left * width, pred.top * height),
                                (pred.right * width, pred.bottom * height),
                            ),
                            outline="red",
                            width=3,
                        )
                        # ================================= #

            if opt.draw:
                # == SAVE IMAGE == #
                im.save("out/" + basename(line))
                # ================ #

        print("Computing Precision, Recall and F1-Score")
        # compute results
        total_tp: int = 0
        total_fp: int = 0
        total_fn: int = 0
        avg_prec: float = 0.0
        avg_recall: float = 0.0
        avg_f1: float = 0.0
        for k, v in raw_results.items():
            tp: int = v["tp"]
            fp: int = v["fp"]
            fn: int = v["fn"]
            total_tp += tp
            total_fp += fp
            total_fn += fn
            retrieved = tp + fp
            relevant = tp + fn
            if retrieved == 0:
                prec: float = 0.0
            else:
                prec = tp / retrieved
            if relevant == 0:
                recall: float = 0.0
            else:
                recall = tp / relevant
            if (prec + recall) == 0:
                f1: float = 0.0
            else:
                f1 = 2 * prec * recall / (prec + recall)

            avg_prec += prec
            avg_recall += recall
            avg_f1 += f1
            print(
                k,
                "=>",
                "tp: {}, fp: {}, fn: {}, prec: {:.4}, recall: {:.4}, f1: {:.4}".format(
                    tp, fp, fn, prec, recall, f1
                ),
            )
        total_prec = total_tp / (total_tp + total_fp)
        total_recall = total_tp / (total_tp + total_fn)
        total_f1 = 2 * total_prec * total_recall / (total_prec + total_recall)
        avg_prec /= len(raw_results)
        avg_recall /= len(raw_results)
        avg_f1 /= len(raw_results)
        print(
            "micro => prec: {:.4}, recall: {:.4}, f1: {:.4}".format(
                avg_prec, avg_recall, avg_f1
            )
        )
        print(
            "macro => prec: {:.4}, recall: {:.4}, f1: {:.4}".format(
                total_prec, total_recall, total_f1
            )
        )