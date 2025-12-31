#!/usr/bin/env python3
"""
모델 출력 형식 디버깅 스크립트
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'nimg_v2'))

import torch
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def debug_model_output():
    """모델 출력 형식 확인"""
    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    TEST_IMAGE_PATH = "/root/fursys_imgprosessing_ws/20251208_155531_output/color_000000.png"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    logger.info("모델 로드 중...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    logger.info(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")

    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model'].float()
            logger.info(f"Model type: {type(model)}")
            logger.info(f"Model class: {model.__class__.__name__}")

            if hasattr(model, 'names'):
                logger.info(f"Class names count: {len(model.names)}")
                # 처음 10개 클래스명 출력
                if isinstance(model.names, dict):
                    for i, (k, v) in enumerate(list(model.names.items())[:10]):
                        logger.info(f"  {k}: {v}")
                elif isinstance(model.names, list):
                    for i, name in enumerate(model.names[:10]):
                        logger.info(f"  {i}: {name}")

            if hasattr(model, 'stride'):
                logger.info(f"Stride: {model.stride}")

            if hasattr(model, 'yaml'):
                logger.info(f"YAML config: {model.yaml}")

    model = checkpoint['model'].float().to(device).eval()

    # 이미지 로드 및 전처리
    logger.info("이미지 전처리 중...")
    image = cv2.imread(TEST_IMAGE_PATH)
    logger.info(f"Original image shape: {image.shape}")

    # letterbox
    def letterbox(im, new_shape=640, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return im

    im = letterbox(image, 640, 32)
    logger.info(f"Letterbox image shape: {im.shape}")

    # BGR -> RGB, HWC -> CHW
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device).float() / 255.0
    if len(im.shape) == 3:
        im = im[None]

    logger.info(f"Tensor shape: {im.shape}")

    # 추론
    logger.info("추론 중...")
    with torch.no_grad():
        pred = model(im)

    # 출력 분석
    logger.info(f"Output type: {type(pred)}")

    if isinstance(pred, tuple):
        logger.info(f"Output is tuple with {len(pred)} elements")
        for i, p in enumerate(pred):
            if isinstance(p, torch.Tensor):
                logger.info(f"  pred[{i}] shape: {p.shape}, dtype: {p.dtype}")
                logger.info(f"  pred[{i}] min: {p.min():.4f}, max: {p.max():.4f}")
            else:
                logger.info(f"  pred[{i}] type: {type(p)}")
    elif isinstance(pred, torch.Tensor):
        logger.info(f"Output shape: {pred.shape}")
        logger.info(f"Output dtype: {pred.dtype}")
        logger.info(f"Output min: {pred.min():.4f}, max: {pred.max():.4f}")

        # 출력 구조 분석
        # YOLOv5: (batch, num_detections, 5 + num_classes)
        # 5 = (x_center, y_center, width, height, objectness)
        if len(pred.shape) == 3:
            batch, num_det, features = pred.shape
            logger.info(f"Batch: {batch}, Detections: {num_det}, Features: {features}")

            # 첫 번째 탐지 샘플
            sample = pred[0, :5, :]
            logger.info(f"First 5 detections sample:\n{sample}")

            # objectness (5번째 열) 확인
            objectness = pred[0, :, 4]
            logger.info(f"Objectness scores - min: {objectness.min():.4f}, max: {objectness.max():.4f}")
            logger.info(f"Objectness > 0.5: {(objectness > 0.5).sum()} boxes")
            logger.info(f"Objectness > 0.25: {(objectness > 0.25).sum()} boxes")

            # 클래스 확률 확인
            if features > 5:
                class_probs = pred[0, :, 5:]
                logger.info(f"Class probs shape: {class_probs.shape}")
                logger.info(f"Class probs - min: {class_probs.min():.4f}, max: {class_probs.max():.4f}")

                # objectness가 높은 박스의 클래스 확인
                high_obj_mask = objectness > 0.1
                if high_obj_mask.sum() > 0:
                    high_obj_class = class_probs[high_obj_mask]
                    max_class_conf, max_class_idx = high_obj_class.max(dim=1)
                    logger.info(f"High objectness boxes - class indices: {max_class_idx[:10].cpu().numpy()}")
                    logger.info(f"High objectness boxes - class confs: {max_class_conf[:10].cpu().numpy()}")


if __name__ == "__main__":
    debug_model_output()
