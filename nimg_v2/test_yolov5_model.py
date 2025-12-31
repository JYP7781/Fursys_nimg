#!/usr/bin/env python3
"""
YOLOv5 모델 (best.pt) 테스트
"""

import sys
import os
from pathlib import Path

# nimg 경로 추가
NIMG_PATH = Path(__file__).parent.parent / 'nimg' / 'nimg'
sys.path.insert(0, str(NIMG_PATH))

import torch
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_yolov5_best():
    """YOLOv5 best.pt 모델 테스트"""
    MODEL_PATH = "/root/fursys_imgprosessing_ws/src/nimg/best.pt"
    TEST_IMAGE = "/root/fursys_imgprosessing_ws/20251208_155531_output/color_000000.png"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Loading YOLOv5 model: {MODEL_PATH}")

    # YOLOv5 모델 직접 로드
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = checkpoint['model'].float().to(device)
    model.eval()

    # 모델 정보
    logger.info(f"Model type: {type(model)}")
    if hasattr(model, 'names'):
        logger.info(f"Classes: {len(model.names)}")
        logger.info(f"Class names sample: {list(model.names.values())[:5] if isinstance(model.names, dict) else model.names[:5]}")

    stride = int(model.stride.max()) if hasattr(model, 'stride') else 32
    logger.info(f"Stride: {stride}")

    # 이미지 전처리
    image = cv2.imread(TEST_IMAGE)
    logger.info(f"Image shape: {image.shape}")

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

    im = letterbox(image, 640, stride)
    logger.info(f"Letterbox shape: {im.shape}")

    # 전처리
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device).float()
    im /= 255.0
    if len(im.shape) == 3:
        im = im[None]

    logger.info(f"Input tensor: shape={im.shape}, dtype={im.dtype}")

    # 추론
    with torch.no_grad():
        pred = model(im)

    # 출력 분석
    if isinstance(pred, tuple):
        pred = pred[0]

    logger.info(f"Output shape: {pred.shape}")
    logger.info(f"Output range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")

    # YOLOv5 출력: (batch, num_boxes, 5 + nc)
    # 5 = (x, y, w, h, obj_conf)
    x = pred[0]
    obj_conf = x[:, 4]
    class_conf = x[:, 5:]

    logger.info(f"Objectness - min: {obj_conf.min().item():.4f}, max: {obj_conf.max().item():.4f}")

    # 신뢰도 필터링 테스트
    for thresh in [0.5, 0.25, 0.1, 0.05, 0.01]:
        mask = obj_conf > thresh
        logger.info(f"obj_conf > {thresh}: {mask.sum().item()} boxes")

    # 높은 objectness의 박스 확인
    if obj_conf.max() > 0.01:
        top_k = 5
        top_obj, top_idx = obj_conf.topk(min(top_k, len(obj_conf)))
        logger.info(f"\nTop {top_k} objectness boxes:")
        for i, (obj, idx) in enumerate(zip(top_obj, top_idx)):
            box = x[idx, :4].cpu().numpy()
            cls_conf, cls_idx = class_conf[idx].max(0)
            total_conf = obj.item() * cls_conf.item()
            logger.info(f"  [{i}] obj={obj.item():.4f}, cls_conf={cls_conf.item():.4f}, total={total_conf:.4f}, class={cls_idx.item()}")


def test_hub_load():
    """torch.hub로 YOLOv5 로드"""
    MODEL_PATH = "/root/fursys_imgprosessing_ws/src/nimg/best.pt"
    TEST_IMAGE = "/root/fursys_imgprosessing_ws/20251208_155531_output/color_000000.png"

    logger.info("\n" + "=" * 60)
    logger.info("torch.hub로 YOLOv5 모델 로드")

    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
        model.conf = 0.25
        model.iou = 0.45

        logger.info(f"Classes: {len(model.names)}")

        image = cv2.imread(TEST_IMAGE)
        results = model(image)

        logger.info(f"Detections: {len(results.xyxy[0])}")

        for det in results.xyxy[0][:5]:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            logger.info(f"  - {model.names[int(cls)]}: {conf:.4f}")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_yolov5_best()
    test_hub_load()
