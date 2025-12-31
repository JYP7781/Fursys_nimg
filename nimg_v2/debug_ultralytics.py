#!/usr/bin/env python3
"""
ultralytics API 직접 사용 테스트
"""

import sys
import os
from pathlib import Path

import torch
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_ultralytics_api():
    """ultralytics YOLO API 직접 사용"""
    from ultralytics import YOLO

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    TEST_IMAGE_PATH = "/root/fursys_imgprosessing_ws/20251208_155531_output/color_000000.png"

    logger.info("ultralytics YOLO API 테스트")

    # 모델 로드
    model = YOLO(MODEL_PATH)
    logger.info(f"Model loaded: {len(model.names)} classes")

    # 이미지 로드
    image = cv2.imread(TEST_IMAGE_PATH)
    logger.info(f"Image shape: {image.shape}")

    # 탐지 (다양한 conf 테스트)
    for conf in [0.5, 0.25, 0.1, 0.05, 0.01]:
        results = model(image, conf=conf, verbose=False)
        boxes = results[0].boxes
        logger.info(f"conf={conf}: {len(boxes)} detections")

        if len(boxes) > 0 and conf == 0.25:
            for i, box in enumerate(boxes[:5]):
                xyxy = box.xyxy[0].cpu().numpy()
                c = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                logger.info(f"  [{i}] {name} (class={cls}): conf={c:.4f}, box={xyxy}")

    # 결과 시각화
    results = model(image, conf=0.5)
    result_img = results[0].plot()
    cv2.imwrite('/root/fursys_imgprosessing_ws/src/nimg_v2/ultralytics_result.png', result_img)
    logger.info("Result saved to ultralytics_result.png")


def test_model_predict_mode():
    """모델의 predict 모드 확인"""
    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    TEST_IMAGE_PATH = "/root/fursys_imgprosessing_ws/20251208_155531_output/color_000000.png"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 직접 모델 로드
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = checkpoint['model'].to(device).eval()

    logger.info(f"Model type: {type(model)}")

    # 모델의 fuse() 메서드 확인
    if hasattr(model, 'fuse'):
        logger.info("Fusing model...")
        model = model.fuse()

    # 추론 모드 확인
    if hasattr(model, 'model'):
        for m in model.model:
            if hasattr(m, 'export'):
                logger.info(f"Layer {type(m).__name__}: export={m.export if hasattr(m, 'export') else 'N/A'}")

    # forward 메서드 확인
    logger.info(f"Model forward: {model.forward.__doc__ if hasattr(model.forward, '__doc__') else 'N/A'}")


if __name__ == "__main__":
    test_ultralytics_api()
    print()
    test_model_predict_mode()
