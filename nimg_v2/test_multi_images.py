#!/usr/bin/env python3
"""
여러 이미지에서 객체 탐지 테스트
"""

import sys
import os
from pathlib import Path

import torch
import cv2
import numpy as np
import logging
from glob import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_multiple_images():
    """여러 이미지에서 테스트"""
    from ultralytics import YOLO

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    DATA_DIR = "/root/fursys_imgprosessing_ws/20251208_155531_output"

    logger.info("여러 이미지 테스트")

    # 모델 로드
    model = YOLO(MODEL_PATH)
    logger.info(f"Model loaded: {len(model.names)} classes")

    # 여러 이미지 테스트
    image_files = sorted(glob(f"{DATA_DIR}/color_*.png"))[:100:10]  # 10개 간격으로 10개
    logger.info(f"Testing {len(image_files)} images")

    total_detections = 0
    for img_path in image_files:
        image = cv2.imread(img_path)
        results = model(image, conf=0.25, verbose=False)
        boxes = results[0].boxes
        n = len(boxes)
        total_detections += n
        if n > 0:
            logger.info(f"{Path(img_path).name}: {n} detections")
            for box in boxes[:3]:
                c = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                logger.info(f"  - {name}: {c:.4f}")

    logger.info(f"\nTotal detections: {total_detections}")

    if total_detections == 0:
        logger.warning("No detections in any image!")
        logger.info("\n이 이미지들에 해당 클래스의 객체가 없을 수 있습니다.")
        logger.info("다른 테스트 이미지가 필요합니다.")


def test_different_data_dir():
    """다른 데이터 폴더 확인"""
    from ultralytics import YOLO

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"

    model = YOLO(MODEL_PATH)

    # 다른 데이터 폴더 확인
    data_dirs = [
        "/root/fursys_imgprosessing_ws/20251208_161246_output",
        "/root/fursys_imgprosessing_ws/img_data",
    ]

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            logger.info(f"\n검사 중: {data_dir}")
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(glob(f"{data_dir}/**/{ext}", recursive=True))
                image_files.extend(glob(f"{data_dir}/{ext}"))

            if image_files:
                logger.info(f"  {len(image_files)} images found")

                # 첫 10개 이미지 테스트
                for img_path in image_files[:10]:
                    try:
                        image = cv2.imread(img_path)
                        if image is not None:
                            results = model(image, conf=0.25, verbose=False)
                            n = len(results[0].boxes)
                            if n > 0:
                                logger.info(f"  {Path(img_path).name}: {n} detections!")
                                return  # 탐지 성공시 종료
                    except:
                        pass
            else:
                logger.info(f"  No images found")


def test_with_nimg_best():
    """기존 nimg의 best.pt 모델로 테스트"""
    from ultralytics import YOLO

    MODEL_PATH = "/root/fursys_imgprosessing_ws/src/nimg/best.pt"
    TEST_IMAGE = "/root/fursys_imgprosessing_ws/20251208_155531_output/color_000000.png"

    if not os.path.exists(MODEL_PATH):
        logger.warning(f"best.pt not found: {MODEL_PATH}")
        return

    logger.info(f"\n기존 nimg best.pt 모델 테스트")

    try:
        model = YOLO(MODEL_PATH)
        logger.info(f"Model loaded: {len(model.names)} classes")

        image = cv2.imread(TEST_IMAGE)
        results = model(image, conf=0.25, verbose=False)
        n = len(results[0].boxes)
        logger.info(f"Detections: {n}")

        if n > 0:
            for box in results[0].boxes[:5]:
                c = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                logger.info(f"  - {name}: {c:.4f}")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    test_multiple_images()
    test_different_data_dir()
    test_with_nimg_best()
