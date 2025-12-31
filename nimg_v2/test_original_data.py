#!/usr/bin/env python3
"""
원본 img_data 폴더의 이미지로 테스트
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


def test_with_original_data():
    """원본 img_data 이미지로 테스트"""
    from ultralytics import YOLO

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    DATA_DIRS = [
        "/root/fursys_imgprosessing_ws/src/nimg/img_data/20240525",
        "/root/fursys_imgprosessing_ws/src/nimg/img_data/20240529",
    ]

    logger.info("=" * 60)
    logger.info("원본 img_data 이미지로 테스트")
    logger.info("=" * 60)

    # 모델 로드
    model = YOLO(MODEL_PATH)
    logger.info(f"Model loaded: {len(model.names)} classes")

    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            continue

        logger.info(f"\n테스트 폴더: {data_dir}")

        # 이미지 찾기
        image_files = []
        for ext in ['*.png', '*.jpg']:
            image_files.extend(glob(f"{data_dir}/{ext}"))
            image_files.extend(glob(f"{data_dir}/**/{ext}", recursive=True))

        image_files = sorted(set(image_files))[:20]  # 최대 20개
        logger.info(f"테스트할 이미지: {len(image_files)}개")

        total_detections = 0
        detection_images = []

        for img_path in image_files:
            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue

                # 다양한 conf 테스트
                for conf in [0.5, 0.25, 0.1]:
                    results = model(image, conf=conf, verbose=False)
                    n = len(results[0].boxes)
                    if n > 0:
                        total_detections += n
                        detection_images.append((img_path, n, conf))
                        logger.info(f"  {Path(img_path).name}: {n} detections (conf={conf})")

                        # 상세 정보
                        for box in results[0].boxes[:3]:
                            c = float(box.conf[0])
                            cls = int(box.cls[0])
                            name = model.names[cls]
                            logger.info(f"    - {name}: {c:.4f}")
                        break  # 탐지되면 다음 이미지로

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        logger.info(f"\n{data_dir} 결과: {total_detections} detections in {len(detection_images)} images")

        # 결과 이미지 저장 (첫 번째 탐지 이미지)
        if detection_images:
            img_path = detection_images[0][0]
            image = cv2.imread(img_path)
            results = model(image, conf=0.25)
            result_img = results[0].plot()
            output_path = f"/root/fursys_imgprosessing_ws/src/nimg_v2/original_data_result_{Path(data_dir).name}.png"
            cv2.imwrite(output_path, result_img)
            logger.info(f"결과 이미지 저장: {output_path}")


def test_with_best_pt():
    """기존 best.pt 모델로도 테스트"""
    from ultralytics import YOLO

    MODEL_PATH = "/root/fursys_imgprosessing_ws/src/nimg/best.pt"
    DATA_DIR = "/root/fursys_imgprosessing_ws/src/nimg/img_data/20240525"

    if not os.path.exists(MODEL_PATH):
        logger.warning(f"best.pt not found: {MODEL_PATH}")
        return

    logger.info("\n" + "=" * 60)
    logger.info("기존 best.pt 모델로 테스트")
    logger.info("=" * 60)

    model = YOLO(MODEL_PATH)
    logger.info(f"Model loaded: {len(model.names)} classes")

    # 첫 번째 이미지 테스트
    image_files = glob(f"{DATA_DIR}/*.png")[:5]
    for img_path in image_files:
        image = cv2.imread(img_path)
        results = model(image, conf=0.25, verbose=False)
        n = len(results[0].boxes)
        logger.info(f"{Path(img_path).name}: {n} detections")

        if n > 0:
            for box in results[0].boxes[:3]:
                c = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                logger.info(f"  - {name}: {c:.4f}")


def view_sample_image():
    """샘플 이미지 확인"""
    import base64

    img_path = "/root/fursys_imgprosessing_ws/src/nimg/img_data/20240525/n0_164413_235568.png"
    image = cv2.imread(img_path)
    logger.info(f"\n샘플 이미지: {img_path}")
    logger.info(f"Shape: {image.shape}")

    # 작은 크기로 저장
    resized = cv2.resize(image, (640, 480))
    cv2.imwrite("/root/fursys_imgprosessing_ws/src/nimg_v2/sample_original.png", resized)
    logger.info("샘플 이미지 저장: sample_original.png")


if __name__ == "__main__":
    view_sample_image()
    test_with_original_data()
    test_with_best_pt()
