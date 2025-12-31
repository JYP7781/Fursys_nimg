#!/usr/bin/env python3
"""
nimg_v2 종합 테스트 - 다양한 conf로 원본 데이터 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import logging
from glob import glob

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_comprehensive():
    """종합 테스트"""
    from nimg_v2.detection.yolo_detector import YOLODetector

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    DATA_DIRS = [
        "/root/fursys_imgprosessing_ws/src/nimg/img_data/20240525",
        "/root/fursys_imgprosessing_ws/src/nimg/img_data/20240529",
    ]

    logger.info("=" * 60)
    logger.info("nimg_v2 종합 테스트")
    logger.info("=" * 60)

    # 다양한 conf로 테스트
    for conf in [0.5, 0.25, 0.1]:
        logger.info(f"\n{'='*60}")
        logger.info(f"conf_threshold = {conf}")
        logger.info("=" * 60)

        detector = YOLODetector(
            model_path=MODEL_PATH,
            conf_threshold=conf,
            iou_threshold=0.45,
            img_size=640,
            half=False
        )
        detector.warmup()

        for data_dir in DATA_DIRS:
            image_files = sorted(glob(f"{data_dir}/*.png"))[:15]
            total = 0

            for img_path in image_files:
                image = cv2.imread(img_path)
                if image is None:
                    continue

                detections = detector.detect(image)
                n = len(detections)
                total += n

                if n > 0:
                    names = [d.class_name for d in detections[:3]]
                    logger.info(f"  {Path(img_path).name}: {n} ({', '.join(names)})")

            logger.info(f"{Path(data_dir).name}: 총 {total}개 탐지")


def test_compared_folders():
    """다른 폴더들 비교"""
    from nimg_v2.detection.yolo_detector import YOLODetector

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"

    detector = YOLODetector(
        model_path=MODEL_PATH,
        conf_threshold=0.1,  # 낮은 임계값
        iou_threshold=0.45,
        img_size=640,
        half=False
    )
    detector.warmup()

    logger.info("\n" + "=" * 60)
    logger.info("폴더별 탐지 비교 (conf=0.1)")
    logger.info("=" * 60)

    folders = [
        ("/root/fursys_imgprosessing_ws/src/nimg/img_data/20240525", "원본 20240525"),
        ("/root/fursys_imgprosessing_ws/src/nimg/img_data/20240529", "원본 20240529"),
        ("/root/fursys_imgprosessing_ws/20251208_155531_output", "카메라 155531"),
        ("/root/fursys_imgprosessing_ws/20251208_161246_output", "카메라 161246"),
    ]

    for folder, name in folders:
        images = sorted(glob(f"{folder}/*.png"))[:20]
        total = 0
        detected_images = 0

        for img_path in images:
            image = cv2.imread(img_path)
            if image is None:
                continue

            detections = detector.detect(image)
            n = len(detections)
            if n > 0:
                total += n
                detected_images += 1

        logger.info(f"{name}: {detected_images}/{len(images)} images, {total} total detections")


if __name__ == "__main__":
    test_comprehensive()
    test_compared_folders()
