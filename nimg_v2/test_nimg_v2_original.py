#!/usr/bin/env python3
"""
nimg_v2로 원본 img_data 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import logging
from glob import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_nimg_v2_with_original():
    """nimg_v2 YOLODetector로 원본 데이터 테스트"""
    from nimg_v2.detection.yolo_detector import YOLODetector

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    DATA_DIR = "/root/fursys_imgprosessing_ws/src/nimg/img_data/20240529"

    logger.info("=" * 60)
    logger.info("nimg_v2 YOLODetector로 원본 데이터 테스트")
    logger.info("=" * 60)

    # YOLODetector 초기화
    detector = YOLODetector(
        model_path=MODEL_PATH,
        conf_threshold=0.25,
        iou_threshold=0.45,
        img_size=640,
        half=False
    )
    detector.warmup()

    logger.info(f"Model loaded: {len(detector.model.names)} classes")

    # 이미지 테스트
    image_files = sorted(glob(f"{DATA_DIR}/*.png"))[:10]
    logger.info(f"테스트할 이미지: {len(image_files)}개")

    total_detections = 0
    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            continue

        detections = detector.detect(image)
        n = len(detections)
        total_detections += n

        if n > 0:
            logger.info(f"{Path(img_path).name}: {n} detections")
            for det in detections[:3]:
                logger.info(f"  - {det.class_name}: {det.confidence:.4f}")

    logger.info(f"\n총 탐지: {total_detections}개")

    # 결과 이미지 저장
    if total_detections > 0:
        # 탐지가 있는 이미지 찾기
        for img_path in image_files:
            image = cv2.imread(img_path)
            detections = detector.detect(image)
            if len(detections) > 0:
                result_img = detector.draw_detections(image, detections)
                cv2.imwrite("/root/fursys_imgprosessing_ws/src/nimg_v2/nimg_v2_result.png", result_img)
                logger.info("결과 저장: nimg_v2_result.png")
                break


if __name__ == "__main__":
    test_nimg_v2_with_original()
