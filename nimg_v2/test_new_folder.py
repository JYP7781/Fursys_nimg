#!/usr/bin/env python3
"""
20251208_161246_output 폴더 이미지로 테스트
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'nimg_v2'))

import cv2
import logging
from glob import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_with_ultralytics():
    """ultralytics YOLO로 테스트"""
    from ultralytics import YOLO

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    DATA_DIR = "/root/fursys_imgprosessing_ws/20251208_161246_output"

    logger.info("=" * 60)
    logger.info("ultralytics YOLO API로 테스트")
    logger.info(f"폴더: {DATA_DIR}")
    logger.info("=" * 60)

    model = YOLO(MODEL_PATH)
    logger.info(f"모델 로드 완료: {len(model.names)} 클래스")

    # 여러 이미지 테스트 (10개 간격으로 샘플링)
    image_files = sorted(glob(f"{DATA_DIR}/color_*.png"))
    sample_images = image_files[::100][:20]  # 100개 간격으로 20개
    logger.info(f"테스트 이미지: {len(sample_images)}개")

    total_detections = 0
    detected_images = []

    for img_path in sample_images:
        image = cv2.imread(img_path)
        results = model(image, conf=0.5, verbose=False)
        n = len(results[0].boxes)
        total_detections += n

        if n > 0:
            detected_images.append((img_path, n, results[0]))
            logger.info(f"✓ {Path(img_path).name}: {n}개 탐지!")
            for box in results[0].boxes[:3]:
                c = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                logger.info(f"    - {name}: {c:.4f}")

    logger.info(f"\n총 탐지: {total_detections}개 ({len(detected_images)}개 이미지에서)")

    # 탐지된 첫 이미지 저장
    if detected_images:
        img_path, n, result = detected_images[0]
        result_img = result.plot()
        output_path = "/root/fursys_imgprosessing_ws/src/nimg_v2/detection_result_161246.png"
        cv2.imwrite(output_path, result_img)
        logger.info(f"결과 이미지 저장: {output_path}")

    return total_detections > 0


def test_with_nimg_v2():
    """nimg_v2 YOLODetector로 테스트"""
    from nimg_v2.detection.yolo_detector import YOLODetector

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    DATA_DIR = "/root/fursys_imgprosessing_ws/20251208_161246_output"

    logger.info("\n" + "=" * 60)
    logger.info("nimg_v2 YOLODetector로 테스트")
    logger.info("=" * 60)

    detector = YOLODetector(
        model_path=MODEL_PATH,
        conf_threshold=0.5,
        iou_threshold=0.45,
        img_size=640,
        half=True,
        use_yolov5_backend=True
    )

    logger.info(f"탐지기 초기화 완료")
    logger.info(f"  - 클래스 수: {len(detector.names)}")
    logger.info(f"  - 백엔드: {'YOLOv5' if detector._using_yolov5_backend else 'ultralytics'}")

    # 여러 이미지 테스트
    image_files = sorted(glob(f"{DATA_DIR}/color_*.png"))
    sample_images = image_files[::100][:20]

    total_detections = 0
    detected_images = []

    for img_path in sample_images:
        image = cv2.imread(img_path)
        detections = detector.detect(image)
        n = len(detections)
        total_detections += n

        if n > 0:
            detected_images.append((img_path, detections))
            logger.info(f"✓ {Path(img_path).name}: {n}개 탐지!")
            for det in detections[:3]:
                logger.info(f"    - {det.class_name}: {det.confidence:.4f}")

    logger.info(f"\n총 탐지: {total_detections}개 ({len(detected_images)}개 이미지에서)")

    # 탐지된 첫 이미지 저장
    if detected_images:
        img_path, detections = detected_images[0]
        image = cv2.imread(img_path)
        result_img = detector.draw_detections(image, detections)
        output_path = "/root/fursys_imgprosessing_ws/src/nimg_v2/nimg_v2_result_161246.png"
        cv2.imwrite(output_path, result_img)
        logger.info(f"결과 이미지 저장: {output_path}")

    return total_detections > 0


if __name__ == "__main__":
    # ultralytics 테스트
    result1 = test_with_ultralytics()

    # nimg_v2 테스트
    result2 = test_with_nimg_v2()

    print("\n" + "=" * 60)
    print(f"ultralytics YOLO: {'성공' if result1 else '실패'}")
    print(f"nimg_v2 YOLODetector: {'성공' if result2 else '실패'}")
    print("=" * 60)
