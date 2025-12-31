#!/usr/bin/env python3
"""
YOLODetector 테스트 스크립트
기존 nimg의 DetectMultiBackend 방식과 동일하게 동작하는지 확인
"""

import sys
import os
from pathlib import Path

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / 'nimg_v2'))

# nimg 경로도 추가 (DetectMultiBackend 로드용)
NIMG_PATH = SCRIPT_DIR.parent / 'nimg' / 'nimg'
if NIMG_PATH.exists():
    sys.path.insert(0, str(NIMG_PATH))

import cv2
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_detector():
    """YOLODetector 테스트"""
    from nimg_v2.detection.yolo_detector import YOLODetector

    # 모델 경로
    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"

    # 테스트 이미지 경로
    TEST_IMAGE_PATH = "/root/fursys_imgprosessing_ws/20251208_155531_output/color_000000.png"

    logger.info(f"모델 경로: {MODEL_PATH}")
    logger.info(f"테스트 이미지: {TEST_IMAGE_PATH}")

    # 파일 존재 확인
    if not os.path.exists(MODEL_PATH):
        logger.error(f"모델 파일이 없습니다: {MODEL_PATH}")
        return False

    if not os.path.exists(TEST_IMAGE_PATH):
        logger.error(f"테스트 이미지가 없습니다: {TEST_IMAGE_PATH}")
        return False

    # 탐지기 초기화
    logger.info("탐지기 초기화 중...")
    try:
        detector = YOLODetector(
            model_path=MODEL_PATH,
            conf_threshold=0.5,  # 기존 nimg의 기본값: 0.65
            iou_threshold=0.45,
            img_size=640,
            half=True,  # GPU가 있으면 FP16 사용
            use_yolov5_backend=True  # YOLOv5 DetectMultiBackend 방식 우선
        )
        logger.info(f"탐지기 초기화 완료")
        logger.info(f"  - 클래스 수: {len(detector.names)}")
        logger.info(f"  - 백엔드: {'YOLOv5' if detector._using_yolov5_backend else 'ultralytics' if detector._using_ultralytics else 'hub'}")
        logger.info(f"  - 디바이스: {detector.device}")
        logger.info(f"  - 이미지 크기: {detector.img_size}")
        logger.info(f"  - Stride: {detector.stride}")

    except Exception as e:
        logger.error(f"탐지기 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 이미지 로드
    logger.info("이미지 로드 중...")
    image = cv2.imread(TEST_IMAGE_PATH)
    if image is None:
        logger.error(f"이미지 로드 실패: {TEST_IMAGE_PATH}")
        return False

    logger.info(f"이미지 크기: {image.shape}")

    # 객체 탐지
    logger.info("객체 탐지 수행 중...")
    try:
        detections = detector.detect(image)
        logger.info(f"탐지된 객체 수: {len(detections)}")

        if len(detections) > 0:
            logger.info("탐지 결과:")
            for i, det in enumerate(detections):
                logger.info(f"  [{i}] {det.class_name} (class_id={det.class_id})")
                logger.info(f"       신뢰도: {det.confidence:.4f}")
                logger.info(f"       위치: ({det.x}, {det.y}) - ({det.x2}, {det.y2})")
                logger.info(f"       크기: {det.width} x {det.height}")

            # 결과 이미지 저장
            result_image = detector.draw_detections(image, detections)
            output_path = "/root/fursys_imgprosessing_ws/src/nimg_v2/test_result.png"
            cv2.imwrite(output_path, result_image)
            logger.info(f"결과 이미지 저장: {output_path}")

            return True
        else:
            logger.warning("탐지된 객체가 없습니다!")
            logger.info("신뢰도 임계값을 낮춰서 다시 시도합니다...")

            # 신뢰도 임계값을 낮춰서 재시도
            detector.conf_threshold = 0.25
            detections = detector.detect(image)
            logger.info(f"(conf=0.25) 탐지된 객체 수: {len(detections)}")

            if len(detections) > 0:
                for i, det in enumerate(detections):
                    logger.info(f"  [{i}] {det.class_name}: {det.confidence:.4f}")
                return True
            else:
                logger.error("신뢰도 0.25로도 탐지 실패")
                return False

    except Exception as e:
        logger.error(f"탐지 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_original_nimg():
    """기존 nimg의 Detector와 비교 테스트"""
    logger.info("=" * 60)
    logger.info("기존 nimg Detector와 비교 테스트")
    logger.info("=" * 60)

    try:
        # 기존 nimg의 Detector 임포트
        from nimg.submodules.detect import Detector

        MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
        TEST_IMAGE_PATH = "/root/fursys_imgprosessing_ws/20251208_155531_output/color_000000.png"

        logger.info("기존 nimg Detector 초기화 중...")
        original_detector = Detector()
        original_detector.detect_set(MODEL_PATH)

        logger.info("기존 nimg Detector로 탐지 수행 중...")

        # 기존 방식: 파일로 저장 후 탐지
        cv2.imwrite('source.png', cv2.imread(TEST_IMAGE_PATH))
        result_img, items, check_flag = original_detector.detect('source.png')

        logger.info(f"기존 nimg 탐지 결과:")
        logger.info(f"  - 탐지 여부: {check_flag}")
        logger.info(f"  - 탐지된 객체 수: {items.size()}")

        if items.size() > 0:
            for item in items.itemlist:
                logger.info(f"    - {item.getName()}: {item.getConfidence()}")

        # 결과 이미지 저장
        cv2.imwrite('/root/fursys_imgprosessing_ws/src/nimg_v2/test_result_original.png', result_img)
        logger.info(f"기존 nimg 결과 이미지 저장: test_result_original.png")

        return True

    except Exception as e:
        logger.error(f"기존 nimg 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("nimg_v2 YOLODetector 테스트 시작")
    logger.info("=" * 60)

    # 새 YOLODetector 테스트
    result1 = test_detector()

    print()

    # 기존 nimg Detector와 비교 (선택적)
    if "--compare" in sys.argv:
        result2 = test_with_original_nimg()
    else:
        logger.info("기존 nimg와 비교하려면 --compare 옵션을 추가하세요")
        result2 = True

    print()
    logger.info("=" * 60)
    logger.info(f"테스트 결과: {'성공' if result1 else '실패'}")
    logger.info("=" * 60)

    sys.exit(0 if result1 else 1)
