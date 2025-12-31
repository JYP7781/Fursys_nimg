#!/usr/bin/env python3
"""
nimg_v2 테스트 - color_001000.png 기준 프레임 테스트
20251208_161246_output 폴더 데이터 사용
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_single_image_detection():
    """color_001000.png 단일 이미지 탐지 테스트"""
    from nimg_v2.detection.yolo_detector import YOLODetector

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    IMAGE_PATH = "/root/fursys_imgprosessing_ws/20251208_161246_output/color_001000.png"
    DEPTH_PATH = "/root/fursys_imgprosessing_ws/20251208_161246_output/depth_001000.png"

    logger.info("=" * 70)
    logger.info("Test 1: 단일 이미지 탐지 테스트 (color_001000.png)")
    logger.info("=" * 70)

    # 이미지 로드
    image = cv2.imread(IMAGE_PATH)
    depth_raw = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)

    if image is None:
        logger.error(f"이미지 로드 실패: {IMAGE_PATH}")
        return None

    logger.info(f"이미지 크기: {image.shape}")
    logger.info(f"Depth 크기: {depth_raw.shape if depth_raw is not None else 'N/A'}")

    # 다양한 conf 임계값으로 테스트
    results = {}
    for conf in [0.5, 0.3, 0.25, 0.1]:
        detector = YOLODetector(
            model_path=MODEL_PATH,
            conf_threshold=conf,
            iou_threshold=0.45,
            img_size=640,
            half=False
        )
        detector.warmup()

        start_time = time.time()
        detections = detector.detect(image)
        elapsed = time.time() - start_time

        results[conf] = {
            'count': len(detections),
            'time_ms': elapsed * 1000,
            'detections': detections
        }

        logger.info(f"\nconf={conf}: {len(detections)} 객체 탐지 ({elapsed*1000:.1f}ms)")

        for det in detections[:5]:  # 상위 5개만 출력
            logger.info(f"  - {det.class_name}: conf={det.confidence:.3f}, "
                       f"bbox=({det.x}, {det.y}, {det.width}, {det.height})")

    return results, image, depth_raw


def test_depth_analysis(depth_raw, detections):
    """Depth 데이터 분석 테스트"""
    logger.info("\n" + "=" * 70)
    logger.info("Test 2: Depth 데이터 분석")
    logger.info("=" * 70)

    if depth_raw is None:
        logger.warning("Depth 이미지 없음")
        return

    # Depth를 미터로 변환
    depth_m = depth_raw.astype(np.float32) * 0.001

    # 전체 Depth 통계
    valid_mask = (depth_m > 0.1) & (depth_m < 10.0)
    valid_depth = depth_m[valid_mask]

    logger.info(f"유효 Depth 픽셀: {valid_mask.sum()} / {depth_m.size} ({100*valid_mask.sum()/depth_m.size:.1f}%)")
    logger.info(f"Depth 범위: {valid_depth.min():.3f}m ~ {valid_depth.max():.3f}m")
    logger.info(f"Depth 평균: {valid_depth.mean():.3f}m (std: {valid_depth.std():.3f}m)")

    # 각 탐지 객체의 Depth 분석
    if detections:
        logger.info("\n탐지 객체별 Depth:")
        for det in detections[:5]:
            roi_depth = depth_m[det.y:det.y2, det.x:det.x2]
            roi_valid = roi_depth[(roi_depth > 0.1) & (roi_depth < 10.0)]

            if len(roi_valid) > 0:
                logger.info(f"  - {det.class_name}: 거리={roi_valid.mean():.3f}m "
                           f"(min={roi_valid.min():.3f}, max={roi_valid.max():.3f})")


def test_position_estimation():
    """3D 위치 추정 테스트"""
    from nimg_v2.estimation.position_estimator import PositionEstimator
    from nimg_v2.detection.yolo_detector import YOLODetector

    logger.info("\n" + "=" * 70)
    logger.info("Test 3: 3D 위치 추정 테스트")
    logger.info("=" * 70)

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    IMAGE_PATH = "/root/fursys_imgprosessing_ws/20251208_161246_output/color_001000.png"
    DEPTH_PATH = "/root/fursys_imgprosessing_ws/20251208_161246_output/depth_001000.png"

    # D455 카메라 내부 파라미터
    intrinsics = {
        'fx': 636.3392652788157,
        'fy': 636.4266464742717,
        'cx': 654.3418233071645,
        'cy': 399.58963414918554
    }

    # 모듈 로드
    detector = YOLODetector(MODEL_PATH, conf_threshold=0.25, half=False)
    detector.warmup()

    position_estimator = PositionEstimator(intrinsics)

    # 이미지 로드
    image = cv2.imread(IMAGE_PATH)
    depth_raw = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)
    depth_m = depth_raw.astype(np.float32) * 0.001

    # 탐지
    detections = detector.detect(image)

    logger.info(f"탐지 객체 수: {len(detections)}")

    # 각 객체의 3D 위치 추정
    for det in detections[:5]:
        pos_result = position_estimator.estimate_position(det.bbox, depth_m)

        if pos_result is not None:
            pos = pos_result.position
            logger.info(f"\n{det.class_name} (conf={det.confidence:.2f}):")
            logger.info(f"  2D bbox: ({det.x}, {det.y}, {det.width}, {det.height})")
            logger.info(f"  3D 위치: X={pos[0]:.3f}m, Y={pos[1]:.3f}m, Z={pos[2]:.3f}m")
            logger.info(f"  거리: {np.linalg.norm(pos):.3f}m")
            logger.info(f"  신뢰도: {pos_result.confidence:.2f}")
        else:
            logger.warning(f"  {det.class_name}: 위치 추정 실패")


def test_orientation_estimation():
    """방향(각도) 추정 테스트"""
    from nimg_v2.estimation.orientation_estimator import OrientationEstimator
    from nimg_v2.detection.yolo_detector import YOLODetector

    logger.info("\n" + "=" * 70)
    logger.info("Test 4: 방향(각도) 추정 테스트")
    logger.info("=" * 70)

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    IMAGE_PATH = "/root/fursys_imgprosessing_ws/20251208_161246_output/color_001000.png"
    DEPTH_PATH = "/root/fursys_imgprosessing_ws/20251208_161246_output/depth_001000.png"

    intrinsics = {
        'fx': 636.3392652788157,
        'fy': 636.4266464742717,
        'cx': 654.3418233071645,
        'cy': 399.58963414918554
    }

    # 모듈 로드
    detector = YOLODetector(MODEL_PATH, conf_threshold=0.25, half=False)
    detector.warmup()

    orientation_estimator = OrientationEstimator(min_points=100)

    # 이미지 로드
    image = cv2.imread(IMAGE_PATH)
    depth_raw = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)
    depth_m = depth_raw.astype(np.float32) * 0.001

    # 탐지
    detections = detector.detect(image)

    # 각 객체의 방향 추정
    for det in detections[:5]:
        orientation = orientation_estimator.estimate_from_depth(
            depth_m, det.bbox, intrinsics
        )

        if orientation is not None:
            logger.info(f"\n{det.class_name} (conf={det.confidence:.2f}):")
            logger.info(f"  Roll:  {orientation.roll:.2f}° (신뢰도: {orientation.roll_confidence:.2f})")
            logger.info(f"  Pitch: {orientation.pitch:.2f}° (신뢰도: {orientation.pitch_confidence:.2f})")
            logger.info(f"  Yaw:   {orientation.yaw:.2f}° (신뢰도: {orientation.yaw_confidence:.2f})")
        else:
            logger.warning(f"  {det.class_name}: 방향 추정 실패")


def test_frame_sequence():
    """프레임 시퀀스 처리 테스트 (color_001000.png 기준)"""
    from nimg_v2.data.data_loader import DataLoader
    from nimg_v2.detection.yolo_detector import YOLODetector
    from nimg_v2.estimation.position_estimator import PositionEstimator
    from nimg_v2.estimation.orientation_estimator import OrientationEstimator
    from nimg_v2.tracking.kalman_filter_3d import KalmanFilter3D
    from nimg_v2.analysis.change_calculator import ChangeCalculator

    logger.info("\n" + "=" * 70)
    logger.info("Test 5: 프레임 시퀀스 처리 테스트")
    logger.info("=" * 70)

    DATA_DIR = "/root/fursys_imgprosessing_ws/20251208_161246_output"
    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    REFERENCE_FRAME = 1000  # color_001000.png

    intrinsics = {
        'fx': 636.3392652788157,
        'fy': 636.4266464742717,
        'cx': 654.3418233071645,
        'cy': 399.58963414918554
    }

    # 컴포넌트 초기화
    loader = DataLoader(DATA_DIR)
    detector = YOLODetector(MODEL_PATH, conf_threshold=0.25, half=False)
    detector.warmup()
    position_estimator = PositionEstimator(intrinsics)
    orientation_estimator = OrientationEstimator(min_points=100)
    kalman_filter = KalmanFilter3D(dt=1/30.0)
    change_calculator = ChangeCalculator()

    logger.info(f"데이터셋: {loader.num_frames} 프레임")
    logger.info(f"기준 프레임: {REFERENCE_FRAME}")

    # 기준 프레임 처리
    logger.info("\n--- 기준 프레임 처리 ---")
    ref_frame = loader.load_frame(REFERENCE_FRAME)
    ref_detections = detector.detect(ref_frame.rgb)

    if len(ref_detections) == 0:
        logger.error("기준 프레임에서 객체를 탐지하지 못했습니다!")
        return

    best_det = detector.get_best_detection(ref_detections, by='confidence')
    logger.info(f"기준 객체: {best_det.class_name} (conf={best_det.confidence:.2f})")

    # 기준 위치/방향 추정
    ref_pos = position_estimator.estimate_position(best_det.bbox, ref_frame.depth)
    ref_orient = orientation_estimator.estimate_from_depth(
        ref_frame.depth, best_det.bbox, intrinsics
    )

    if ref_pos is None or ref_orient is None:
        logger.error("기준 프레임 위치/방향 추정 실패")
        return

    logger.info(f"기준 위치: ({ref_pos.position[0]:.3f}, {ref_pos.position[1]:.3f}, {ref_pos.position[2]:.3f})m")
    logger.info(f"기준 각도: R={ref_orient.roll:.2f}°, P={ref_orient.pitch:.2f}°, Y={ref_orient.yaw:.2f}°")

    # 기준 설정
    change_calculator.set_reference(ref_pos.position, ref_orient, REFERENCE_FRAME)
    kalman_filter.initialize(ref_pos.position)

    # 후속 프레임 처리 (기준 프레임 전후 50프레임)
    test_frames = list(range(max(0, REFERENCE_FRAME - 50), min(loader.num_frames, REFERENCE_FRAME + 51)))
    test_frames = [f for f in test_frames if f != REFERENCE_FRAME]

    logger.info(f"\n--- 후속 프레임 처리 ({len(test_frames)} 프레임) ---")

    results = []
    for frame_idx in test_frames:
        try:
            frame = loader.load_frame(frame_idx)
            detections = detector.detect(frame.rgb)

            if len(detections) == 0:
                continue

            best = detector.get_best_detection(detections, by='confidence')
            pos = position_estimator.estimate_position(best.bbox, frame.depth)
            orient = orientation_estimator.estimate_from_depth(
                frame.depth, best.bbox, intrinsics
            )

            if pos is None or orient is None:
                continue

            # Kalman Filter로 속도 추정
            est_pos, velocity, accel = kalman_filter.predict_and_update(pos.position)

            # 변화량 계산
            result = change_calculator.calculate_change(
                current_position=est_pos,
                current_velocity=velocity,
                current_orientation=orient,
                frame_idx=frame_idx,
                timestamp=frame.timestamp,
                position_confidence=pos.confidence
            )

            if result is not None:
                results.append(result)

        except Exception as e:
            logger.debug(f"Frame {frame_idx} 처리 오류: {e}")
            continue

    logger.info(f"\n처리 완료: {len(results)} 결과")

    # 결과 요약
    if results:
        logger.info("\n--- 변화량 측정 결과 요약 ---")

        # 위치 변화
        dx = [r.dx for r in results]
        dy = [r.dy for r in results]
        dz = [r.dz for r in results]

        logger.info(f"위치 변화 (X): min={min(dx):.3f}m, max={max(dx):.3f}m, mean={np.mean(dx):.3f}m")
        logger.info(f"위치 변화 (Y): min={min(dy):.3f}m, max={max(dy):.3f}m, mean={np.mean(dy):.3f}m")
        logger.info(f"위치 변화 (Z): min={min(dz):.3f}m, max={max(dz):.3f}m, mean={np.mean(dz):.3f}m")

        # 속도
        speeds = [r.speed for r in results]
        logger.info(f"속도: min={min(speeds):.3f}m/s, max={max(speeds):.3f}m/s, mean={np.mean(speeds):.3f}m/s")

        # 각도 변화
        roll_changes = [r.roll_change for r in results]
        pitch_changes = [r.pitch_change for r in results]
        yaw_changes = [r.yaw_change for r in results]

        logger.info(f"Roll 변화: min={min(roll_changes):.2f}°, max={max(roll_changes):.2f}°")
        logger.info(f"Pitch 변화: min={min(pitch_changes):.2f}°, max={max(pitch_changes):.2f}°")
        logger.info(f"Yaw 변화: min={min(yaw_changes):.2f}°, max={max(yaw_changes):.2f}°")

    return results


def save_result_image():
    """결과 이미지 저장"""
    from nimg_v2.detection.yolo_detector import YOLODetector
    from nimg_v2.estimation.position_estimator import PositionEstimator

    logger.info("\n" + "=" * 70)
    logger.info("Test 6: 결과 이미지 저장")
    logger.info("=" * 70)

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    IMAGE_PATH = "/root/fursys_imgprosessing_ws/20251208_161246_output/color_001000.png"
    DEPTH_PATH = "/root/fursys_imgprosessing_ws/20251208_161246_output/depth_001000.png"
    OUTPUT_PATH = "/root/fursys_imgprosessing_ws/src/nimg_v2/test_color_001000_result.png"

    intrinsics = {
        'fx': 636.3392652788157,
        'fy': 636.4266464742717,
        'cx': 654.3418233071645,
        'cy': 399.58963414918554
    }

    # 로드
    detector = YOLODetector(MODEL_PATH, conf_threshold=0.25, half=False)
    detector.warmup()
    position_estimator = PositionEstimator(intrinsics)

    image = cv2.imread(IMAGE_PATH)
    depth_raw = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)
    depth_m = depth_raw.astype(np.float32) * 0.001

    # 탐지
    detections = detector.detect(image)

    # 결과 그리기
    result_img = image.copy()

    for det in detections:
        # 바운딩 박스
        color = (0, 255, 0)  # 초록색
        cv2.rectangle(result_img, (det.x, det.y), (det.x2, det.y2), color, 2)

        # 3D 위치 추정
        pos_result = position_estimator.estimate_position(det.bbox, depth_m)

        if pos_result is not None:
            pos = pos_result.position
            distance = np.linalg.norm(pos)
            label = f"{det.class_name}: {det.confidence:.2f} ({distance:.2f}m)"
        else:
            label = f"{det.class_name}: {det.confidence:.2f}"

        # 라벨
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result_img,
                     (det.x, det.y - label_size[1] - 10),
                     (det.x + label_size[0] + 4, det.y),
                     color, -1)
        cv2.putText(result_img, label, (det.x + 2, det.y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 정보 표시
    info_text = f"Frame: color_001000.png | Detections: {len(detections)} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    cv2.putText(result_img, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 저장
    cv2.imwrite(OUTPUT_PATH, result_img)
    logger.info(f"결과 이미지 저장: {OUTPUT_PATH}")
    logger.info(f"탐지 객체 수: {len(detections)}")

    return OUTPUT_PATH


def main():
    """메인 테스트 실행"""
    logger.info("=" * 70)
    logger.info("nimg_v2 종합 테스트 시작")
    logger.info(f"테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("기준 이미지: color_001000.png")
    logger.info("=" * 70)

    try:
        # Test 1: 단일 이미지 탐지
        result = test_single_image_detection()
        if result:
            results, image, depth_raw = result

            # Test 2: Depth 분석
            best_conf = 0.25
            if best_conf in results and results[best_conf]['detections']:
                test_depth_analysis(depth_raw, results[best_conf]['detections'])

        # Test 3: 위치 추정
        test_position_estimation()

        # Test 4: 방향 추정
        test_orientation_estimation()

        # Test 5: 프레임 시퀀스
        test_frame_sequence()

        # Test 6: 결과 이미지 저장
        save_result_image()

        logger.info("\n" + "=" * 70)
        logger.info("모든 테스트 완료")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
