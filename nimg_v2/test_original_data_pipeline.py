#!/usr/bin/env python3
"""
nimg_v2 테스트 - 원본 데이터 (20240525/20240529) 기반 테스트
AI 모델이 학습된 가구 제품 이미지로 전체 파이프라인 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import logging
import time
from glob import glob
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_detection_original_data():
    """원본 데이터 탐지 테스트"""
    from nimg_v2.detection.yolo_detector import YOLODetector

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    DATA_DIRS = [
        "/root/fursys_imgprosessing_ws/src/nimg/img_data/20240525",
        "/root/fursys_imgprosessing_ws/src/nimg/img_data/20240529",
    ]

    logger.info("=" * 70)
    logger.info("Test 1: 원본 데이터 객체 탐지 테스트")
    logger.info("=" * 70)

    detector = YOLODetector(
        model_path=MODEL_PATH,
        conf_threshold=0.25,
        iou_threshold=0.45,
        half=False
    )
    detector.warmup()

    all_results = []

    for data_dir in DATA_DIRS:
        images = sorted(glob(f"{data_dir}/*.png"))
        dir_name = Path(data_dir).name

        logger.info(f"\n--- {dir_name} ({len(images)} images) ---")

        detected = 0
        total_detections = 0

        for img_path in images:
            image = cv2.imread(img_path)
            if image is None:
                continue

            detections = detector.detect(image)

            if len(detections) > 0:
                detected += 1
                total_detections += len(detections)

                # 결과 저장
                for det in detections:
                    all_results.append({
                        'folder': dir_name,
                        'image': Path(img_path).name,
                        'class_name': det.class_name,
                        'confidence': det.confidence,
                        'bbox': det.bbox,
                        'area': det.area
                    })

                # 상위 5개 이미지만 상세 출력
                if detected <= 5:
                    logger.info(f"  {Path(img_path).name}: {len(detections)} 객체")
                    for d in detections[:3]:
                        logger.info(f"    - {d.class_name}: conf={d.confidence:.3f}")

        logger.info(f"\n  {dir_name} 요약: {detected}/{len(images)} 이미지에서 {total_detections} 객체 탐지")

    return all_results


def test_3d_estimation_original_data():
    """원본 데이터 3D 추정 테스트 (Depth 없음 - 시뮬레이션)"""
    from nimg_v2.detection.yolo_detector import YOLODetector
    from nimg_v2.estimation.position_estimator import PositionEstimator
    from nimg_v2.estimation.orientation_estimator import OrientationEstimator

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    DATA_DIR = "/root/fursys_imgprosessing_ws/src/nimg/img_data/20240525"

    logger.info("\n" + "=" * 70)
    logger.info("Test 2: 3D 위치/방향 추정 테스트 (시뮬레이션 Depth)")
    logger.info("=" * 70)

    # D455 내부 파라미터
    intrinsics = {
        'fx': 636.3392652788157,
        'fy': 636.4266464742717,
        'cx': 654.3418233071645,
        'cy': 399.58963414918554
    }

    detector = YOLODetector(MODEL_PATH, conf_threshold=0.25, half=False)
    detector.warmup()

    position_estimator = PositionEstimator(intrinsics)
    orientation_estimator = OrientationEstimator(min_points=50)

    images = sorted(glob(f"{DATA_DIR}/*.png"))[:10]

    results = []

    for img_path in images:
        image = cv2.imread(img_path)
        if image is None:
            continue

        detections = detector.detect(image)
        if len(detections) == 0:
            continue

        # 시뮬레이션 Depth 생성 (1.5m ~ 3.0m 범위)
        h, w = image.shape[:2]
        simulated_depth = np.random.uniform(1.5, 3.0, (h, w)).astype(np.float32)

        for det in detections[:3]:
            # 3D 위치 추정
            pos_result = position_estimator.estimate_position(det.bbox, simulated_depth)

            # 방향 추정
            orientation = orientation_estimator.estimate_from_depth(
                simulated_depth, det.bbox, intrinsics
            )

            if pos_result is not None:
                pos = pos_result.position
                distance = np.linalg.norm(pos)

                result = {
                    'image': Path(img_path).name,
                    'class_name': det.class_name,
                    'confidence': det.confidence,
                    'position': pos.tolist(),
                    'distance': distance,
                    'orientation': {
                        'roll': orientation.roll if orientation else None,
                        'pitch': orientation.pitch if orientation else None,
                        'yaw': orientation.yaw if orientation else None
                    } if orientation else None
                }
                results.append(result)

                logger.info(f"\n{Path(img_path).name} - {det.class_name}:")
                logger.info(f"  위치: X={pos[0]:.3f}m, Y={pos[1]:.3f}m, Z={pos[2]:.3f}m")
                logger.info(f"  거리: {distance:.3f}m")
                if orientation:
                    logger.info(f"  각도: R={orientation.roll:.2f}°, P={orientation.pitch:.2f}°, Y={orientation.yaw:.2f}°")

    return results


def test_kalman_filter():
    """Kalman Filter 추적 테스트"""
    from nimg_v2.tracking.kalman_filter_3d import KalmanFilter3D

    logger.info("\n" + "=" * 70)
    logger.info("Test 3: Kalman Filter 3D 추적 테스트")
    logger.info("=" * 70)

    kf = KalmanFilter3D(dt=1/30.0)

    # 시뮬레이션 궤적 생성 (직선 운동)
    np.random.seed(42)
    true_positions = []
    for i in range(30):
        # 초기 위치 (1, 0.5, 2)에서 X방향으로 이동
        x = 1.0 + i * 0.01  # 0.3m/s
        y = 0.5 + np.random.normal(0, 0.005)  # 노이즈
        z = 2.0 + np.random.normal(0, 0.005)
        true_positions.append(np.array([x, y, z]))

    # 초기화
    kf.initialize(true_positions[0])

    estimates = []
    velocities = []

    for i, pos in enumerate(true_positions[1:], 1):
        # 노이즈 추가 (측정 오차)
        measured = pos + np.random.normal(0, 0.01, 3)

        est_pos, velocity, accel = kf.predict_and_update(measured)
        estimates.append(est_pos)
        velocities.append(velocity)

    # 결과 분석
    logger.info(f"\n궤적 길이: {len(true_positions)} 프레임")

    # 최종 속도 추정값
    final_vel = velocities[-1]
    logger.info(f"\n최종 속도 추정:")
    logger.info(f"  Vx = {final_vel[0]:.4f} m/s (실제: ~0.3 m/s)")
    logger.info(f"  Vy = {final_vel[1]:.4f} m/s (실제: ~0)")
    logger.info(f"  Vz = {final_vel[2]:.4f} m/s (실제: ~0)")
    logger.info(f"  속력 = {np.linalg.norm(final_vel):.4f} m/s")

    # 위치 추정 오차
    pos_errors = [np.linalg.norm(est - true) for est, true in zip(estimates, true_positions[1:])]
    logger.info(f"\n위치 추정 오차:")
    logger.info(f"  평균: {np.mean(pos_errors)*1000:.2f} mm")
    logger.info(f"  최대: {np.max(pos_errors)*1000:.2f} mm")

    return estimates, velocities


def test_change_calculator():
    """변화량 계산기 테스트"""
    from nimg_v2.analysis.change_calculator import ChangeCalculator
    from nimg_v2.estimation.orientation_estimator import OrientationResult

    logger.info("\n" + "=" * 70)
    logger.info("Test 4: 변화량 계산기 테스트")
    logger.info("=" * 70)

    calculator = ChangeCalculator()

    # OrientationResult 헬퍼 함수
    def make_orientation(roll, pitch, yaw, confidence=0.9):
        return OrientationResult(
            roll=roll, pitch=pitch, yaw=yaw,
            center=np.array([0.0, 0.0, 2.0]),
            axes=np.eye(3),
            eigenvalues=np.array([1.0, 0.5, 0.1]),
            confidence=confidence,
            num_points=1000
        )

    # 기준 상태 설정
    ref_position = np.array([1.0, 0.5, 2.0])
    ref_orientation = make_orientation(0.0, 0.0, 0.0)
    calculator.set_reference(ref_position, ref_orientation, frame_idx=0)

    logger.info(f"기준 위치: {ref_position}")
    logger.info(f"기준 각도: R=0°, P=0°, Y=0°")

    # 변화된 상태들 테스트
    test_cases = [
        # (position, velocity, orientation, description)
        (
            np.array([1.1, 0.5, 2.0]),
            np.array([0.3, 0.0, 0.0]),
            make_orientation(5.0, 0.0, 0.0),
            "X방향 0.1m 이동, Roll 5도"
        ),
        (
            np.array([1.0, 0.6, 2.0]),
            np.array([0.0, 0.3, 0.0]),
            make_orientation(0.0, 10.0, 0.0),
            "Y방향 0.1m 이동, Pitch 10도"
        ),
        (
            np.array([1.0, 0.5, 2.2]),
            np.array([0.0, 0.0, 0.6]),
            make_orientation(0.0, 0.0, 15.0),
            "Z방향 0.2m 이동, Yaw 15도"
        ),
    ]

    results = []
    for i, (pos, vel, orient, desc) in enumerate(test_cases, 1):
        result = calculator.calculate_change(
            current_position=pos,
            current_velocity=vel,
            current_orientation=orient,
            frame_idx=i,
            timestamp=i/30.0,
            position_confidence=0.9
        )

        if result:
            results.append(result)
            logger.info(f"\n테스트 {i}: {desc}")
            dx, dy, dz = result.position_change
            logger.info(f"  위치 변화: dx={dx:.3f}m, dy={dy:.3f}m, dz={dz:.3f}m")
            logger.info(f"  거리 변화: {result.distance_from_reference:.3f}m")
            logger.info(f"  속력: {result.speed:.3f}m/s")
            logger.info(f"  각도 변화: R={result.roll_change:.2f}°, P={result.pitch_change:.2f}°, Y={result.yaw_change:.2f}°")

    return results


def test_visualizer():
    """시각화 결과 저장 테스트"""
    from nimg_v2.detection.yolo_detector import YOLODetector

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    DATA_DIR = "/root/fursys_imgprosessing_ws/src/nimg/img_data/20240525"
    OUTPUT_PATH = "/root/fursys_imgprosessing_ws/src/nimg_v2/test_original_result.png"

    logger.info("\n" + "=" * 70)
    logger.info("Test 5: 탐지 결과 시각화")
    logger.info("=" * 70)

    detector = YOLODetector(MODEL_PATH, conf_threshold=0.25, half=False)
    detector.warmup()

    images = sorted(glob(f"{DATA_DIR}/*.png"))

    # 탐지되는 이미지 찾기
    best_image = None
    best_detections = []

    for img_path in images:
        image = cv2.imread(img_path)
        if image is None:
            continue

        detections = detector.detect(image)
        if len(detections) > len(best_detections):
            best_image = image.copy()
            best_detections = detections
            best_path = img_path

    if best_image is not None and len(best_detections) > 0:
        # 결과 그리기
        result_img = detector.draw_detections(best_image, best_detections)

        # 정보 추가
        info = f"Image: {Path(best_path).name} | Detections: {len(best_detections)} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(result_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imwrite(OUTPUT_PATH, result_img)
        logger.info(f"\n결과 이미지 저장: {OUTPUT_PATH}")
        logger.info(f"원본 이미지: {best_path}")
        logger.info(f"탐지된 객체: {len(best_detections)}개")
        for d in best_detections[:5]:
            logger.info(f"  - {d.class_name}: {d.confidence:.3f}")

    return OUTPUT_PATH


def main():
    """메인 테스트 실행"""
    logger.info("=" * 70)
    logger.info("nimg_v2 종합 테스트 - 원본 데이터 기반")
    logger.info(f"테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    try:
        # Test 1: 원본 데이터 탐지
        detection_results = test_detection_original_data()
        logger.info(f"\n탐지 결과 총계: {len(detection_results)}개 객체")

        # Test 2: 3D 추정
        estimation_results = test_3d_estimation_original_data()

        # Test 3: Kalman Filter
        test_kalman_filter()

        # Test 4: 변화량 계산
        test_change_calculator()

        # Test 5: 시각화
        test_visualizer()

        logger.info("\n" + "=" * 70)
        logger.info("모든 테스트 완료!")
        logger.info("=" * 70)

        # 최종 요약
        logger.info("\n" + "=" * 70)
        logger.info("테스트 결과 요약")
        logger.info("=" * 70)

        unique_classes = set(r['class_name'] for r in detection_results)
        logger.info(f"탐지된 고유 클래스: {len(unique_classes)}개")
        logger.info(f"총 탐지 객체: {len(detection_results)}개")

        # 클래스별 통계
        from collections import Counter
        class_counts = Counter(r['class_name'] for r in detection_results)
        logger.info("\n클래스별 탐지 수:")
        for cls, cnt in class_counts.most_common(10):
            logger.info(f"  {cls}: {cnt}개")

    except Exception as e:
        logger.error(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
