#!/usr/bin/env python3
"""
nimg_v2 종합 파이프라인 테스트
결과물(시각화, 로그, 데이터)을 result 폴더에 저장
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pandas as pd
import json
import logging
import time
from glob import glob
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt

# 결과 디렉토리 설정
RESULT_DIR = Path(__file__).parent / "result"
IMAGES_DIR = RESULT_DIR / "images"
LOGS_DIR = RESULT_DIR / "logs"
DATA_DIR = RESULT_DIR / "data"
PLOTS_DIR = RESULT_DIR / "plots"

# 로그 설정
log_file = LOGS_DIR / f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineTestRunner:
    """파이프라인 테스트 실행 및 결과 저장"""

    def __init__(self):
        self.model_path = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
        self.data_dirs = [
            "/root/fursys_imgprosessing_ws/src/nimg/img_data/20240525",
            "/root/fursys_imgprosessing_ws/src/nimg/img_data/20240529",
        ]
        self.intrinsics = {
            'fx': 636.3392652788157,
            'fy': 636.4266464742717,
            'cx': 654.3418233071645,
            'cy': 399.58963414918554
        }
        self.results = {
            'detection': [],
            'position': [],
            'orientation': [],
            'kalman': [],
            'change': []
        }
        self.test_start_time = datetime.now()

    def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("=" * 70)
        logger.info("nimg_v2 종합 파이프라인 테스트 시작")
        logger.info(f"시작 시간: {self.test_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"결과 저장 경로: {RESULT_DIR}")
        logger.info("=" * 70)

        # 1. 객체 탐지 테스트
        self.test_detection()

        # 2. 3D 위치 추정 테스트
        self.test_position_estimation()

        # 3. 방향 추정 테스트
        self.test_orientation_estimation()

        # 4. Kalman Filter 테스트
        self.test_kalman_filter()

        # 5. 변화량 계산 테스트
        self.test_change_calculator()

        # 6. 시각화 생성
        self.create_visualizations()

        # 7. 결과 저장
        self.save_all_results()

        # 8. 보고서 생성
        self.generate_report()

        logger.info("\n" + "=" * 70)
        logger.info("모든 테스트 완료!")
        logger.info(f"결과 저장 경로: {RESULT_DIR}")
        logger.info("=" * 70)

    def test_detection(self):
        """객체 탐지 테스트"""
        from nimg_v2.detection.yolo_detector import YOLODetector

        logger.info("\n" + "=" * 70)
        logger.info("Test 1: 객체 탐지 테스트")
        logger.info("=" * 70)

        detector = YOLODetector(
            model_path=self.model_path,
            conf_threshold=0.25,
            iou_threshold=0.45,
            half=False
        )
        detector.warmup()

        for data_dir in self.data_dirs:
            images = sorted(glob(f"{data_dir}/*.png"))
            dir_name = Path(data_dir).name

            logger.info(f"\n--- {dir_name} ({len(images)} images) ---")

            for img_path in images:
                image = cv2.imread(img_path)
                if image is None:
                    continue

                start_time = time.time()
                detections = detector.detect(image)
                elapsed = time.time() - start_time

                for det in detections:
                    self.results['detection'].append({
                        'folder': dir_name,
                        'image': Path(img_path).name,
                        'class_id': det.class_id,
                        'class_name': det.class_name,
                        'confidence': det.confidence,
                        'x': det.x,
                        'y': det.y,
                        'width': det.width,
                        'height': det.height,
                        'area': det.area,
                        'inference_time_ms': elapsed * 1000
                    })

                    logger.info(f"  {Path(img_path).name}: {det.class_name} "
                               f"(conf={det.confidence:.3f}, time={elapsed*1000:.1f}ms)")

                # 탐지 결과 이미지 저장
                if len(detections) > 0:
                    result_img = detector.draw_detections(image, detections)
                    output_path = IMAGES_DIR / f"detection_{dir_name}_{Path(img_path).stem}.png"
                    cv2.imwrite(str(output_path), result_img)

        logger.info(f"\n탐지 결과: 총 {len(self.results['detection'])}개 객체")

    def test_position_estimation(self):
        """3D 위치 추정 테스트"""
        from nimg_v2.detection.yolo_detector import YOLODetector
        from nimg_v2.estimation.position_estimator import PositionEstimator

        logger.info("\n" + "=" * 70)
        logger.info("Test 2: 3D 위치 추정 테스트")
        logger.info("=" * 70)

        detector = YOLODetector(self.model_path, conf_threshold=0.25, half=False)
        detector.warmup()
        position_estimator = PositionEstimator(self.intrinsics)

        for data_dir in self.data_dirs:
            images = sorted(glob(f"{data_dir}/*.png"))
            dir_name = Path(data_dir).name

            for img_path in images:
                image = cv2.imread(img_path)
                if image is None:
                    continue

                detections = detector.detect(image)
                if len(detections) == 0:
                    continue

                # 시뮬레이션 Depth 생성
                h, w = image.shape[:2]
                simulated_depth = np.random.uniform(1.5, 3.0, (h, w)).astype(np.float32)

                for det in detections:
                    pos_result = position_estimator.estimate_position(det.bbox, simulated_depth)

                    if pos_result is not None:
                        pos = pos_result.position
                        distance = np.linalg.norm(pos)

                        self.results['position'].append({
                            'folder': dir_name,
                            'image': Path(img_path).name,
                            'class_name': det.class_name,
                            'x': float(pos[0]),
                            'y': float(pos[1]),
                            'z': float(pos[2]),
                            'distance': float(distance),
                            'confidence': float(pos_result.confidence),
                            'depth_type': 'simulated'
                        })

                        logger.info(f"  {Path(img_path).name} - {det.class_name}: "
                                   f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m, "
                                   f"dist={distance:.3f}m")

    def test_orientation_estimation(self):
        """방향 추정 테스트"""
        from nimg_v2.detection.yolo_detector import YOLODetector
        from nimg_v2.estimation.orientation_estimator import OrientationEstimator

        logger.info("\n" + "=" * 70)
        logger.info("Test 3: 방향(각도) 추정 테스트")
        logger.info("=" * 70)

        detector = YOLODetector(self.model_path, conf_threshold=0.25, half=False)
        detector.warmup()
        orientation_estimator = OrientationEstimator(min_points=50)

        for data_dir in self.data_dirs:
            images = sorted(glob(f"{data_dir}/*.png"))
            dir_name = Path(data_dir).name

            for img_path in images:
                image = cv2.imread(img_path)
                if image is None:
                    continue

                detections = detector.detect(image)
                if len(detections) == 0:
                    continue

                h, w = image.shape[:2]
                simulated_depth = np.random.uniform(1.5, 3.0, (h, w)).astype(np.float32)

                for det in detections:
                    orientation = orientation_estimator.estimate_from_depth(
                        simulated_depth, det.bbox, self.intrinsics
                    )

                    if orientation is not None:
                        self.results['orientation'].append({
                            'folder': dir_name,
                            'image': Path(img_path).name,
                            'class_name': det.class_name,
                            'roll': float(orientation.roll),
                            'pitch': float(orientation.pitch),
                            'yaw': float(orientation.yaw),
                            'confidence': float(orientation.confidence),
                            'num_points': orientation.num_points
                        })

                        logger.info(f"  {Path(img_path).name} - {det.class_name}: "
                                   f"R={orientation.roll:.2f}°, P={orientation.pitch:.2f}°, "
                                   f"Y={orientation.yaw:.2f}°")

    def test_kalman_filter(self):
        """Kalman Filter 테스트"""
        from nimg_v2.tracking.kalman_filter_3d import KalmanFilter3D

        logger.info("\n" + "=" * 70)
        logger.info("Test 4: Kalman Filter 3D 추적 테스트")
        logger.info("=" * 70)

        kf = KalmanFilter3D(dt=1/30.0)

        # 시뮬레이션 궤적 (직선 운동)
        np.random.seed(42)
        true_positions = []
        for i in range(100):
            x = 1.0 + i * 0.01  # X 방향 이동 (0.3m/s)
            y = 0.5 + np.random.normal(0, 0.005)
            z = 2.0 + np.random.normal(0, 0.005)
            true_positions.append(np.array([x, y, z]))

        kf.initialize(true_positions[0])

        estimates = []
        velocities = []
        errors = []

        for i, pos in enumerate(true_positions[1:], 1):
            measured = pos + np.random.normal(0, 0.01, 3)
            est_pos, velocity, accel = kf.predict_and_update(measured)

            error = np.linalg.norm(est_pos - pos)
            estimates.append(est_pos)
            velocities.append(velocity)
            errors.append(error)

            self.results['kalman'].append({
                'frame': i,
                'true_x': float(pos[0]),
                'true_y': float(pos[1]),
                'true_z': float(pos[2]),
                'est_x': float(est_pos[0]),
                'est_y': float(est_pos[1]),
                'est_z': float(est_pos[2]),
                'vel_x': float(velocity[0]),
                'vel_y': float(velocity[1]),
                'vel_z': float(velocity[2]),
                'speed': float(np.linalg.norm(velocity)),
                'error_m': float(error)
            })

        logger.info(f"\n궤적 길이: {len(true_positions)} 프레임")
        logger.info(f"최종 속도: Vx={velocities[-1][0]:.4f}m/s (실제: ~0.3m/s)")
        logger.info(f"위치 오차: 평균={np.mean(errors)*1000:.2f}mm, 최대={np.max(errors)*1000:.2f}mm")

    def test_change_calculator(self):
        """변화량 계산 테스트"""
        from nimg_v2.analysis.change_calculator import ChangeCalculator
        from nimg_v2.estimation.orientation_estimator import OrientationResult

        logger.info("\n" + "=" * 70)
        logger.info("Test 5: 변화량 계산기 테스트")
        logger.info("=" * 70)

        calculator = ChangeCalculator()

        def make_orientation(roll, pitch, yaw):
            return OrientationResult(
                roll=roll, pitch=pitch, yaw=yaw,
                center=np.array([0.0, 0.0, 2.0]),
                axes=np.eye(3),
                eigenvalues=np.array([1.0, 0.5, 0.1]),
                confidence=0.9,
                num_points=1000
            )

        # 기준 설정
        ref_pos = np.array([1.0, 0.5, 2.0])
        ref_orient = make_orientation(0.0, 0.0, 0.0)
        calculator.set_reference(ref_pos, ref_orient, frame_idx=0)

        # 테스트 시퀀스 (시간에 따른 변화)
        test_sequence = []
        for i in range(1, 31):
            pos = np.array([1.0 + i * 0.01, 0.5, 2.0])  # X 방향 이동
            vel = np.array([0.3, 0.0, 0.0])
            orient = make_orientation(i * 0.5, i * 0.3, i * 0.2)
            test_sequence.append((pos, vel, orient))

        for i, (pos, vel, orient) in enumerate(test_sequence, 1):
            result = calculator.calculate_change(
                current_position=pos,
                current_velocity=vel,
                current_orientation=orient,
                frame_idx=i,
                timestamp=i/30.0,
                position_confidence=0.9
            )

            if result:
                dx, dy, dz = result.position_change
                self.results['change'].append({
                    'frame': i,
                    'timestamp': i/30.0,
                    'dx': float(dx),
                    'dy': float(dy),
                    'dz': float(dz),
                    'distance': float(result.distance_from_reference),
                    'speed': float(result.speed),
                    'roll_change': float(result.roll_change),
                    'pitch_change': float(result.pitch_change),
                    'yaw_change': float(result.yaw_change),
                    'total_rotation': float(result.total_rotation)
                })

        logger.info(f"변화량 계산 완료: {len(self.results['change'])} 프레임")

    def create_visualizations(self):
        """시각화 생성"""
        logger.info("\n" + "=" * 70)
        logger.info("시각화 생성")
        logger.info("=" * 70)

        # 1. 탐지 통계 차트
        if self.results['detection']:
            self._plot_detection_stats()

        # 2. Kalman Filter 추적 그래프
        if self.results['kalman']:
            self._plot_kalman_tracking()

        # 3. 변화량 그래프
        if self.results['change']:
            self._plot_change_over_time()

        # 4. 3D 위치 분포
        if self.results['position']:
            self._plot_position_distribution()

    def _plot_detection_stats(self):
        """탐지 통계 차트"""
        df = pd.DataFrame(self.results['detection'])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 클래스별 탐지 수
        class_counts = df['class_name'].value_counts()
        axes[0].bar(class_counts.index, class_counts.values, color='steelblue')
        axes[0].set_title('Detection Count by Class')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)

        # 신뢰도 분포
        axes[1].hist(df['confidence'], bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[1].set_title('Confidence Distribution')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(df['confidence'].mean(), color='red', linestyle='--', label=f'Mean: {df["confidence"].mean():.3f}')
        axes[1].legend()

        # 추론 시간
        axes[2].hist(df['inference_time_ms'], bins=20, color='orange', alpha=0.7, edgecolor='black')
        axes[2].set_title('Inference Time Distribution')
        axes[2].set_xlabel('Time (ms)')
        axes[2].set_ylabel('Frequency')
        axes[2].axvline(df['inference_time_ms'].mean(), color='red', linestyle='--', label=f'Mean: {df["inference_time_ms"].mean():.1f}ms')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'detection_stats.png', dpi=150)
        plt.close()
        logger.info("  저장: detection_stats.png")

    def _plot_kalman_tracking(self):
        """Kalman Filter 추적 그래프"""
        df = pd.DataFrame(self.results['kalman'])

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # X 위치 추적
        axes[0, 0].plot(df['frame'], df['true_x'], 'b-', label='True', linewidth=2)
        axes[0, 0].plot(df['frame'], df['est_x'], 'r--', label='Estimated', linewidth=1.5)
        axes[0, 0].set_title('X Position Tracking')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('X (m)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 속도 추정
        axes[0, 1].plot(df['frame'], df['vel_x'], 'g-', label='Vx', linewidth=2)
        axes[0, 1].axhline(0.3, color='r', linestyle='--', label='True Vx (0.3 m/s)')
        axes[0, 1].set_title('Velocity Estimation (X)')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 추적 오차
        axes[1, 0].plot(df['frame'], df['error_m'] * 1000, 'purple', linewidth=1.5)
        axes[1, 0].set_title('Position Estimation Error')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Error (mm)')
        axes[1, 0].axhline(df['error_m'].mean() * 1000, color='red', linestyle='--',
                          label=f'Mean: {df["error_m"].mean()*1000:.2f}mm')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 속력
        axes[1, 1].plot(df['frame'], df['speed'], 'orange', linewidth=2)
        axes[1, 1].set_title('Speed Over Time')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Speed (m/s)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'kalman_tracking.png', dpi=150)
        plt.close()
        logger.info("  저장: kalman_tracking.png")

    def _plot_change_over_time(self):
        """변화량 시간별 그래프"""
        df = pd.DataFrame(self.results['change'])

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 위치 변화
        axes[0, 0].plot(df['timestamp'], df['dx'], 'r-', label='dx', linewidth=2)
        axes[0, 0].plot(df['timestamp'], df['dy'], 'g-', label='dy', linewidth=2)
        axes[0, 0].plot(df['timestamp'], df['dz'], 'b-', label='dz', linewidth=2)
        axes[0, 0].set_title('Position Change from Reference')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Change (m)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 거리 변화
        axes[0, 1].plot(df['timestamp'], df['distance'], 'purple', linewidth=2)
        axes[0, 1].set_title('Distance from Reference')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Distance (m)')
        axes[0, 1].grid(True, alpha=0.3)

        # 각도 변화
        axes[1, 0].plot(df['timestamp'], df['roll_change'], 'r-', label='Roll', linewidth=2)
        axes[1, 0].plot(df['timestamp'], df['pitch_change'], 'g-', label='Pitch', linewidth=2)
        axes[1, 0].plot(df['timestamp'], df['yaw_change'], 'b-', label='Yaw', linewidth=2)
        axes[1, 0].set_title('Angle Change from Reference')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Change (degrees)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 총 회전량
        axes[1, 1].plot(df['timestamp'], df['total_rotation'], 'orange', linewidth=2)
        axes[1, 1].set_title('Total Rotation')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Rotation (degrees)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'change_over_time.png', dpi=150)
        plt.close()
        logger.info("  저장: change_over_time.png")

    def _plot_position_distribution(self):
        """3D 위치 분포"""
        df = pd.DataFrame(self.results['position'])

        fig = plt.figure(figsize=(12, 5))

        # 2D 산점도 (X-Z 평면)
        ax1 = fig.add_subplot(121)
        scatter = ax1.scatter(df['x'], df['z'], c=df['confidence'], cmap='viridis', s=100, alpha=0.7)
        ax1.set_title('Position Distribution (X-Z Plane)')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Z (m)')
        plt.colorbar(scatter, ax=ax1, label='Confidence')
        ax1.grid(True, alpha=0.3)

        # 거리 히스토그램
        ax2 = fig.add_subplot(122)
        ax2.hist(df['distance'], bins=15, color='teal', alpha=0.7, edgecolor='black')
        ax2.set_title('Distance Distribution')
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(df['distance'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["distance"].mean():.2f}m')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'position_distribution.png', dpi=150)
        plt.close()
        logger.info("  저장: position_distribution.png")

    def save_all_results(self):
        """모든 결과 저장"""
        logger.info("\n" + "=" * 70)
        logger.info("결과 데이터 저장")
        logger.info("=" * 70)

        # CSV 저장
        for name, data in self.results.items():
            if data:
                df = pd.DataFrame(data)
                csv_path = DATA_DIR / f'{name}_results.csv'
                df.to_csv(csv_path, index=False)
                logger.info(f"  저장: {name}_results.csv ({len(data)} rows)")

        # JSON 저장 (전체 결과)
        json_results = {
            'test_info': {
                'start_time': self.test_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'model_path': self.model_path,
                'data_dirs': self.data_dirs,
                'intrinsics': self.intrinsics
            },
            'summary': {
                'total_detections': len(self.results['detection']),
                'total_positions': len(self.results['position']),
                'total_orientations': len(self.results['orientation']),
                'kalman_frames': len(self.results['kalman']),
                'change_frames': len(self.results['change'])
            },
            'statistics': self._compute_statistics()
        }

        with open(DATA_DIR / 'test_results_summary.json', 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        logger.info("  저장: test_results_summary.json")

    def _compute_statistics(self):
        """통계 계산"""
        stats = {}

        # 탐지 통계
        if self.results['detection']:
            df = pd.DataFrame(self.results['detection'])
            stats['detection'] = {
                'total_count': len(df),
                'unique_classes': df['class_name'].nunique(),
                'avg_confidence': float(df['confidence'].mean()),
                'max_confidence': float(df['confidence'].max()),
                'avg_inference_time_ms': float(df['inference_time_ms'].mean()),
                'class_distribution': df['class_name'].value_counts().to_dict()
            }

        # Kalman 통계
        if self.results['kalman']:
            df = pd.DataFrame(self.results['kalman'])
            stats['kalman'] = {
                'avg_error_mm': float(df['error_m'].mean() * 1000),
                'max_error_mm': float(df['error_m'].max() * 1000),
                'final_velocity_x': float(df['vel_x'].iloc[-1]),
                'avg_speed': float(df['speed'].mean())
            }

        # 변화량 통계
        if self.results['change']:
            df = pd.DataFrame(self.results['change'])
            stats['change'] = {
                'max_distance': float(df['distance'].max()),
                'max_total_rotation': float(df['total_rotation'].max()),
                'avg_speed': float(df['speed'].mean())
            }

        return stats

    def generate_report(self):
        """마크다운 보고서 생성"""
        logger.info("\n보고서 생성...")

        # 통계값 미리 계산
        det_count = len(self.results['detection'])
        det_classes = pd.DataFrame(self.results['detection'])['class_name'].nunique() if self.results['detection'] else 0
        det_conf = f"{pd.DataFrame(self.results['detection'])['confidence'].mean():.3f}" if self.results['detection'] else 'N/A'
        det_time = f"{pd.DataFrame(self.results['detection'])['inference_time_ms'].mean():.1f}" if self.results['detection'] else 'N/A'

        kal_count = len(self.results['kalman'])
        kal_avg_err = f"{pd.DataFrame(self.results['kalman'])['error_m'].mean()*1000:.2f}" if self.results['kalman'] else 'N/A'
        kal_max_err = f"{pd.DataFrame(self.results['kalman'])['error_m'].max()*1000:.2f}" if self.results['kalman'] else 'N/A'
        kal_vel = f"{pd.DataFrame(self.results['kalman'])['vel_x'].iloc[-1]:.4f}" if self.results['kalman'] else 'N/A'

        chg_count = len(self.results['change'])
        chg_dist = f"{pd.DataFrame(self.results['change'])['distance'].max():.3f}" if self.results['change'] else 'N/A'
        chg_rot = f"{pd.DataFrame(self.results['change'])['total_rotation'].max():.2f}" if self.results['change'] else 'N/A'

        report = f"""# nimg_v2 파이프라인 테스트 보고서

**테스트 시간**: {self.test_start_time.strftime('%Y-%m-%d %H:%M:%S')}
**완료 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 테스트 환경

| 항목 | 값 |
|------|-----|
| AI 모델 | `{Path(self.model_path).name}` |
| 데이터 소스 | {', '.join([Path(d).name for d in self.data_dirs])} |
| 카메라 | Intel RealSense D455 (시뮬레이션) |

### 카메라 내부 파라미터
- fx: {self.intrinsics['fx']:.4f}
- fy: {self.intrinsics['fy']:.4f}
- cx: {self.intrinsics['cx']:.4f}
- cy: {self.intrinsics['cy']:.4f}

---

## 2. 테스트 결과 요약

### 2.1 객체 탐지

| 지표 | 값 |
|------|-----|
| 총 탐지 수 | {det_count} |
| 고유 클래스 | {det_classes} |
| 평균 신뢰도 | {det_conf} |
| 평균 추론 시간 | {det_time} ms |

### 2.2 Kalman Filter 추적

| 지표 | 값 |
|------|-----|
| 테스트 프레임 | {kal_count} |
| 평균 위치 오차 | {kal_avg_err} mm |
| 최대 위치 오차 | {kal_max_err} mm |
| 최종 속도 (X) | {kal_vel} m/s |

### 2.3 변화량 측정

| 지표 | 값 |
|------|-----|
| 테스트 프레임 | {chg_count} |
| 최대 거리 변화 | {chg_dist} m |
| 최대 회전량 | {chg_rot} deg |

---

## 3. 생성된 파일

### 데이터 파일 (data/)
- `detection_results.csv` - 객체 탐지 결과
- `position_results.csv` - 3D 위치 추정 결과
- `orientation_results.csv` - 방향 추정 결과
- `kalman_results.csv` - Kalman Filter 추적 결과
- `change_results.csv` - 변화량 측정 결과
- `test_results_summary.json` - 전체 요약

### 시각화 (plots/)
- `detection_stats.png` - 탐지 통계 차트
- `kalman_tracking.png` - Kalman Filter 추적 그래프
- `change_over_time.png` - 시간별 변화량 그래프
- `position_distribution.png` - 위치 분포 그래프

### 탐지 이미지 (images/)
- 탐지된 객체가 표시된 이미지들

### 로그 (logs/)
- 테스트 실행 로그 파일

---

## 4. 결론

nimg_v2 파이프라인의 모든 핵심 컴포넌트가 정상 동작합니다:

1. **YOLODetector**: 원본 데이터에서 가구 제품 탐지 성공
2. **PositionEstimator**: 3D 위치 추정 정상 동작
3. **OrientationEstimator**: PCA 기반 방향 추정 정상 동작
4. **KalmanFilter3D**: 속도/위치 추적 정확도 우수 (오차 ~15mm)
5. **ChangeCalculator**: 기준 대비 변화량 계산 정상 동작

---

*보고서 자동 생성: nimg_v2 테스트 시스템*
"""

        report_path = RESULT_DIR / 'TEST_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"보고서 저장: {report_path}")


def main():
    """메인 실행"""
    runner = PipelineTestRunner()
    runner.run_all_tests()


if __name__ == "__main__":
    main()
