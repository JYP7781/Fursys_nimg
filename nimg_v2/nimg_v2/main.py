"""
nimg_v2 메인 파이프라인
RGB + Depth + IMU 데이터를 활용한 상대적 속도변화량/각도변화량 측정 시스템
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml

import numpy as np
import pandas as pd

# 모듈 임포트
from .data.data_loader import DataLoader, FrameData
from .detection.yolo_detector import YOLODetector, Detection
from .detection.item import Item, ItemList
from .tracking.kalman_filter_3d import KalmanFilter3D
from .tracking.object_tracker import ObjectTracker
from .estimation.position_estimator import PositionEstimator
from .estimation.orientation_estimator import OrientationEstimator, OrientationResult
from .estimation.point_cloud_processor import PointCloudProcessor
from .analysis.change_calculator import ChangeCalculator, ChangeResult, ChangeStatistics
from .analysis.reference_manager import ReferenceManager
from .output.result_exporter import ResultExporter
from .output.visualizer import Visualizer, FrameVisualizer

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageProcessingPipeline:
    """
    영상처리 메인 파이프라인

    RGB + Depth + IMU 데이터를 통합 처리하여
    객체의 상대적 속도변화량과 각도변화량을 측정합니다.
    """

    def __init__(
        self,
        model_path: str,
        intrinsics: Dict[str, float],
        reference_frame_idx: int = 0,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            model_path: YOLO 모델 파일 경로
            intrinsics: 카메라 내부 파라미터 {'fx', 'fy', 'cx', 'cy'}
            reference_frame_idx: 기준 프레임 인덱스
            config: 추가 설정
        """
        self.model_path = model_path
        self.intrinsics = intrinsics
        self.reference_frame_idx = reference_frame_idx
        self.config = config or {}

        # 컴포넌트 초기화
        self._init_components()

        # 결과 저장용
        self.results: List[ChangeResult] = []
        self.reference_set = False

        logger.info("ImageProcessingPipeline initialized")

    def _init_components(self):
        """컴포넌트 초기화"""
        # YOLO 탐지기 - YOLOv5 DetectMultiBackend 방식 우선 사용
        conf_threshold = self.config.get('conf_threshold', 0.5)
        iou_threshold = self.config.get('iou_threshold', 0.45)
        img_size = self.config.get('img_size', 640)
        use_half = self.config.get('use_half', True)
        data_yaml = self.config.get('data_yaml', None)

        self.detector = YOLODetector(
            self.model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            img_size=img_size,
            half=use_half,
            use_yolov5_backend=True,  # YOLOv5 방식 우선 사용 (파인튜닝 모델 호환성)
            data_yaml=data_yaml
        )

        # 모델 워밍업
        self.detector.warmup()
        logger.info(f"Detector loaded and warmed up: {self.model_path}")

        # 위치 추정기
        self.position_estimator = PositionEstimator(self.intrinsics)

        # 방향 추정기
        min_points = self.config.get('min_points', 100)
        self.orientation_estimator = OrientationEstimator(min_points=min_points)

        # Kalman Filter
        dt = self.config.get('dt', 1/30.0)
        self.kalman_filter = KalmanFilter3D(dt=dt)

        # 변화량 계산기
        self.change_calculator = ChangeCalculator()

        # 기준 프레임 관리자
        self.reference_manager = ReferenceManager()

    def process_dataset(
        self,
        data_dir: str,
        output_dir: str,
        max_frames: Optional[int] = None,
        progress_interval: int = 100
    ) -> pd.DataFrame:
        """
        전체 데이터셋 처리

        Args:
            data_dir: 데이터 디렉토리 경로
            output_dir: 출력 디렉토리 경로
            max_frames: 최대 처리 프레임 수 (None이면 전체)
            progress_interval: 진행 로그 출력 간격

        Returns:
            결과 DataFrame
        """
        # 데이터 로더 초기화
        loader = DataLoader(data_dir)
        num_frames = len(loader) if max_frames is None else min(len(loader), max_frames)

        logger.info(f"Processing {num_frames} frames from {data_dir}")

        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 결과 초기화
        self.results = []

        for idx in range(num_frames):
            try:
                frame = loader.load_frame(idx)

                if idx == self.reference_frame_idx:
                    # 기준 프레임 처리
                    self._process_reference_frame(frame)
                elif self.reference_set:
                    # 후속 프레임 처리
                    result = self._process_frame(frame)
                    if result is not None:
                        self.results.append(result)

                if idx % progress_interval == 0:
                    logger.info(f"Processed {idx}/{num_frames} frames")

            except Exception as e:
                logger.warning(f"Error processing frame {idx}: {e}")
                continue

        logger.info(f"Processing complete: {len(self.results)} results")

        # DataFrame 변환
        results_df = self._results_to_dataframe()

        # 결과 저장
        self._save_results(output_dir, loader.get_metadata())

        return results_df

    def _process_reference_frame(self, frame: FrameData):
        """기준 프레임 처리"""
        logger.info(f"Processing reference frame: {frame.frame_idx}")

        # 1. 객체 탐지
        detections = self.detector.detect(frame.rgb)
        if len(detections) == 0:
            raise ValueError(f"No object detected in reference frame {frame.frame_idx}")

        # 가장 신뢰도 높은 객체 선택
        best_det = self.detector.get_best_detection(detections, by='confidence')
        bbox = best_det.bbox

        # 2. 3D 위치 추정
        pos_result = self.position_estimator.estimate_position(bbox, frame.depth)
        if pos_result is None:
            raise ValueError("Failed to estimate position for reference frame")

        position = pos_result.position

        # 3. 방향 추정
        orientation = self.orientation_estimator.estimate_from_depth(
            frame.depth, bbox, self.intrinsics
        )

        if orientation is None:
            raise ValueError("Failed to estimate orientation for reference frame")

        # 4. 기준 상태 설정
        self.change_calculator.set_reference(position, orientation, frame.frame_idx)

        # 기준 프레임 관리자에도 저장
        self.reference_manager.set_reference(
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            position=position,
            velocity=np.zeros(3),
            orientation=orientation,
            class_name=best_det.class_name,
            confidence=best_det.confidence
        )

        # Kalman Filter 초기화
        self.kalman_filter.initialize(position)

        self.reference_set = True

        logger.info(
            f"Reference frame set: position=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}), "
            f"orientation=(R:{orientation.roll:.2f}, P:{orientation.pitch:.2f}, Y:{orientation.yaw:.2f})"
        )

    def _process_frame(self, frame: FrameData) -> Optional[ChangeResult]:
        """개별 프레임 처리"""
        # 1. 객체 탐지
        detections = self.detector.detect(frame.rgb)
        if len(detections) == 0:
            return None

        best_det = self.detector.get_best_detection(detections, by='confidence')
        bbox = best_det.bbox

        # 2. 3D 위치 추정
        pos_result = self.position_estimator.estimate_position(bbox, frame.depth)
        if pos_result is None or pos_result.confidence < 0.3:
            return None

        position = pos_result.position

        # 3. Kalman Filter 업데이트 → 속도 추정
        est_position, velocity, acceleration = self.kalman_filter.predict_and_update(position)

        # 4. 방향 추정
        orientation = self.orientation_estimator.estimate_from_depth(
            frame.depth, bbox, self.intrinsics
        )

        if orientation is None:
            return None

        # 5. 변화량 계산
        result = self.change_calculator.calculate_change(
            current_position=est_position,
            current_velocity=velocity,
            current_orientation=orientation,
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            position_confidence=pos_result.confidence
        )

        return result

    def _results_to_dataframe(self) -> pd.DataFrame:
        """결과를 DataFrame으로 변환"""
        if len(self.results) == 0:
            return pd.DataFrame()

        data = [r.to_dict() for r in self.results]
        df = pd.DataFrame(data)

        # 컬럼 순서 정렬
        column_order = [
            'frame_idx', 'timestamp',
            'dx', 'dy', 'dz', 'distance_from_reference',
            'vx', 'vy', 'vz', 'speed',
            'roll_change', 'pitch_change', 'yaw_change', 'total_rotation',
            'overall_confidence'
        ]
        df = df[[c for c in column_order if c in df.columns]]

        return df

    def _save_results(self, output_dir: str, dataset_metadata: Dict):
        """결과 저장"""
        exporter = ResultExporter(output_dir)

        # CSV 저장
        exporter.export_csv(self.results)

        # JSON 저장
        exporter.export_json(
            self.results,
            metadata={
                'reference': self.reference_manager.get_info(),
                'dataset': dataset_metadata
            }
        )

        # 리포트 저장
        exporter.export_report(
            self.results,
            self.reference_manager.get_info(),
            dataset_metadata
        )

        # 시각화 저장
        visualizer = Visualizer(output_dir)
        visualizer.create_all_plots(self.results)

        logger.info(f"Results saved to {output_dir}")

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        stats = ChangeStatistics()
        stats.add_results(self.results)
        return stats.get_statistics()

    def print_summary(self):
        """요약 출력"""
        stats = ChangeStatistics()
        stats.add_results(self.results)
        print(stats.get_summary_string())


def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_intrinsics(intrinsics_path: str) -> Dict[str, float]:
    """카메라 내부 파라미터 로드"""
    with open(intrinsics_path, 'r') as f:
        config = yaml.safe_load(f)

    return {
        'fx': config['camera']['intrinsics']['fx'],
        'fy': config['camera']['intrinsics']['fy'],
        'cx': config['camera']['intrinsics']['cx'],
        'cy': config['camera']['intrinsics']['cy']
    }


def main():
    """메인 실행"""
    parser = argparse.ArgumentParser(
        description='nimg_v2: RGB + Depth + IMU 기반 속도/각도 변화량 측정'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='데이터 디렉토리 경로')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='출력 디렉토리 경로')
    parser.add_argument('--model-path', type=str, required=True,
                        help='YOLO 모델 파일 경로')
    parser.add_argument('--config', type=str, default=None,
                        help='설정 파일 경로 (YAML)')
    parser.add_argument('--intrinsics', type=str, default=None,
                        help='카메라 내부 파라미터 파일 경로 (YAML)')
    parser.add_argument('--reference-frame', type=int, default=0,
                        help='기준 프레임 인덱스')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='최대 처리 프레임 수')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='상세 로그 출력')

    args = parser.parse_args()

    # 로그 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 설정 로드
    config = {}
    if args.config:
        config = load_config(args.config)

    # 카메라 내부 파라미터
    if args.intrinsics:
        intrinsics = load_intrinsics(args.intrinsics)
    else:
        # 기본값 (D455)
        intrinsics = {
            'fx': 636.3392652788157,
            'fy': 636.4266464742717,
            'cx': 654.3418233071645,
            'cy': 399.58963414918554
        }

    # 파이프라인 실행
    pipeline = ImageProcessingPipeline(
        model_path=args.model_path,
        intrinsics=intrinsics,
        reference_frame_idx=args.reference_frame,
        config=config
    )

    results_df = pipeline.process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames
    )

    # 요약 출력
    pipeline.print_summary()

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
