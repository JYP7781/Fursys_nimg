"""
데이터 로더 모듈
RGB, Depth, IMU 데이터를 통합 로드하는 클래스
"""

import os
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, List, Iterator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """단일 프레임의 모든 데이터를 담는 데이터 클래스"""
    frame_idx: int
    rgb: np.ndarray
    depth: np.ndarray
    timestamp: float
    accel: Optional[np.ndarray] = None  # [ax, ay, az] m/s²
    gyro: Optional[np.ndarray] = None   # [gx, gy, gz] rad/s
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None

    @property
    def has_imu(self) -> bool:
        """IMU 데이터 존재 여부"""
        return self.accel is not None and self.gyro is not None

    @property
    def depth_valid_ratio(self) -> float:
        """유효 depth 픽셀 비율"""
        if self.depth is None:
            return 0.0
        valid = (self.depth > 0.1) & (self.depth < 10.0)
        return np.sum(valid) / self.depth.size


class DataLoader:
    """
    RGB, Depth, IMU 데이터 통합 로더

    사용법:
        loader = DataLoader("./20251208_155531_output")
        for frame in loader:
            process(frame)
    """

    def __init__(
        self,
        data_dir: str,
        depth_scale: float = 0.001,
        fps: float = 30.0,
        depth_min: float = 0.1,
        depth_max: float = 10.0
    ):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
            depth_scale: depth 값을 미터로 변환하는 스케일
            fps: 프레임 레이트 (IMU 동기화용)
            depth_min: 최소 유효 거리 (m)
            depth_max: 최대 유효 거리 (m)
        """
        self.data_dir = Path(data_dir)
        self.depth_scale = depth_scale
        self.fps = fps
        self.depth_min = depth_min
        self.depth_max = depth_max

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # 파일 목록 로드
        self._load_file_lists()

        # IMU 데이터 로드
        self._load_imu_data()

        logger.info(f"DataLoader initialized: {self.num_frames} frames, "
                   f"IMU: {'available' if self.has_imu else 'not available'}")

    def _load_file_lists(self):
        """RGB 및 Depth 파일 목록 로드"""
        all_files = list(self.data_dir.iterdir())

        self.color_files = sorted([
            f.name for f in all_files
            if f.name.startswith('color_') and f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ])

        self.depth_files = sorted([
            f.name for f in all_files
            if f.name.startswith('depth_') and f.suffix.lower() == '.png'
        ])

        if len(self.color_files) == 0:
            raise ValueError(f"No color images found in {self.data_dir}")
        if len(self.depth_files) == 0:
            raise ValueError(f"No depth images found in {self.data_dir}")

        self.num_frames = min(len(self.color_files), len(self.depth_files))
        logger.info(f"Found {len(self.color_files)} color and {len(self.depth_files)} depth images")

    def _load_imu_data(self):
        """IMU 데이터 로드"""
        imu_path = self.data_dir / 'imu_data.csv'

        if imu_path.exists():
            try:
                self.imu_df = pd.read_csv(imu_path)
                self.has_imu = True

                # 타입별 분리
                self.accel_df = self.imu_df[self.imu_df['type'] == 'accel'].copy()
                self.gyro_df = self.imu_df[self.imu_df['type'] == 'gyro'].copy()

                # 타임스탬프 정규화 (상대 시간으로 변환)
                if len(self.accel_df) > 0:
                    min_ts = self.accel_df['timestamp'].min()
                    self.accel_df['rel_time'] = (self.accel_df['timestamp'] - min_ts) / 1000.0
                if len(self.gyro_df) > 0:
                    min_ts = self.gyro_df['timestamp'].min()
                    self.gyro_df['rel_time'] = (self.gyro_df['timestamp'] - min_ts) / 1000.0

                logger.info(f"IMU data loaded: {len(self.accel_df)} accel, {len(self.gyro_df)} gyro samples")
            except Exception as e:
                logger.warning(f"Failed to load IMU data: {e}")
                self.imu_df = None
                self.has_imu = False
        else:
            self.imu_df = None
            self.has_imu = False
            logger.info("No IMU data file found")

    def __len__(self) -> int:
        """프레임 수 반환"""
        return self.num_frames

    def __getitem__(self, idx: int) -> FrameData:
        """인덱스로 프레임 접근"""
        return self.load_frame(idx)

    def __iter__(self) -> Iterator[FrameData]:
        """이터레이터 지원"""
        for idx in range(self.num_frames):
            yield self.load_frame(idx)

    def load_frame(self, idx: int) -> FrameData:
        """
        특정 인덱스의 프레임 데이터 로드

        Args:
            idx: 프레임 인덱스

        Returns:
            FrameData: 프레임 데이터
        """
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Frame index {idx} out of range [0, {self.num_frames})")

        # RGB 로드
        rgb_path = self.data_dir / self.color_files[idx]
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            raise IOError(f"Failed to load RGB image: {rgb_path}")

        # Depth 로드 (16-bit PNG)
        depth_path = self.data_dir / self.depth_files[idx]
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise IOError(f"Failed to load depth image: {depth_path}")

        # 미터 단위로 변환
        depth = depth_raw.astype(np.float32) * self.depth_scale

        # 유효 범위 외 값 처리
        depth[(depth < self.depth_min) | (depth > self.depth_max)] = 0.0

        # 타임스탬프 계산
        timestamp = idx / self.fps

        # IMU 데이터 조회
        accel, gyro = None, None
        if self.has_imu:
            accel, gyro = self._get_imu_for_frame(idx)

        return FrameData(
            frame_idx=idx,
            rgb=rgb,
            depth=depth,
            timestamp=timestamp,
            accel=accel,
            gyro=gyro,
            rgb_path=str(rgb_path),
            depth_path=str(depth_path)
        )

    def _get_imu_for_frame(self, idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        프레임에 해당하는 IMU 데이터 조회

        Args:
            idx: 프레임 인덱스

        Returns:
            (accel, gyro): 각각 [x, y, z] 배열
        """
        if not self.has_imu:
            return None, None

        # 프레임 시간 범위
        t_start = idx / self.fps
        t_end = (idx + 1) / self.fps

        # 해당 시간 범위의 가속도계 데이터 평균
        accel_mask = (self.accel_df['rel_time'] >= t_start) & (self.accel_df['rel_time'] < t_end)
        accel_slice = self.accel_df[accel_mask]

        if len(accel_slice) > 0:
            accel = np.array([
                accel_slice['x'].mean(),
                accel_slice['y'].mean(),
                accel_slice['z'].mean()
            ])
        else:
            # 시간 범위에 데이터가 없으면 전체 평균 사용
            accel = np.array([
                self.accel_df['x'].mean(),
                self.accel_df['y'].mean(),
                self.accel_df['z'].mean()
            ])

        # 해당 시간 범위의 자이로스코프 데이터 평균
        gyro_mask = (self.gyro_df['rel_time'] >= t_start) & (self.gyro_df['rel_time'] < t_end)
        gyro_slice = self.gyro_df[gyro_mask]

        if len(gyro_slice) > 0:
            gyro = np.array([
                gyro_slice['x'].mean(),
                gyro_slice['y'].mean(),
                gyro_slice['z'].mean()
            ])
        else:
            gyro = np.array([
                self.gyro_df['x'].mean(),
                self.gyro_df['y'].mean(),
                self.gyro_df['z'].mean()
            ])

        return accel, gyro

    def get_frame_range(self, start: int, end: int) -> List[FrameData]:
        """
        프레임 범위 로드

        Args:
            start: 시작 인덱스
            end: 종료 인덱스 (미포함)

        Returns:
            프레임 데이터 리스트
        """
        return [self.load_frame(idx) for idx in range(start, min(end, self.num_frames))]

    def get_image_size(self) -> Tuple[int, int]:
        """
        이미지 크기 반환 (첫 번째 프레임 기준)

        Returns:
            (height, width)
        """
        rgb_path = self.data_dir / self.color_files[0]
        rgb = cv2.imread(str(rgb_path))
        return rgb.shape[:2]

    def get_metadata(self) -> dict:
        """데이터셋 메타데이터 반환"""
        h, w = self.get_image_size()
        return {
            'data_dir': str(self.data_dir),
            'num_frames': self.num_frames,
            'fps': self.fps,
            'image_size': {'width': w, 'height': h},
            'has_imu': self.has_imu,
            'depth_scale': self.depth_scale,
            'depth_range': {'min': self.depth_min, 'max': self.depth_max}
        }
