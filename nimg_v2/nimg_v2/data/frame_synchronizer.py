"""
프레임 동기화 모듈
RGB, Depth, IMU 데이터의 시간 동기화를 담당
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyncedIMUData:
    """동기화된 IMU 데이터"""
    timestamp: float
    accel: np.ndarray  # [ax, ay, az]
    gyro: np.ndarray   # [gx, gy, gz]
    accel_samples: int  # 평균에 사용된 가속도 샘플 수
    gyro_samples: int   # 평균에 사용된 자이로 샘플 수


class FrameSynchronizer:
    """
    프레임-IMU 데이터 동기화 클래스

    IMU 데이터는 일반적으로 더 높은 주파수로 샘플링되므로,
    각 프레임에 해당하는 시간 윈도우 내의 IMU 데이터를 평균화합니다.
    """

    def __init__(
        self,
        fps: float = 30.0,
        time_window_mode: str = 'center'
    ):
        """
        Args:
            fps: 프레임 레이트
            time_window_mode: 시간 윈도우 모드
                - 'center': 프레임 중심 기준 ±dt/2
                - 'forward': 프레임 시작부터 다음 프레임까지
                - 'backward': 이전 프레임부터 현재 프레임까지
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.time_window_mode = time_window_mode

    def get_time_window(self, frame_idx: int) -> Tuple[float, float]:
        """
        프레임에 해당하는 시간 윈도우 계산

        Args:
            frame_idx: 프레임 인덱스

        Returns:
            (t_start, t_end): 시간 윈도우
        """
        frame_time = frame_idx * self.dt

        if self.time_window_mode == 'center':
            t_start = frame_time - self.dt / 2
            t_end = frame_time + self.dt / 2
        elif self.time_window_mode == 'forward':
            t_start = frame_time
            t_end = frame_time + self.dt
        elif self.time_window_mode == 'backward':
            t_start = frame_time - self.dt
            t_end = frame_time
        else:
            raise ValueError(f"Unknown time_window_mode: {self.time_window_mode}")

        return max(0, t_start), t_end

    def sync_imu_to_frame(
        self,
        frame_idx: int,
        accel_df: pd.DataFrame,
        gyro_df: pd.DataFrame,
        time_column: str = 'rel_time'
    ) -> Optional[SyncedIMUData]:
        """
        프레임 인덱스에 해당하는 IMU 데이터 동기화

        Args:
            frame_idx: 프레임 인덱스
            accel_df: 가속도계 데이터프레임
            gyro_df: 자이로스코프 데이터프레임
            time_column: 시간 컬럼명

        Returns:
            SyncedIMUData: 동기화된 IMU 데이터
        """
        if accel_df is None or gyro_df is None:
            return None

        t_start, t_end = self.get_time_window(frame_idx)
        timestamp = frame_idx * self.dt

        # 가속도계 데이터 필터링 및 평균
        accel_mask = (accel_df[time_column] >= t_start) & (accel_df[time_column] < t_end)
        accel_slice = accel_df[accel_mask]

        if len(accel_slice) > 0:
            accel = np.array([
                accel_slice['x'].mean(),
                accel_slice['y'].mean(),
                accel_slice['z'].mean()
            ])
            accel_samples = len(accel_slice)
        else:
            # 가장 가까운 샘플 사용
            accel = self._get_nearest_sample(accel_df, timestamp, time_column)
            accel_samples = 0

        # 자이로스코프 데이터 필터링 및 평균
        gyro_mask = (gyro_df[time_column] >= t_start) & (gyro_df[time_column] < t_end)
        gyro_slice = gyro_df[gyro_mask]

        if len(gyro_slice) > 0:
            gyro = np.array([
                gyro_slice['x'].mean(),
                gyro_slice['y'].mean(),
                gyro_slice['z'].mean()
            ])
            gyro_samples = len(gyro_slice)
        else:
            gyro = self._get_nearest_sample(gyro_df, timestamp, time_column)
            gyro_samples = 0

        return SyncedIMUData(
            timestamp=timestamp,
            accel=accel,
            gyro=gyro,
            accel_samples=accel_samples,
            gyro_samples=gyro_samples
        )

    def _get_nearest_sample(
        self,
        df: pd.DataFrame,
        target_time: float,
        time_column: str
    ) -> np.ndarray:
        """가장 가까운 시간의 샘플 반환"""
        if len(df) == 0:
            return np.zeros(3)

        idx = (df[time_column] - target_time).abs().idxmin()
        row = df.loc[idx]
        return np.array([row['x'], row['y'], row['z']])

    def batch_sync(
        self,
        frame_indices: List[int],
        accel_df: pd.DataFrame,
        gyro_df: pd.DataFrame,
        time_column: str = 'rel_time'
    ) -> List[Optional[SyncedIMUData]]:
        """
        여러 프레임에 대한 배치 동기화

        Args:
            frame_indices: 프레임 인덱스 리스트
            accel_df: 가속도계 데이터프레임
            gyro_df: 자이로스코프 데이터프레임
            time_column: 시간 컬럼명

        Returns:
            동기화된 IMU 데이터 리스트
        """
        return [
            self.sync_imu_to_frame(idx, accel_df, gyro_df, time_column)
            for idx in frame_indices
        ]


class IMUInterpolator:
    """
    IMU 데이터 보간 클래스

    더 정밀한 시간 동기화가 필요할 때 사용
    """

    def __init__(self, accel_df: pd.DataFrame, gyro_df: pd.DataFrame):
        """
        Args:
            accel_df: 가속도계 데이터프레임
            gyro_df: 자이로스코프 데이터프레임
        """
        self.accel_df = accel_df.sort_values('rel_time').reset_index(drop=True)
        self.gyro_df = gyro_df.sort_values('rel_time').reset_index(drop=True)

    def interpolate_accel(self, timestamp: float) -> np.ndarray:
        """가속도 데이터 선형 보간"""
        return self._interpolate(self.accel_df, timestamp)

    def interpolate_gyro(self, timestamp: float) -> np.ndarray:
        """자이로스코프 데이터 선형 보간"""
        return self._interpolate(self.gyro_df, timestamp)

    def _interpolate(self, df: pd.DataFrame, timestamp: float) -> np.ndarray:
        """선형 보간 수행"""
        if len(df) == 0:
            return np.zeros(3)

        times = df['rel_time'].values

        # 범위 밖이면 가장 가까운 값 반환
        if timestamp <= times[0]:
            return np.array([df.iloc[0]['x'], df.iloc[0]['y'], df.iloc[0]['z']])
        if timestamp >= times[-1]:
            return np.array([df.iloc[-1]['x'], df.iloc[-1]['y'], df.iloc[-1]['z']])

        # 보간할 두 점 찾기
        idx = np.searchsorted(times, timestamp)
        t0, t1 = times[idx - 1], times[idx]

        # 보간 가중치
        alpha = (timestamp - t0) / (t1 - t0) if t1 != t0 else 0.0

        # 선형 보간
        v0 = np.array([df.iloc[idx - 1]['x'], df.iloc[idx - 1]['y'], df.iloc[idx - 1]['z']])
        v1 = np.array([df.iloc[idx]['x'], df.iloc[idx]['y'], df.iloc[idx]['z']])

        return v0 + alpha * (v1 - v0)

    def get_integrated_rotation(
        self,
        t_start: float,
        t_end: float,
        num_steps: int = 10
    ) -> np.ndarray:
        """
        시간 구간 동안의 회전 적분 (간단한 오일러 적분)

        Args:
            t_start: 시작 시간
            t_end: 종료 시간
            num_steps: 적분 스텝 수

        Returns:
            [roll, pitch, yaw] 변화량 (radians)
        """
        dt = (t_end - t_start) / num_steps
        rotation = np.zeros(3)

        for i in range(num_steps):
            t = t_start + (i + 0.5) * dt
            gyro = self.interpolate_gyro(t)
            rotation += gyro * dt

        return rotation
