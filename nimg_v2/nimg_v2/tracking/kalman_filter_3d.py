"""
3D Kalman Filter 모듈
Constant Acceleration (CA) 모델 기반 3D 위치/속도/가속도 추정
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """Kalman Filter 상태"""
    position: np.ndarray    # [x, y, z]
    velocity: np.ndarray    # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    position_covariance: np.ndarray  # 위치 공분산
    velocity_covariance: np.ndarray  # 속도 공분산


class KalmanFilter3D:
    """
    Constant Acceleration (CA) 모델 기반 3D Kalman Filter

    상태 벡터: [x, y, z, vx, vy, vz, ax, ay, az] (9차원)
    측정 벡터: [x, y, z] (3차원)

    D455 카메라의 depth 오차 특성을 반영한 적응형 측정 노이즈 지원
    """

    def __init__(
        self,
        dt: float = 1/30.0,
        process_noise: float = 0.1,
        measurement_noise_base: float = 0.005,
        acceleration_noise_multiplier: float = 10.0
    ):
        """
        Args:
            dt: 시간 간격 (초)
            process_noise: 프로세스 노이즈 (기본값)
            measurement_noise_base: 기본 측정 노이즈 (1m에서의 오차, 미터)
            acceleration_noise_multiplier: 가속도 노이즈 배율
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise_base = measurement_noise_base
        self.acceleration_noise_multiplier = acceleration_noise_multiplier

        # 상태 차원
        self.dim_x = 9  # [x, y, z, vx, vy, vz, ax, ay, az]
        self.dim_z = 3  # [x, y, z]

        # 상태 벡터
        self.x = np.zeros(self.dim_x)

        # 상태 전이 행렬 F (Constant Acceleration 모델)
        self.F = self._build_transition_matrix(dt)

        # 측정 행렬 H (위치만 측정)
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 2] = 1  # z

        # 상태 공분산 행렬 P
        self.P = np.eye(self.dim_x)

        # 프로세스 노이즈 행렬 Q
        self.Q = self._build_process_noise_matrix(process_noise)

        # 측정 노이즈 행렬 R (초기값, 적응형으로 업데이트)
        self.R = np.eye(self.dim_z) * (measurement_noise_base ** 2)

        # 초기화 상태
        self.initialized = False
        self.frame_count = 0

        logger.debug(f"KalmanFilter3D initialized: dt={dt}, process_noise={process_noise}")

    def _build_transition_matrix(self, dt: float) -> np.ndarray:
        """상태 전이 행렬 생성 (CA 모델)"""
        F = np.eye(9)
        dt2 = 0.5 * dt * dt

        # 위치 = 이전 위치 + 속도*dt + 0.5*가속도*dt^2
        F[0, 3] = dt   # x <- vx
        F[0, 6] = dt2  # x <- ax
        F[1, 4] = dt   # y <- vy
        F[1, 7] = dt2  # y <- ay
        F[2, 5] = dt   # z <- vz
        F[2, 8] = dt2  # z <- az

        # 속도 = 이전 속도 + 가속도*dt
        F[3, 6] = dt   # vx <- ax
        F[4, 7] = dt   # vy <- ay
        F[5, 8] = dt   # vz <- az

        return F

    def _build_process_noise_matrix(self, q: float) -> np.ndarray:
        """프로세스 노이즈 행렬 생성"""
        Q = np.eye(self.dim_x) * q

        # 위치 노이즈 (작음)
        Q[0:3, 0:3] *= 0.1

        # 속도 노이즈 (중간)
        Q[3:6, 3:6] *= 1.0

        # 가속도 노이즈 (큼)
        Q[6:9, 6:9] *= self.acceleration_noise_multiplier

        return Q

    def initialize(self, position: np.ndarray, velocity: Optional[np.ndarray] = None):
        """
        필터 초기화

        Args:
            position: 초기 3D 위치 [x, y, z]
            velocity: 초기 3D 속도 [vx, vy, vz] (옵션)
        """
        self.x = np.zeros(self.dim_x)
        self.x[0:3] = position

        if velocity is not None:
            self.x[3:6] = velocity

        # 공분산 초기화
        self.P = np.eye(self.dim_x)
        self.P[0:3, 0:3] *= 0.1      # 위치 불확실성 낮음
        self.P[3:6, 3:6] *= 1.0      # 속도 불확실성 중간
        self.P[6:9, 6:9] *= 10.0     # 가속도 불확실성 높음

        self.initialized = True
        self.frame_count = 0

        logger.debug(f"KalmanFilter3D initialized with position: {position}")

    def update_measurement_noise(self, distance: float):
        """
        거리 기반 적응형 측정 노이즈 업데이트

        D455 카메라의 depth 오차 특성:
        - baseline = 95mm
        - 오차 ∝ distance²

        Args:
            distance: 객체까지의 거리 (미터)
        """
        # D455 depth 오차 모델: error ≈ base_error * distance²
        error = self.measurement_noise_base * (distance ** 2)

        # 최소/최대 범위 제한
        error = np.clip(error, 0.002, 0.1)  # 2mm ~ 100mm

        # 측정 노이즈 행렬 업데이트
        self.R = np.eye(self.dim_z) * (error ** 2)

    def predict(self) -> np.ndarray:
        """
        예측 단계

        Returns:
            예측된 상태 벡터
        """
        if not self.initialized:
            raise RuntimeError("Kalman filter not initialized")

        # 상태 예측: x_pred = F * x
        self.x = self.F @ self.x

        # 공분산 예측: P_pred = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        업데이트 단계

        Args:
            measurement: 측정값 [x, y, z]

        Returns:
            업데이트된 상태 벡터
        """
        if not self.initialized:
            raise RuntimeError("Kalman filter not initialized")

        # 잔차: y = z - H * x
        y = measurement - self.H @ self.x

        # 잔차 공분산: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # 칼만 이득: K = P * H^T * S^(-1)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 상태 업데이트: x = x + K * y
        self.x = self.x + K @ y

        # 공분산 업데이트: P = (I - K * H) * P
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

        self.frame_count += 1

        return self.x.copy()

    def predict_and_update(
        self,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        예측 및 업데이트를 한 번에 수행

        Args:
            measurement: 측정값 [x, y, z]

        Returns:
            (position, velocity, acceleration): 각각 [x,y,z] 배열
        """
        if not self.initialized:
            self.initialize(measurement)
            return measurement, np.zeros(3), np.zeros(3)

        # 거리 기반 적응형 노이즈 업데이트
        distance = np.linalg.norm(measurement)
        self.update_measurement_noise(distance)

        # 예측 및 업데이트
        self.predict()
        self.update(measurement)

        # 상태 추출
        position = self.x[0:3].copy()
        velocity = self.x[3:6].copy()
        acceleration = self.x[6:9].copy()

        return position, velocity, acceleration

    def get_state(self) -> KalmanState:
        """현재 상태 반환"""
        return KalmanState(
            position=self.x[0:3].copy(),
            velocity=self.x[3:6].copy(),
            acceleration=self.x[6:9].copy(),
            position_covariance=self.P[0:3, 0:3].copy(),
            velocity_covariance=self.P[3:6, 3:6].copy()
        )

    def get_position(self) -> np.ndarray:
        """현재 위치 반환"""
        return self.x[0:3].copy()

    def get_velocity(self) -> np.ndarray:
        """현재 속도 반환"""
        return self.x[3:6].copy()

    def get_speed(self) -> float:
        """현재 속력 반환 (m/s)"""
        return np.linalg.norm(self.x[3:6])

    def get_acceleration(self) -> np.ndarray:
        """현재 가속도 반환"""
        return self.x[6:9].copy()

    def get_position_uncertainty(self) -> np.ndarray:
        """위치 불확실성 (표준편차) 반환"""
        return np.sqrt(np.diag(self.P[0:3, 0:3]))

    def get_velocity_uncertainty(self) -> np.ndarray:
        """속도 불확실성 (표준편차) 반환"""
        return np.sqrt(np.diag(self.P[3:6, 3:6]))

    def predict_future(self, steps: int) -> np.ndarray:
        """
        미래 위치 예측 (현재 상태 변경 없이)

        Args:
            steps: 예측할 프레임 수

        Returns:
            예측된 위치 [x, y, z]
        """
        x_pred = self.x.copy()

        for _ in range(steps):
            x_pred = self.F @ x_pred

        return x_pred[0:3]

    def reset(self):
        """필터 리셋"""
        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x)
        self.initialized = False
        self.frame_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """상태를 딕셔너리로 변환"""
        return {
            'position': self.x[0:3].tolist(),
            'velocity': self.x[3:6].tolist(),
            'acceleration': self.x[6:9].tolist(),
            'speed': self.get_speed(),
            'position_uncertainty': self.get_position_uncertainty().tolist(),
            'velocity_uncertainty': self.get_velocity_uncertainty().tolist(),
            'initialized': self.initialized,
            'frame_count': self.frame_count
        }


class KalmanFilter3DWithIMU(KalmanFilter3D):
    """
    IMU 데이터를 활용한 확장 Kalman Filter

    가속도계 데이터를 추가 측정으로 사용
    """

    def __init__(
        self,
        dt: float = 1/30.0,
        process_noise: float = 0.1,
        measurement_noise_base: float = 0.005,
        imu_noise: float = 0.01
    ):
        super().__init__(dt, process_noise, measurement_noise_base)

        self.imu_noise = imu_noise

        # IMU 측정 행렬 (가속도)
        self.H_imu = np.zeros((3, self.dim_x))
        self.H_imu[0, 6] = 1  # ax
        self.H_imu[1, 7] = 1  # ay
        self.H_imu[2, 8] = 1  # az

        # IMU 측정 노이즈
        self.R_imu = np.eye(3) * (imu_noise ** 2)

    def update_with_imu(
        self,
        position: np.ndarray,
        imu_accel: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        위치와 IMU 데이터로 업데이트

        Args:
            position: 측정된 위치 [x, y, z]
            imu_accel: IMU 가속도 [ax, ay, az] (옵션)

        Returns:
            (position, velocity, acceleration)
        """
        if not self.initialized:
            self.initialize(position)
            if imu_accel is not None:
                self.x[6:9] = imu_accel
            return position, np.zeros(3), imu_accel if imu_accel is not None else np.zeros(3)

        # 예측
        self.predict()

        # 위치 업데이트
        distance = np.linalg.norm(position)
        self.update_measurement_noise(distance)
        self.update(position)

        # IMU 업데이트 (있는 경우)
        if imu_accel is not None:
            self._update_imu(imu_accel)

        return self.x[0:3].copy(), self.x[3:6].copy(), self.x[6:9].copy()

    def _update_imu(self, imu_accel: np.ndarray):
        """IMU 가속도로 업데이트"""
        # 중력 보정 (카메라 좌표계 기준)
        gravity = np.array([0, 9.81, 0])  # Y축이 아래 방향
        corrected_accel = imu_accel - gravity

        # 잔차
        y = corrected_accel - self.H_imu @ self.x

        # 잔차 공분산
        S = self.H_imu @ self.P @ self.H_imu.T + self.R_imu

        # 칼만 이득
        K = self.P @ self.H_imu.T @ np.linalg.inv(S)

        # 상태 업데이트
        self.x = self.x + K @ y

        # 공분산 업데이트
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H_imu) @ self.P
