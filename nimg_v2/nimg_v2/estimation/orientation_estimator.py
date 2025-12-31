"""
3D 방향 추정 모듈
PCA/OBB 기반 Roll, Pitch, Yaw 추정
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Open3D 임포트 (없으면 fallback 사용)
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logger.warning("Open3D not available, using numpy-only implementation")


@dataclass
class OrientationResult:
    """방향 추정 결과"""
    roll: float    # X축 회전 (degrees)
    pitch: float   # Y축 회전 (degrees)
    yaw: float     # Z축 회전 (degrees)
    center: np.ndarray     # 객체 중심점 [x, y, z]
    axes: np.ndarray       # 주축 방향 벡터 (3x3)
    eigenvalues: np.ndarray  # PCA 고유값
    confidence: float      # 추정 신뢰도 (0-1)
    num_points: int        # 사용된 포인트 수


class OrientationEstimator:
    """
    PCA/OBB 기반 3D 방향 추정기

    Point Cloud에서 주성분 분석을 통해
    객체의 Roll, Pitch, Yaw 각도를 추정합니다.
    """

    def __init__(
        self,
        min_points: int = 100,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0
    ):
        """
        Args:
            min_points: 최소 포인트 수
            outlier_nb_neighbors: 이상치 제거용 이웃 수
            outlier_std_ratio: 이상치 표준편차 비율
        """
        self.min_points = min_points
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio

        logger.debug(f"OrientationEstimator initialized: min_points={min_points}")

    def estimate_from_depth(
        self,
        depth: np.ndarray,
        bbox: Tuple[int, int, int, int],
        intrinsics: Dict[str, float],
        depth_min: float = 0.1,
        depth_max: float = 10.0
    ) -> Optional[OrientationResult]:
        """
        Depth 이미지에서 3D 방향 추정

        Args:
            depth: Depth 이미지 (미터 단위)
            bbox: 2D 바운딩 박스 (x, y, width, height)
            intrinsics: 카메라 내부 파라미터 {'fx', 'fy', 'cx', 'cy'}
            depth_min: 최소 유효 거리
            depth_max: 최대 유효 거리

        Returns:
            OrientationResult 또는 None
        """
        x, y, w, h = bbox

        # ROI 추출
        depth_roi = depth[y:y+h, x:x+w]

        # 유효 depth 마스크
        valid_mask = (depth_roi > depth_min) & (depth_roi < depth_max)

        if np.sum(valid_mask) < self.min_points:
            logger.debug(f"Insufficient points: {np.sum(valid_mask)} < {self.min_points}")
            return None

        # 2D → 3D 변환
        points_3d = self._depth_to_pointcloud(
            depth_roi, valid_mask, x, y, intrinsics
        )

        if len(points_3d) < self.min_points:
            return None

        # PCA 분석
        return self._pca_orientation(points_3d)

    def estimate_from_pointcloud(
        self,
        points: np.ndarray
    ) -> Optional[OrientationResult]:
        """
        Point Cloud에서 직접 방향 추정

        Args:
            points: 3D 포인트 배열 (N, 3)

        Returns:
            OrientationResult 또는 None
        """
        if len(points) < self.min_points:
            logger.debug(f"Insufficient points: {len(points)} < {self.min_points}")
            return None

        return self._pca_orientation(points)

    def _depth_to_pointcloud(
        self,
        depth_roi: np.ndarray,
        mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        intrinsics: Dict[str, float]
    ) -> np.ndarray:
        """Depth ROI를 3D Point Cloud로 변환"""
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']

        # 유효 픽셀 좌표
        v, u = np.where(mask)
        z = depth_roi[v, u]

        # 전체 이미지 좌표로 변환
        u_full = u + offset_x
        v_full = v + offset_y

        # 3D 좌표 계산
        x_3d = (u_full - cx) * z / fx
        y_3d = (v_full - cy) * z / fy

        return np.column_stack([x_3d, y_3d, z])

    def _pca_orientation(self, points: np.ndarray) -> OrientationResult:
        """PCA 기반 방향 추정"""
        # 이상치 제거
        if HAS_OPEN3D and len(points) > 100:
            points_clean = self._remove_outliers_open3d(points)
        else:
            points_clean = self._remove_outliers_numpy(points)

        if len(points_clean) < self.min_points:
            points_clean = points

        # 중심점 계산
        center = np.mean(points_clean, axis=0)

        # 공분산 행렬 및 PCA
        centered = points_clean - center
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 큰 순서로 정렬 (첫 번째가 가장 큰 주성분)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 주축에서 Roll, Pitch, Yaw 추출
        roll, pitch, yaw = self._axes_to_euler(eigenvectors)

        # 신뢰도 계산
        confidence = self._compute_confidence(eigenvalues, len(points_clean))

        return OrientationResult(
            roll=np.degrees(roll),
            pitch=np.degrees(pitch),
            yaw=np.degrees(yaw),
            center=center,
            axes=eigenvectors,
            eigenvalues=eigenvalues,
            confidence=confidence,
            num_points=len(points_clean)
        )

    def _remove_outliers_open3d(self, points: np.ndarray) -> np.ndarray:
        """Open3D를 사용한 이상치 제거"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.outlier_nb_neighbors,
            std_ratio=self.outlier_std_ratio
        )

        return np.asarray(pcd_clean.points)

    def _remove_outliers_numpy(self, points: np.ndarray) -> np.ndarray:
        """Numpy만 사용한 간단한 이상치 제거"""
        # Z-score 기반 이상치 제거
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        threshold = mean_dist + self.outlier_std_ratio * std_dist
        mask = distances < threshold

        return points[mask]

    def _axes_to_euler(self, axes: np.ndarray) -> Tuple[float, float, float]:
        """
        주축 벡터에서 오일러 각도 추출

        카메라 좌표계 기준:
        - X: 오른쪽
        - Y: 아래
        - Z: 앞 (카메라 방향)

        Args:
            axes: 주축 벡터 (3x3), 각 열이 주성분 방향

        Returns:
            (roll, pitch, yaw) in radians
        """
        # 세 번째 주성분 (가장 짧은 축)이 두께 방향
        # → 물체의 "위" 방향에 해당
        z_axis = axes[:, 2]  # 가장 작은 분산 방향

        # 기준 축과의 각도 계산
        # Pitch: X-Z 평면에서 Z축과의 각도
        pitch = np.arcsin(np.clip(-z_axis[0], -1, 1))

        # Roll: Y-Z 평면에서 Z축과의 각도
        cos_pitch = np.cos(pitch)
        if abs(cos_pitch) > 1e-6:
            roll = np.arctan2(z_axis[1], z_axis[2])
        else:
            roll = 0.0

        # Yaw: X-Y 평면에서의 회전 (첫 번째 주성분 사용)
        x_axis = axes[:, 0]  # 가장 큰 분산 방향
        yaw = np.arctan2(x_axis[1], x_axis[0])

        return roll, pitch, yaw

    def _compute_confidence(
        self,
        eigenvalues: np.ndarray,
        num_points: int
    ) -> float:
        """
        신뢰도 계산

        - 고유값 비율이 명확할수록 좋음
        - 포인트 수가 많을수록 좋음
        """
        total = np.sum(eigenvalues)
        if total < 1e-10:
            return 0.0

        # 첫 번째 주성분의 설명력
        explained_ratio = eigenvalues[0] / total

        # 첫 번째와 두 번째 고유값 비율 (값이 클수록 방향성 명확)
        ratio_12 = eigenvalues[0] / (eigenvalues[1] + 1e-10)

        # 포인트 수 기여
        point_score = min(num_points / 1000, 1.0)

        # 종합 신뢰도
        confidence = 0.3 * explained_ratio + 0.4 * min(ratio_12 / 10, 1.0) + 0.3 * point_score

        return min(max(confidence, 0.0), 1.0)


class OrientationTracker:
    """
    방향 추정 결과 추적 및 스무딩
    """

    def __init__(
        self,
        smoothing_factor: float = 0.3,
        max_angle_change: float = 30.0
    ):
        """
        Args:
            smoothing_factor: 스무딩 계수 (0-1, 높을수록 새 값 반영)
            max_angle_change: 프레임당 최대 각도 변화
        """
        self.smoothing_factor = smoothing_factor
        self.max_angle_change = max_angle_change

        self.prev_roll: Optional[float] = None
        self.prev_pitch: Optional[float] = None
        self.prev_yaw: Optional[float] = None

    def update(self, result: OrientationResult) -> OrientationResult:
        """
        스무딩 적용

        Args:
            result: 현재 프레임 추정 결과

        Returns:
            스무딩된 결과
        """
        if self.prev_roll is None:
            self.prev_roll = result.roll
            self.prev_pitch = result.pitch
            self.prev_yaw = result.yaw
            return result

        # 각도 변화 제한
        roll = self._limit_change(result.roll, self.prev_roll)
        pitch = self._limit_change(result.pitch, self.prev_pitch)
        yaw = self._limit_change(result.yaw, self.prev_yaw)

        # 지수 이동 평균
        roll = self._ema(roll, self.prev_roll)
        pitch = self._ema(pitch, self.prev_pitch)
        yaw = self._ema(yaw, self.prev_yaw)

        # 이전 값 저장
        self.prev_roll = roll
        self.prev_pitch = pitch
        self.prev_yaw = yaw

        return OrientationResult(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            center=result.center,
            axes=result.axes,
            eigenvalues=result.eigenvalues,
            confidence=result.confidence,
            num_points=result.num_points
        )

    def _limit_change(self, current: float, previous: float) -> float:
        """각도 변화 제한"""
        diff = self._normalize_angle(current - previous)

        if abs(diff) > self.max_angle_change:
            diff = np.sign(diff) * self.max_angle_change

        return previous + diff

    def _ema(self, current: float, previous: float) -> float:
        """지수 이동 평균"""
        return self.smoothing_factor * current + (1 - self.smoothing_factor) * previous

    def _normalize_angle(self, angle: float) -> float:
        """각도를 -180 ~ 180 범위로 정규화"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def reset(self):
        """추적 리셋"""
        self.prev_roll = None
        self.prev_pitch = None
        self.prev_yaw = None
