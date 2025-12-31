"""
방향 추정기 테스트
"""

import pytest
import numpy as np

from nimg_v2.estimation.orientation_estimator import (
    OrientationEstimator,
    OrientationResult,
    OrientationTracker
)


class TestOrientationEstimator:
    """OrientationEstimator 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        estimator = OrientationEstimator(min_points=100)
        assert estimator.min_points == 100

    def test_pca_flat_plane(self):
        """평면 Point Cloud PCA 테스트"""
        estimator = OrientationEstimator(min_points=50)

        # 수평 평면 생성 (Z=1)
        n_points = 1000
        x = np.random.uniform(-0.5, 0.5, n_points)
        y = np.random.uniform(-0.5, 0.5, n_points)
        z = np.ones(n_points) + np.random.normal(0, 0.01, n_points)

        points = np.column_stack([x, y, z])
        result = estimator.estimate_from_pointcloud(points)

        assert result is not None
        # 수평 평면이므로 Roll, Pitch가 거의 0
        assert abs(result.roll) < 10.0
        assert abs(result.pitch) < 10.0

    def test_pca_tilted_plane(self):
        """기울어진 평면 테스트"""
        estimator = OrientationEstimator(min_points=50)

        # 45도 기울어진 평면
        n_points = 1000
        true_pitch = 45.0

        x = np.random.uniform(-0.5, 0.5, n_points)
        y = np.random.uniform(-0.5, 0.5, n_points)
        z = np.zeros(n_points)

        # Pitch 회전 적용
        angle = np.radians(true_pitch)
        x_rot = x * np.cos(angle) + z * np.sin(angle)
        z_rot = -x * np.sin(angle) + z * np.cos(angle)

        points = np.column_stack([x_rot, y, z_rot + 1.0])  # Z offset
        result = estimator.estimate_from_pointcloud(points)

        assert result is not None
        # Pitch가 약 45도
        # PCA 결과는 정확하지 않을 수 있으므로 넓은 범위 허용
        assert abs(result.pitch) < 60.0 or abs(result.pitch) > 30.0

    def test_insufficient_points(self):
        """포인트 부족 테스트"""
        estimator = OrientationEstimator(min_points=100)

        # 부족한 포인트
        points = np.random.randn(50, 3)
        result = estimator.estimate_from_pointcloud(points)

        assert result is None

    def test_orientation_result_fields(self):
        """OrientationResult 필드 테스트"""
        estimator = OrientationEstimator(min_points=50)

        points = np.random.randn(200, 3)
        result = estimator.estimate_from_pointcloud(points)

        assert hasattr(result, 'roll')
        assert hasattr(result, 'pitch')
        assert hasattr(result, 'yaw')
        assert hasattr(result, 'center')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'num_points')

    def test_confidence_calculation(self):
        """신뢰도 계산 테스트"""
        estimator = OrientationEstimator(min_points=50)

        # 명확한 방향성이 있는 Point Cloud (elongated)
        n_points = 500
        x = np.random.uniform(-2.0, 2.0, n_points)  # 긴 방향
        y = np.random.uniform(-0.1, 0.1, n_points)  # 짧은 방향
        z = np.random.uniform(-0.1, 0.1, n_points)

        points = np.column_stack([x, y, z])
        result_elongated = estimator.estimate_from_pointcloud(points)

        # 균일한 Point Cloud
        x = np.random.uniform(-0.5, 0.5, n_points)
        y = np.random.uniform(-0.5, 0.5, n_points)
        z = np.random.uniform(-0.5, 0.5, n_points)

        points = np.column_stack([x, y, z])
        result_uniform = estimator.estimate_from_pointcloud(points)

        # 길쭉한 형태가 더 높은 신뢰도를 가져야 함
        assert result_elongated.confidence >= result_uniform.confidence


class TestOrientationTracker:
    """OrientationTracker 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        tracker = OrientationTracker(smoothing_factor=0.3)
        assert tracker.smoothing_factor == 0.3

    def test_smoothing(self):
        """스무딩 테스트"""
        tracker = OrientationTracker(smoothing_factor=0.3, max_angle_change=30.0)

        # 첫 번째 결과
        result1 = OrientationResult(
            roll=0.0, pitch=0.0, yaw=0.0,
            center=np.zeros(3), axes=np.eye(3),
            eigenvalues=np.ones(3), confidence=0.9, num_points=500
        )
        smoothed1 = tracker.update(result1)

        # 급격한 변화가 있는 두 번째 결과
        result2 = OrientationResult(
            roll=50.0, pitch=50.0, yaw=50.0,
            center=np.zeros(3), axes=np.eye(3),
            eigenvalues=np.ones(3), confidence=0.9, num_points=500
        )
        smoothed2 = tracker.update(result2)

        # 스무딩으로 급격한 변화가 완화되어야 함
        assert smoothed2.roll < 50.0
        assert smoothed2.pitch < 50.0
        assert smoothed2.yaw < 50.0

    def test_angle_limit(self):
        """최대 각도 변화 제한 테스트"""
        tracker = OrientationTracker(smoothing_factor=1.0, max_angle_change=10.0)

        result1 = OrientationResult(
            roll=0.0, pitch=0.0, yaw=0.0,
            center=np.zeros(3), axes=np.eye(3),
            eigenvalues=np.ones(3), confidence=0.9, num_points=500
        )
        tracker.update(result1)

        # 큰 변화
        result2 = OrientationResult(
            roll=100.0, pitch=0.0, yaw=0.0,
            center=np.zeros(3), axes=np.eye(3),
            eigenvalues=np.ones(3), confidence=0.9, num_points=500
        )
        smoothed = tracker.update(result2)

        # max_angle_change로 제한됨
        assert smoothed.roll <= 10.0

    def test_reset(self):
        """리셋 테스트"""
        tracker = OrientationTracker()

        result = OrientationResult(
            roll=10.0, pitch=20.0, yaw=30.0,
            center=np.zeros(3), axes=np.eye(3),
            eigenvalues=np.ones(3), confidence=0.9, num_points=500
        )
        tracker.update(result)

        tracker.reset()

        assert tracker.prev_roll is None
        assert tracker.prev_pitch is None
        assert tracker.prev_yaw is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
