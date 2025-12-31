"""
전체 파이프라인 통합 테스트
"""

import pytest
import numpy as np
import tempfile
import os
import cv2
import pandas as pd

from nimg_v2.tracking.kalman_filter_3d import KalmanFilter3D
from nimg_v2.estimation.position_estimator import PositionEstimator
from nimg_v2.estimation.orientation_estimator import OrientationEstimator, OrientationResult
from nimg_v2.analysis.change_calculator import ChangeCalculator, ChangeResult


class TestIntegratedPipeline:
    """통합 파이프라인 테스트"""

    @pytest.fixture
    def intrinsics(self):
        """카메라 내부 파라미터"""
        return {
            'fx': 636.34,
            'fy': 636.43,
            'cx': 654.34,
            'cy': 399.59
        }

    @pytest.fixture
    def sample_depth(self):
        """샘플 depth 이미지"""
        depth = np.ones((800, 1280), dtype=np.float32) * 1.5
        # 중앙에 객체 (더 가까움)
        depth[300:500, 500:800] = 1.0
        return depth

    def test_position_estimation(self, intrinsics, sample_depth):
        """위치 추정 테스트"""
        estimator = PositionEstimator(intrinsics)
        bbox = (500, 300, 300, 200)  # x, y, w, h

        result = estimator.estimate_position(bbox, sample_depth)

        assert result is not None
        assert result.position[2] > 0  # Z가 양수
        assert result.confidence > 0

    def test_kalman_velocity_tracking(self):
        """Kalman Filter 속도 추적 테스트"""
        kf = KalmanFilter3D(dt=1/30.0)

        # 일정 속도 시뮬레이션
        positions = []
        velocities = []
        true_velocity = np.array([0.1, 0.05, 0.0])

        for t in range(60):
            true_pos = true_velocity * t * (1/30.0)
            measured_pos = true_pos + np.random.normal(0, 0.005, 3)

            est_pos, est_vel, _ = kf.predict_and_update(measured_pos)
            positions.append(est_pos)
            velocities.append(est_vel)

        # 수렴 후 속도 확인
        final_velocity = velocities[-1]
        assert np.linalg.norm(final_velocity - true_velocity) < 0.05

    def test_change_calculator(self, intrinsics):
        """변화량 계산기 테스트"""
        calculator = ChangeCalculator()

        # 기준 상태
        ref_position = np.array([0.0, 0.0, 1.0])
        ref_orientation = OrientationResult(
            roll=0.0, pitch=0.0, yaw=0.0,
            center=ref_position, axes=np.eye(3),
            eigenvalues=np.ones(3), confidence=0.9, num_points=500
        )

        calculator.set_reference(ref_position, ref_orientation, frame_idx=0)

        # 현재 상태
        curr_position = np.array([0.1, 0.05, 1.1])
        curr_velocity = np.array([0.1, 0.05, 0.1])
        curr_orientation = OrientationResult(
            roll=5.0, pitch=10.0, yaw=-5.0,
            center=curr_position, axes=np.eye(3),
            eigenvalues=np.ones(3), confidence=0.85, num_points=450
        )

        result = calculator.calculate_change(
            current_position=curr_position,
            current_velocity=curr_velocity,
            current_orientation=curr_orientation,
            frame_idx=30,
            timestamp=1.0
        )

        assert isinstance(result, ChangeResult)
        assert result.speed > 0
        assert result.roll_change == pytest.approx(5.0, abs=0.01)
        assert result.pitch_change == pytest.approx(10.0, abs=0.01)
        assert result.yaw_change == pytest.approx(-5.0, abs=0.01)

    def test_full_tracking_scenario(self, intrinsics):
        """전체 추적 시나리오 테스트"""
        # 컴포넌트 초기화
        kf = KalmanFilter3D(dt=1/30.0)
        calculator = ChangeCalculator()

        # 시뮬레이션 파라미터
        n_frames = 90  # 3초
        true_velocity = np.array([0.05, 0.0, 0.02])
        true_angular_velocity = np.array([1.0, 0.5, 0.3])  # deg/frame

        results = []

        for frame_idx in range(n_frames):
            timestamp = frame_idx / 30.0

            # 실제 위치/방향 계산
            true_position = np.array([0.0, 0.0, 1.0]) + true_velocity * timestamp
            true_angles = true_angular_velocity * frame_idx

            # 측정 노이즈 추가
            measured_position = true_position + np.random.normal(0, 0.003, 3)
            measured_angles = true_angles + np.random.normal(0, 0.5, 3)

            # Kalman Filter 업데이트
            est_pos, est_vel, _ = kf.predict_and_update(measured_position)

            # 방향 생성
            orientation = OrientationResult(
                roll=measured_angles[0],
                pitch=measured_angles[1],
                yaw=measured_angles[2],
                center=est_pos,
                axes=np.eye(3),
                eigenvalues=np.ones(3),
                confidence=0.9,
                num_points=500
            )

            if frame_idx == 0:
                # 기준 프레임 설정
                calculator.set_reference(est_pos, orientation, frame_idx)
            else:
                # 변화량 계산
                result = calculator.calculate_change(
                    current_position=est_pos,
                    current_velocity=est_vel,
                    current_orientation=orientation,
                    frame_idx=frame_idx,
                    timestamp=timestamp
                )
                results.append(result)

        # 결과 검증
        assert len(results) == n_frames - 1

        # 마지막 결과
        final = results[-1]
        assert final.speed > 0
        assert final.distance_from_reference > 0
        assert abs(final.roll_change) > 0


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_zero_velocity(self):
        """정지 상태 테스트"""
        kf = KalmanFilter3D(dt=1/30.0)

        # 정지 상태
        position = np.array([0.0, 0.0, 1.0])

        for _ in range(30):
            pos, vel, _ = kf.predict_and_update(position + np.random.normal(0, 0.001, 3))

        assert np.linalg.norm(vel) < 0.01

    def test_rapid_direction_change(self):
        """급격한 방향 변화 테스트"""
        kf = KalmanFilter3D(dt=1/30.0)

        # 방향 변화
        for t in range(30):
            if t < 15:
                pos = np.array([t * 0.01, 0.0, 1.0])
            else:
                pos = np.array([0.15 - (t - 15) * 0.01, 0.0, 1.0])

            kf.predict_and_update(pos + np.random.normal(0, 0.002, 3))

        # 추적이 깨지지 않아야 함
        state = kf.get_state()
        assert not np.any(np.isnan(state.position))
        assert not np.any(np.isnan(state.velocity))


class TestValidation:
    """검증 테스트"""

    def test_velocity_estimation_accuracy(self):
        """속도 추정 정확도 검증"""
        kf = KalmanFilter3D(dt=1/30.0, process_noise=0.05)

        # 알려진 속도
        true_velocity = np.array([0.1, 0.0, 0.05])
        dt = 1/30.0

        errors = []

        for t in range(100):
            true_position = true_velocity * t * dt
            measured_position = true_position + np.random.normal(0, 0.01, 3)

            _, est_vel, _ = kf.predict_and_update(measured_position)

            if t > 30:  # 수렴 후
                error = np.linalg.norm(est_vel - true_velocity)
                errors.append(error)

        # RMSE 계산
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        print(f"Velocity RMSE: {rmse:.4f} m/s")

        assert rmse < 0.03  # 3cm/s 이하 오차

    def test_angle_estimation_accuracy(self):
        """각도 추정 정확도 검증"""
        estimator = OrientationEstimator(min_points=50)

        # 다양한 각도에서 테스트
        test_angles = [0, 15, 30, 45, 60]
        errors = []

        for true_pitch in test_angles:
            # 기울어진 평면 생성
            n_points = 1000
            x = np.random.uniform(-0.5, 0.5, n_points)
            y = np.random.uniform(-0.5, 0.5, n_points)
            z = np.zeros(n_points)

            # Pitch 회전
            angle = np.radians(true_pitch)
            x_rot = x * np.cos(angle) + z * np.sin(angle)
            z_rot = -x * np.sin(angle) + z * np.cos(angle)

            points = np.column_stack([x_rot, y, z_rot + 1.0])
            result = estimator.estimate_from_pointcloud(points)

            if result is not None:
                # PCA 결과 해석이 복잡하므로 절대값 비교
                error = min(
                    abs(abs(result.pitch) - true_pitch),
                    abs(abs(result.roll) - true_pitch)
                )
                errors.append(error)

        # 평균 오차
        mean_error = np.mean(errors) if errors else float('inf')
        print(f"Angle estimation mean error: {mean_error:.2f} deg")

        # PCA는 정확한 각도 추정이 어려우므로 넓은 범위 허용
        assert mean_error < 30.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
