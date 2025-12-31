"""
Kalman Filter 테스트
"""

import pytest
import numpy as np

from nimg_v2.tracking.kalman_filter_3d import KalmanFilter3D, KalmanFilter3DWithIMU


class TestKalmanFilter3D:
    """KalmanFilter3D 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        kf = KalmanFilter3D(dt=1/30.0)

        assert not kf.initialized
        assert kf.dim_x == 9
        assert kf.dim_z == 3

    def test_initialize_with_position(self):
        """위치로 초기화 테스트"""
        kf = KalmanFilter3D()
        position = np.array([1.0, 2.0, 3.0])

        kf.initialize(position)

        assert kf.initialized
        np.testing.assert_array_equal(kf.get_position(), position)

    def test_predict_and_update(self):
        """예측 및 업데이트 테스트"""
        kf = KalmanFilter3D(dt=1/30.0)

        # 초기 위치
        initial_pos = np.array([0.0, 0.0, 1.0])
        kf.initialize(initial_pos)

        # 새 측정값
        new_pos = np.array([0.01, 0.0, 1.0])
        est_pos, est_vel, est_acc = kf.predict_and_update(new_pos)

        assert est_pos is not None
        assert est_vel is not None
        assert est_acc is not None
        assert len(est_pos) == 3
        assert len(est_vel) == 3

    def test_velocity_estimation(self):
        """속도 추정 정확도 테스트"""
        kf = KalmanFilter3D(dt=1/30.0, process_noise=0.01)
        dt = 1/30.0

        # 일정 속도로 이동
        true_velocity = np.array([0.1, 0.0, 0.0])  # 0.1 m/s in X direction

        estimated_velocities = []

        for t in range(100):
            true_position = true_velocity * t * dt
            # 약간의 노이즈 추가
            measured_position = true_position + np.random.normal(0, 0.005, 3)

            _, est_vel, _ = kf.predict_and_update(measured_position)
            estimated_velocities.append(est_vel.copy())

        # 마지막 30개 프레임의 평균 속도
        last_velocities = np.array(estimated_velocities[-30:])
        mean_velocity = np.mean(last_velocities, axis=0)

        # X 속도가 true_velocity에 근접해야 함
        assert abs(mean_velocity[0] - true_velocity[0]) < 0.02

    def test_adaptive_measurement_noise(self):
        """적응형 측정 노이즈 테스트"""
        kf = KalmanFilter3D()
        kf.initialize(np.array([0, 0, 1.0]))

        # 가까운 거리
        kf.update_measurement_noise(1.0)
        R_close = kf.R.copy()

        # 먼 거리
        kf.update_measurement_noise(5.0)
        R_far = kf.R.copy()

        # 먼 거리에서 노이즈가 더 커야 함
        assert R_far[0, 0] > R_close[0, 0]

    def test_get_state(self):
        """상태 조회 테스트"""
        kf = KalmanFilter3D()
        kf.initialize(np.array([1.0, 2.0, 3.0]))

        state = kf.get_state()

        assert 'position' in dir(state)
        assert 'velocity' in dir(state)
        assert 'acceleration' in dir(state)

    def test_predict_future(self):
        """미래 위치 예측 테스트"""
        kf = KalmanFilter3D(dt=1/30.0)
        kf.initialize(np.array([0.0, 0.0, 1.0]))

        # 몇 프레임 업데이트
        for t in range(10):
            pos = np.array([t * 0.01, 0.0, 1.0])
            kf.predict_and_update(pos)

        # 미래 예측
        future_pos = kf.predict_future(10)

        assert len(future_pos) == 3
        # 현재 위치보다 앞에 있어야 함
        current_pos = kf.get_position()
        # X 방향으로 이동 중이므로 미래 X가 더 커야 함
        assert future_pos[0] >= current_pos[0]

    def test_reset(self):
        """리셋 테스트"""
        kf = KalmanFilter3D()
        kf.initialize(np.array([1.0, 2.0, 3.0]))

        kf.reset()

        assert not kf.initialized
        assert kf.frame_count == 0


class TestKalmanFilter3DWithIMU:
    """IMU를 사용하는 Kalman Filter 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        kf = KalmanFilter3DWithIMU(dt=1/30.0)

        assert not kf.initialized

    def test_update_with_imu(self):
        """IMU 데이터로 업데이트 테스트"""
        kf = KalmanFilter3DWithIMU(dt=1/30.0)

        position = np.array([0.0, 0.0, 1.0])
        imu_accel = np.array([0.0, -9.81, 0.0])  # 중력만

        est_pos, est_vel, est_acc = kf.update_with_imu(position, imu_accel)

        assert est_pos is not None
        assert kf.initialized


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
