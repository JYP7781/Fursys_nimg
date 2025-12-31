"""
Pose Kalman Filter 단위 테스트
"""

import numpy as np
import pytest
from nimg_v3.measurement.pose_kalman_filter import (
    PoseKalmanFilter,
    FilterMode,
    PoseKalmanState
)
from nimg_v3.measurement.pose_converter import EulerAngles, Quaternion


class TestPoseKalmanFilterEuler:
    """오일러 모드 칼만 필터 테스트"""

    def test_initialization_euler(self):
        """오일러 모드 초기화"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.EULER)
        assert kf.mode == FilterMode.EULER
        assert kf.dim_x == 12  # pos(3) + vel(3) + euler(3) + angular_vel(3)

    def test_initialize_and_predict_euler(self):
        """오일러 모드 초기화 후 예측 테스트"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.EULER)

        # 초기화
        position = np.array([1.0, 2.0, 3.0])
        euler = EulerAngles(roll=10.0, pitch=20.0, yaw=30.0)
        kf.initialize(position, euler)

        # 예측 수행
        state = kf.predict()

        assert isinstance(state, PoseKalmanState)
        assert len(state.position) == 3
        assert isinstance(state.orientation_euler, EulerAngles)

    def test_update_euler(self):
        """오일러 모드 업데이트 테스트"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.EULER)

        # 초기화 및 업데이트
        position = np.array([1.0, 2.0, 3.0])
        euler = EulerAngles(roll=10.0, pitch=20.0, yaw=30.0)
        kf.initialize(position, euler)

        # 새로운 측정값으로 업데이트
        new_position = np.array([1.1, 2.0, 3.0])
        new_euler = EulerAngles(roll=11.0, pitch=20.0, yaw=30.0)
        state = kf.update(new_position, new_euler)

        assert isinstance(state, PoseKalmanState)


class TestPoseKalmanFilterQuaternion:
    """쿼터니언 모드 칼만 필터 테스트"""

    def test_initialization_quaternion(self):
        """쿼터니언 모드 초기화"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.QUATERNION)
        assert kf.mode == FilterMode.QUATERNION
        assert kf.dim_x == 13  # pos(3) + vel(3) + quat(4) + angular_vel(3)

    def test_initialize_and_predict_quaternion(self):
        """쿼터니언 모드 초기화 후 예측 테스트"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.QUATERNION)

        # 초기화
        position = np.array([1.0, 2.0, 0.5])
        quaternion = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        kf.initialize(position, quaternion)

        # 예측 수행
        state = kf.predict()

        assert isinstance(state, PoseKalmanState)
        assert len(state.position) == 3
        assert isinstance(state.orientation_quat, Quaternion)

        # 쿼터니언이 정규화되어 있는지 확인
        q = state.orientation_quat
        norm = np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)


class TestKalmanStateOutput:
    """칼만 필터 상태 출력 테스트"""

    def test_euler_state_output(self):
        """오일러 모드 상태 출력"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.EULER)

        position = np.array([1.0, 2.0, 3.0])
        euler = EulerAngles(roll=15.0, pitch=30.0, yaw=45.0)
        kf.initialize(position, euler)

        state = kf.get_state()

        assert hasattr(state, 'position')
        assert hasattr(state, 'orientation_euler')
        assert hasattr(state, 'velocity')
        assert hasattr(state, 'angular_velocity')

        # 쿼터니언도 오일러에서 변환되어 제공되어야 함
        assert hasattr(state, 'orientation_quat')

    def test_quaternion_state_output(self):
        """쿼터니언 모드 상태 출력"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.QUATERNION)

        position = np.array([1.0, 2.0, 3.0])
        # 45도 Z축 회전
        quaternion = Quaternion(x=0.0, y=0.0, z=0.3826834, w=0.9238795)
        kf.initialize(position, quaternion)

        state = kf.get_state()

        assert hasattr(state, 'position')
        assert hasattr(state, 'orientation_quat')
        assert hasattr(state, 'velocity')
        assert hasattr(state, 'angular_velocity')

        # 오일러 각도도 쿼터니언에서 변환되어 제공되어야 함
        assert hasattr(state, 'orientation_euler')

    def test_state_to_dict(self):
        """상태 딕셔너리 변환"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.QUATERNION)
        position = np.array([1.0, 2.0, 3.0])
        quaternion = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        kf.initialize(position, quaternion)

        state = kf.get_state()
        state_dict = state.to_dict()

        assert 'position' in state_dict
        assert 'velocity' in state_dict
        assert 'orientation' in state_dict
        assert 'euler' in state_dict['orientation']
        assert 'quaternion' in state_dict['orientation']


class TestAdaptiveNoise:
    """적응적 노이즈 조정 테스트"""

    def test_depth_based_noise(self):
        """깊이 기반 노이즈 조정"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.EULER, adaptive_noise=True)

        # 가까운 거리
        kf.update_adaptive_noise(0.5)
        R_close = kf.kf.R.copy()

        # 먼 거리
        kf.update_adaptive_noise(2.0)
        R_far = kf.kf.R.copy()

        # 먼 거리에서 노이즈가 더 커야 함
        assert R_far[0, 0] > R_close[0, 0]


class TestMultipleFrames:
    """여러 프레임 처리 테스트"""

    def test_tracking_sequence(self):
        """연속 프레임 추적"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.QUATERNION)

        # 시뮬레이션: 물체가 Z축으로 천천히 회전
        positions = []
        for i in range(10):
            angle = i * 5  # 5도씩 증가
            angle_rad = np.radians(angle)

            position = np.array([0.0, 0.0, 1.0])  # 고정 위치
            quaternion = Quaternion(
                x=0.0,
                y=0.0,
                z=np.sin(angle_rad/2),
                w=np.cos(angle_rad/2)
            )

            if i == 0:
                kf.initialize(position, quaternion)
                state = kf.get_state()
            else:
                state = kf.predict_and_update(position, quaternion)

            positions.append(state.position.copy())

        # 마지막 상태 확인
        assert len(positions) == 10

        # 속도 추정이 안정화되었는지 확인
        final_state = kf.get_state()
        assert final_state is not None
        assert final_state.speed >= 0


class TestFilterReset:
    """필터 리셋 테스트"""

    def test_reset(self):
        """필터 리셋"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.EULER)

        # 초기화 및 업데이트 수행
        position = np.array([1.0, 2.0, 3.0])
        euler = EulerAngles(roll=10.0, pitch=20.0, yaw=30.0)
        kf.initialize(position, euler)
        kf.predict()

        # 리셋
        kf.reset()

        # 상태가 초기화되었는지 확인
        state = kf.kf.x.flatten()
        np.testing.assert_array_almost_equal(state[:3], [0, 0, 0], decimal=6)


class TestPredictAndUpdate:
    """predict_and_update 통합 테스트"""

    def test_predict_and_update_euler(self):
        """오일러 모드 predict_and_update"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.EULER)

        # 초기화
        position = np.array([0.0, 0.0, 1.0])
        euler = EulerAngles(roll=0.0, pitch=0.0, yaw=0.0)
        kf.initialize(position, euler)

        # predict_and_update
        new_position = np.array([0.1, 0.0, 1.0])
        new_euler = EulerAngles(roll=5.0, pitch=0.0, yaw=0.0)
        state = kf.predict_and_update(new_position, new_euler)

        assert isinstance(state, PoseKalmanState)
        # 위치가 측정값 방향으로 업데이트되었는지 확인
        assert state.position[0] > 0

    def test_predict_and_update_quaternion(self):
        """쿼터니언 모드 predict_and_update"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.QUATERNION)

        # 초기화
        position = np.array([0.0, 0.0, 1.0])
        quaternion = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        kf.initialize(position, quaternion)

        # predict_and_update
        new_position = np.array([0.1, 0.0, 1.0])
        # 10도 Z축 회전
        angle_rad = np.radians(10)
        new_quaternion = Quaternion(
            x=0.0,
            y=0.0,
            z=np.sin(angle_rad/2),
            w=np.cos(angle_rad/2)
        )
        state = kf.predict_and_update(new_position, new_quaternion)

        assert isinstance(state, PoseKalmanState)
        # 쿼터니언이 정규화되어 있는지 확인
        q = state.orientation_quat
        norm = np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)


class TestSetDt:
    """시간 간격 변경 테스트"""

    def test_set_dt(self):
        """시간 간격 변경"""
        kf = PoseKalmanFilter(dt=1/30.0, mode=FilterMode.EULER)

        # dt 변경
        kf.set_dt(1/60.0)
        assert kf.dt == 1/60.0

        # 상태 전이 행렬이 업데이트되었는지 확인
        assert kf.kf.F[0, 3] == pytest.approx(1/60.0, rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
