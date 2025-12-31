#!/usr/bin/env python3
"""
test_pose_converter.py - PoseConverter 단위 테스트

euler_vs_quaternion_rotation_analysis.md에서 분석된
다양한 변환 시나리오를 테스트합니다.

Author: FurSys AI Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pytest

from nimg_v3.measurement.pose_converter import (
    PoseConverter,
    EulerAngles,
    Quaternion,
    RotationOrder,
    pose_to_euler,
    pose_to_quaternion,
    compute_angle_change,
    rotation_matrix_to_6d,
    sixd_to_rotation_matrix
)


class TestEulerAngles:
    """EulerAngles 클래스 테스트"""

    def test_creation(self):
        euler = EulerAngles(roll=30, pitch=45, yaw=60)
        assert euler.roll == 30
        assert euler.pitch == 45
        assert euler.yaw == 60

    def test_to_array(self):
        euler = EulerAngles(roll=10, pitch=20, yaw=30)
        arr = euler.to_array()
        np.testing.assert_array_equal(arr, [10, 20, 30])

    def test_normalize(self):
        euler = EulerAngles(roll=190, pitch=-190, yaw=370)
        normalized = euler.normalize()
        assert -180 <= normalized.roll <= 180
        assert -180 <= normalized.pitch <= 180
        assert -180 <= normalized.yaw <= 180

    def test_subtraction(self):
        e1 = EulerAngles(roll=30, pitch=45, yaw=60)
        e2 = EulerAngles(roll=10, pitch=20, yaw=30)
        diff = e1 - e2
        assert diff.roll == 20
        assert diff.pitch == 25
        assert diff.yaw == 30


class TestQuaternion:
    """Quaternion 클래스 테스트"""

    def test_identity(self):
        q = Quaternion.identity()
        assert q.x == 0
        assert q.y == 0
        assert q.z == 0
        assert q.w == 1

    def test_normalize(self):
        q = Quaternion(x=1, y=1, z=1, w=1)
        q_norm = q.normalize()
        assert abs(q_norm.norm - 1.0) < 1e-6

    def test_conjugate(self):
        q = Quaternion(x=0.1, y=0.2, z=0.3, w=0.9)
        q_conj = q.conjugate()
        assert q_conj.x == -q.x
        assert q_conj.y == -q.y
        assert q_conj.z == -q.z
        assert q_conj.w == q.w

    def test_from_axis_angle(self):
        # Z축 90도 회전
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), 90)
        q = q.normalize()
        assert abs(q.norm - 1.0) < 1e-6

    def test_multiplication(self):
        q1 = Quaternion.identity()
        q2 = Quaternion.from_axis_angle(np.array([0, 0, 1]), 45)
        q3 = q1 * q2
        assert abs(q3.norm - 1.0) < 1e-6


class TestPoseConverter:
    """PoseConverter 클래스 테스트"""

    def test_identity_pose(self):
        converter = PoseConverter()
        pose = np.eye(4)

        components = converter.pose_matrix_to_components(pose)

        np.testing.assert_array_almost_equal(components.translation, [0, 0, 0])
        assert abs(components.euler.roll) < 1e-6
        assert abs(components.euler.pitch) < 1e-6
        assert abs(components.euler.yaw) < 1e-6

    def test_translation_only(self):
        converter = PoseConverter()
        pose = np.eye(4)
        pose[:3, 3] = [1.0, 2.0, 3.0]

        translation, euler = converter.pose_to_euler(pose)

        np.testing.assert_array_almost_equal(translation, [1.0, 2.0, 3.0])
        assert abs(euler.roll) < 1e-6
        assert abs(euler.pitch) < 1e-6
        assert abs(euler.yaw) < 1e-6

    def test_rotation_z_90(self):
        """Z축 90도 회전 테스트"""
        from scipy.spatial.transform import Rotation

        converter = PoseConverter()

        # Z축 90도 회전
        rot = Rotation.from_euler('z', 90, degrees=True)
        pose = np.eye(4)
        pose[:3, :3] = rot.as_matrix()

        _, euler = converter.pose_to_euler(pose)

        # XYZ 순서에서 Z 회전은 yaw
        assert abs(euler.yaw - 90) < 1e-4 or abs(euler.yaw + 270) < 1e-4

    def test_euler_quaternion_roundtrip(self):
        """오일러 -> 쿼터니언 -> 오일러 왕복 변환 테스트"""
        converter = PoseConverter()

        original = EulerAngles(roll=30, pitch=45, yaw=60)
        quat = converter.euler_to_quaternion(original)
        recovered = converter.quaternion_to_euler(quat)

        assert abs(original.roll - recovered.roll) < 1e-4
        assert abs(original.pitch - recovered.pitch) < 1e-4
        assert abs(original.yaw - recovered.yaw) < 1e-4

    def test_quaternion_euler_roundtrip(self):
        """쿼터니언 -> 오일러 -> 쿼터니언 왕복 변환 테스트"""
        converter = PoseConverter()

        original = Quaternion.from_axis_angle(np.array([1, 1, 1]), 60).normalize()
        euler = converter.quaternion_to_euler(original)
        recovered = converter.euler_to_quaternion(euler).normalize()

        # 쿼터니언 비교 (q와 -q는 같은 회전)
        dot = abs(original.dot(recovered))
        assert dot > 0.9999

    def test_slerp(self):
        """SLERP 보간 테스트"""
        q0 = Quaternion.identity()
        q1 = Quaternion.from_axis_angle(np.array([0, 0, 1]), 90).normalize()

        # 중간 지점
        q_mid = PoseConverter.slerp(q0, q1, 0.5)
        assert abs(q_mid.norm - 1.0) < 1e-6

        # 시작과 끝
        q_start = PoseConverter.slerp(q0, q1, 0.0)
        q_end = PoseConverter.slerp(q0, q1, 1.0)

        assert abs(q_start.dot(q0)) > 0.9999
        assert abs(q_end.dot(q1)) > 0.9999


class TestAngleChange:
    """각도 변화 계산 테스트"""

    def test_no_change(self):
        pose1 = np.eye(4)
        pose2 = np.eye(4)

        change = compute_angle_change(pose1, pose2)

        np.testing.assert_array_almost_equal(change, [0, 0, 0])

    def test_yaw_change(self):
        from scipy.spatial.transform import Rotation

        pose1 = np.eye(4)

        rot = Rotation.from_euler('z', 30, degrees=True)
        pose2 = np.eye(4)
        pose2[:3, :3] = rot.as_matrix()

        change = compute_angle_change(pose1, pose2, use_quaternion=True)

        # Z축 회전은 yaw (인덱스 2)
        assert abs(change[2]) > 25  # 약 30도


class Test6DRepresentation:
    """6D 연속 표현 테스트"""

    def test_roundtrip(self):
        from scipy.spatial.transform import Rotation

        # 임의의 회전 행렬
        rot = Rotation.from_euler('xyz', [30, 45, 60], degrees=True)
        R_original = rot.as_matrix()

        # 6D로 변환 후 복원
        sixd = rotation_matrix_to_6d(R_original)
        R_recovered = sixd_to_rotation_matrix(sixd)

        np.testing.assert_array_almost_equal(R_original, R_recovered, decimal=5)


class TestGimbalLock:
    """짐벌 락 감지 테스트"""

    def test_gimbal_lock_warning(self):
        from scipy.spatial.transform import Rotation

        converter = PoseConverter(warn_gimbal_lock=True)

        # Pitch 89도 (짐벌 락 근처)
        rot = Rotation.from_euler('xyz', [0, 89, 0], degrees=True)
        pose = np.eye(4)
        pose[:3, :3] = rot.as_matrix()

        components = converter.pose_matrix_to_components(pose)

        assert components.gimbal_lock_warning is True

    def test_no_gimbal_lock(self):
        from scipy.spatial.transform import Rotation

        converter = PoseConverter(warn_gimbal_lock=True)

        # Pitch 45도 (안전)
        rot = Rotation.from_euler('xyz', [0, 45, 0], degrees=True)
        pose = np.eye(4)
        pose[:3, :3] = rot.as_matrix()

        components = converter.pose_matrix_to_components(pose)

        assert components.gimbal_lock_warning is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
