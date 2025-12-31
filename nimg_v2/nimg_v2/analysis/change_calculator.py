"""
변화량 계산 모듈
기준 프레임 대비 상대적 속도/각도 변화량 계산
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field
import logging

from ..estimation.orientation_estimator import OrientationResult

logger = logging.getLogger(__name__)


@dataclass
class ChangeResult:
    """기준 대비 변화량 결과"""
    # 프레임 정보
    frame_idx: int
    timestamp: float

    # 위치 변화 (미터)
    position_change: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [dx, dy, dz]
    distance_from_reference: float = 0.0  # 기준점으로부터의 거리

    # 속도 (m/s)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [vx, vy, vz]
    speed: float = 0.0  # 속력

    # 각도 변화 (degrees)
    roll_change: float = 0.0
    pitch_change: float = 0.0
    yaw_change: float = 0.0
    total_rotation: float = 0.0  # 총 회전량

    # 현재 절대값 (degrees)
    roll_current: float = 0.0
    pitch_current: float = 0.0
    yaw_current: float = 0.0

    # 신뢰도 (0-1)
    position_confidence: float = 0.0
    orientation_confidence: float = 0.0
    overall_confidence: float = 0.0

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            'frame_idx': self.frame_idx,
            'timestamp': self.timestamp,
            'dx': self.position_change[0],
            'dy': self.position_change[1],
            'dz': self.position_change[2],
            'distance_from_reference': self.distance_from_reference,
            'vx': self.velocity[0],
            'vy': self.velocity[1],
            'vz': self.velocity[2],
            'speed': self.speed,
            'roll_change': self.roll_change,
            'pitch_change': self.pitch_change,
            'yaw_change': self.yaw_change,
            'total_rotation': self.total_rotation,
            'roll_current': self.roll_current,
            'pitch_current': self.pitch_current,
            'yaw_current': self.yaw_current,
            'position_confidence': self.position_confidence,
            'orientation_confidence': self.orientation_confidence,
            'overall_confidence': self.overall_confidence
        }


class ChangeCalculator:
    """
    기준 프레임 대비 변화량 계산기

    기준 프레임의 위치와 방향을 저장하고,
    이후 프레임과의 차이를 계산합니다.
    """

    def __init__(self):
        self.reference_position: Optional[np.ndarray] = None
        self.reference_orientation: Optional[OrientationResult] = None
        self.reference_frame_idx: int = -1
        self.reference_set: bool = False

        logger.debug("ChangeCalculator initialized")

    def set_reference(
        self,
        position: np.ndarray,
        orientation: OrientationResult,
        frame_idx: int = 0
    ):
        """
        기준 프레임 설정

        Args:
            position: 기준 위치 [x, y, z]
            orientation: 기준 방향
            frame_idx: 기준 프레임 인덱스
        """
        self.reference_position = position.copy()
        self.reference_orientation = orientation
        self.reference_frame_idx = frame_idx
        self.reference_set = True

        logger.info(f"Reference set: frame={frame_idx}, "
                   f"pos=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}), "
                   f"orient=(R:{orientation.roll:.1f}, P:{orientation.pitch:.1f}, Y:{orientation.yaw:.1f})")

    def calculate_change(
        self,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        current_orientation: OrientationResult,
        frame_idx: int,
        timestamp: float,
        position_confidence: float = 1.0
    ) -> ChangeResult:
        """
        현재 프레임과 기준 프레임 간의 변화량 계산

        Args:
            current_position: 현재 위치 [x, y, z]
            current_velocity: 현재 속도 [vx, vy, vz]
            current_orientation: 현재 방향
            frame_idx: 현재 프레임 인덱스
            timestamp: 현재 타임스탬프
            position_confidence: 위치 신뢰도

        Returns:
            ChangeResult
        """
        if not self.reference_set:
            raise ValueError("Reference frame not set. Call set_reference() first.")

        # 위치 변화
        position_change = current_position - self.reference_position
        distance_from_reference = np.linalg.norm(position_change)

        # 속도
        speed = np.linalg.norm(current_velocity)

        # 각도 변화
        roll_change = self._normalize_angle(
            current_orientation.roll - self.reference_orientation.roll
        )
        pitch_change = self._normalize_angle(
            current_orientation.pitch - self.reference_orientation.pitch
        )
        yaw_change = self._normalize_angle(
            current_orientation.yaw - self.reference_orientation.yaw
        )

        # 총 회전량
        total_rotation = np.sqrt(roll_change**2 + pitch_change**2 + yaw_change**2)

        # 전체 신뢰도
        overall_confidence = np.sqrt(
            position_confidence * current_orientation.confidence
        )

        return ChangeResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            position_change=position_change,
            distance_from_reference=distance_from_reference,
            velocity=current_velocity.copy(),
            speed=speed,
            roll_change=roll_change,
            pitch_change=pitch_change,
            yaw_change=yaw_change,
            total_rotation=total_rotation,
            roll_current=current_orientation.roll,
            pitch_current=current_orientation.pitch,
            yaw_current=current_orientation.yaw,
            position_confidence=position_confidence,
            orientation_confidence=current_orientation.confidence,
            overall_confidence=overall_confidence
        )

    def _normalize_angle(self, angle: float) -> float:
        """각도를 -180 ~ 180 범위로 정규화"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def get_reference_info(self) -> dict:
        """기준 프레임 정보 반환"""
        if not self.reference_set:
            return {'set': False}

        return {
            'set': True,
            'frame_idx': self.reference_frame_idx,
            'position': self.reference_position.tolist(),
            'orientation': {
                'roll': self.reference_orientation.roll,
                'pitch': self.reference_orientation.pitch,
                'yaw': self.reference_orientation.yaw
            }
        }

    def reset(self):
        """기준 프레임 리셋"""
        self.reference_position = None
        self.reference_orientation = None
        self.reference_frame_idx = -1
        self.reference_set = False
        logger.info("ChangeCalculator reset")


class ChangeStatistics:
    """
    변화량 통계 계산기
    """

    def __init__(self):
        self.results: list = []

    def add_result(self, result: ChangeResult):
        """결과 추가"""
        self.results.append(result)

    def add_results(self, results: list):
        """여러 결과 추가"""
        self.results.extend(results)

    def clear(self):
        """결과 초기화"""
        self.results = []

    def get_statistics(self) -> dict:
        """통계 계산"""
        if len(self.results) == 0:
            return {'count': 0}

        speeds = [r.speed for r in self.results]
        roll_changes = [r.roll_change for r in self.results]
        pitch_changes = [r.pitch_change for r in self.results]
        yaw_changes = [r.yaw_change for r in self.results]
        distances = [r.distance_from_reference for r in self.results]
        confidences = [r.overall_confidence for r in self.results]

        return {
            'count': len(self.results),
            'speed': {
                'mean': float(np.mean(speeds)),
                'std': float(np.std(speeds)),
                'min': float(np.min(speeds)),
                'max': float(np.max(speeds)),
                'median': float(np.median(speeds))
            },
            'roll_change': {
                'mean': float(np.mean(roll_changes)),
                'std': float(np.std(roll_changes)),
                'min': float(np.min(roll_changes)),
                'max': float(np.max(roll_changes))
            },
            'pitch_change': {
                'mean': float(np.mean(pitch_changes)),
                'std': float(np.std(pitch_changes)),
                'min': float(np.min(pitch_changes)),
                'max': float(np.max(pitch_changes))
            },
            'yaw_change': {
                'mean': float(np.mean(yaw_changes)),
                'std': float(np.std(yaw_changes)),
                'min': float(np.min(yaw_changes)),
                'max': float(np.max(yaw_changes))
            },
            'distance_from_reference': {
                'mean': float(np.mean(distances)),
                'max': float(np.max(distances))
            },
            'confidence': {
                'mean': float(np.mean(confidences)),
                'min': float(np.min(confidences))
            }
        }

    def get_summary_string(self) -> str:
        """통계 요약 문자열"""
        stats = self.get_statistics()

        if stats['count'] == 0:
            return "No results to summarize"

        return (
            f"=== Change Statistics ===\n"
            f"Total frames: {stats['count']}\n"
            f"\n"
            f"Speed (m/s):\n"
            f"  Mean: {stats['speed']['mean']:.4f}\n"
            f"  Max:  {stats['speed']['max']:.4f}\n"
            f"  Std:  {stats['speed']['std']:.4f}\n"
            f"\n"
            f"Roll change (deg):\n"
            f"  Range: {stats['roll_change']['min']:.2f} ~ {stats['roll_change']['max']:.2f}\n"
            f"  Std:   {stats['roll_change']['std']:.2f}\n"
            f"\n"
            f"Pitch change (deg):\n"
            f"  Range: {stats['pitch_change']['min']:.2f} ~ {stats['pitch_change']['max']:.2f}\n"
            f"  Std:   {stats['pitch_change']['std']:.2f}\n"
            f"\n"
            f"Yaw change (deg):\n"
            f"  Range: {stats['yaw_change']['min']:.2f} ~ {stats['yaw_change']['max']:.2f}\n"
            f"  Std:   {stats['yaw_change']['std']:.2f}\n"
            f"\n"
            f"Average confidence: {stats['confidence']['mean']:.3f}"
        )
