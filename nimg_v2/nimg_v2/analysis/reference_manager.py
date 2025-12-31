"""
기준 프레임 관리 모듈
기준 프레임 선택, 저장, 로드 기능
"""

import numpy as np
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

from ..estimation.orientation_estimator import OrientationResult

logger = logging.getLogger(__name__)


@dataclass
class ReferenceState:
    """기준 상태 저장"""
    frame_idx: int
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    orientation: OrientationResult
    class_name: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'frame_idx': self.frame_idx,
            'timestamp': self.timestamp,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'orientation': {
                'roll': self.orientation.roll,
                'pitch': self.orientation.pitch,
                'yaw': self.orientation.yaw,
                'center': self.orientation.center.tolist(),
                'confidence': self.orientation.confidence,
                'num_points': self.orientation.num_points
            },
            'class_name': self.class_name,
            'confidence': self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReferenceState':
        """딕셔너리에서 생성"""
        orientation = OrientationResult(
            roll=data['orientation']['roll'],
            pitch=data['orientation']['pitch'],
            yaw=data['orientation']['yaw'],
            center=np.array(data['orientation']['center']),
            axes=np.eye(3),  # 저장하지 않음
            eigenvalues=np.ones(3),  # 저장하지 않음
            confidence=data['orientation']['confidence'],
            num_points=data['orientation']['num_points']
        )

        return cls(
            frame_idx=data['frame_idx'],
            timestamp=data['timestamp'],
            position=np.array(data['position']),
            velocity=np.array(data['velocity']),
            orientation=orientation,
            class_name=data['class_name'],
            confidence=data['confidence']
        )


class ReferenceManager:
    """
    기준 프레임 관리자

    기준 프레임의 선택, 저장, 로드를 담당합니다.
    """

    def __init__(self):
        self.reference: Optional[ReferenceState] = None
        self.history: list = []  # 기준 프레임 변경 이력

        logger.debug("ReferenceManager initialized")

    def set_reference(
        self,
        frame_idx: int,
        timestamp: float,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation: OrientationResult,
        class_name: str = "",
        confidence: float = 1.0
    ):
        """
        기준 프레임 설정

        Args:
            frame_idx: 프레임 인덱스
            timestamp: 타임스탬프
            position: 3D 위치
            velocity: 3D 속도
            orientation: 방향
            class_name: 탐지된 클래스 이름
            confidence: 탐지 신뢰도
        """
        self.reference = ReferenceState(
            frame_idx=frame_idx,
            timestamp=timestamp,
            position=position.copy(),
            velocity=velocity.copy(),
            orientation=orientation,
            class_name=class_name,
            confidence=confidence
        )

        # 이력 저장
        self.history.append({
            'action': 'set',
            'frame_idx': frame_idx,
            'position': position.tolist()
        })

        logger.info(f"Reference set: frame {frame_idx}, class '{class_name}'")

    @property
    def is_set(self) -> bool:
        """기준 프레임 설정 여부"""
        return self.reference is not None

    @property
    def position(self) -> Optional[np.ndarray]:
        """기준 위치"""
        return self.reference.position if self.reference else None

    @property
    def orientation(self) -> Optional[OrientationResult]:
        """기준 방향"""
        return self.reference.orientation if self.reference else None

    def get_reference(self) -> Optional[ReferenceState]:
        """기준 상태 반환"""
        return self.reference

    def reset(self):
        """기준 프레임 리셋"""
        self.reference = None
        self.history.append({'action': 'reset'})
        logger.info("Reference reset")

    def save(self, file_path: str):
        """
        기준 상태 저장

        Args:
            file_path: 저장 파일 경로 (.json)
        """
        if self.reference is None:
            raise ValueError("No reference to save")

        data = {
            'reference': self.reference.to_dict(),
            'history': self.history
        }

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Reference saved to {file_path}")

    def load(self, file_path: str) -> ReferenceState:
        """
        기준 상태 로드

        Args:
            file_path: 로드 파일 경로 (.json)

        Returns:
            로드된 기준 상태
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.reference = ReferenceState.from_dict(data['reference'])
        self.history = data.get('history', [])
        self.history.append({'action': 'load', 'file': file_path})

        logger.info(f"Reference loaded from {file_path}")
        return self.reference

    def get_info(self) -> Dict[str, Any]:
        """기준 정보 반환"""
        if self.reference is None:
            return {'is_set': False}

        return {
            'is_set': True,
            'frame_idx': self.reference.frame_idx,
            'timestamp': self.reference.timestamp,
            'position': self.reference.position.tolist(),
            'orientation': {
                'roll': self.reference.orientation.roll,
                'pitch': self.reference.orientation.pitch,
                'yaw': self.reference.orientation.yaw
            },
            'class_name': self.reference.class_name,
            'confidence': self.reference.confidence,
            'history_length': len(self.history)
        }


class AutoReferenceSelector:
    """
    자동 기준 프레임 선택기

    다양한 기준으로 최적의 기준 프레임을 자동 선택합니다.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        min_valid_depth_ratio: float = 0.3,
        stability_frames: int = 5
    ):
        """
        Args:
            min_confidence: 최소 탐지 신뢰도
            min_valid_depth_ratio: 최소 유효 depth 비율
            stability_frames: 안정성 확인용 프레임 수
        """
        self.min_confidence = min_confidence
        self.min_valid_depth_ratio = min_valid_depth_ratio
        self.stability_frames = stability_frames

        self.candidates: list = []

    def add_candidate(
        self,
        frame_idx: int,
        confidence: float,
        depth_valid_ratio: float,
        position: np.ndarray,
        orientation_confidence: float
    ):
        """
        후보 프레임 추가

        Args:
            frame_idx: 프레임 인덱스
            confidence: 탐지 신뢰도
            depth_valid_ratio: 유효 depth 비율
            position: 3D 위치
            orientation_confidence: 방향 추정 신뢰도
        """
        # 최소 조건 확인
        if confidence < self.min_confidence:
            return
        if depth_valid_ratio < self.min_valid_depth_ratio:
            return

        # 종합 점수 계산
        score = self._compute_score(
            confidence, depth_valid_ratio, orientation_confidence
        )

        self.candidates.append({
            'frame_idx': frame_idx,
            'confidence': confidence,
            'depth_valid_ratio': depth_valid_ratio,
            'position': position.copy(),
            'orientation_confidence': orientation_confidence,
            'score': score
        })

    def _compute_score(
        self,
        confidence: float,
        depth_valid_ratio: float,
        orientation_confidence: float
    ) -> float:
        """종합 점수 계산"""
        # 가중치 적용
        score = (
            0.4 * confidence +
            0.3 * depth_valid_ratio +
            0.3 * orientation_confidence
        )
        return score

    def select_best(self) -> Optional[Dict]:
        """
        최상의 후보 선택

        Returns:
            최상의 후보 또는 None
        """
        if len(self.candidates) == 0:
            return None

        # 점수 기준 정렬
        sorted_candidates = sorted(
            self.candidates, key=lambda x: x['score'], reverse=True
        )

        return sorted_candidates[0]

    def select_with_stability(self) -> Optional[Dict]:
        """
        안정성을 고려한 선택

        연속 프레임에서 안정적인 후보 선택
        """
        if len(self.candidates) < self.stability_frames:
            return self.select_best()

        # 연속 프레임 그룹 찾기
        groups = self._find_stable_groups()

        if not groups:
            return self.select_best()

        # 가장 안정적인 그룹의 중간 프레임 선택
        best_group = max(groups, key=lambda g: len(g))
        mid_idx = len(best_group) // 2

        return best_group[mid_idx]

    def _find_stable_groups(self) -> list:
        """연속 프레임 그룹 찾기"""
        if len(self.candidates) == 0:
            return []

        # 프레임 인덱스로 정렬
        sorted_candidates = sorted(
            self.candidates, key=lambda x: x['frame_idx']
        )

        groups = []
        current_group = [sorted_candidates[0]]

        for i in range(1, len(sorted_candidates)):
            curr = sorted_candidates[i]
            prev = sorted_candidates[i - 1]

            # 연속 프레임이면 같은 그룹
            if curr['frame_idx'] - prev['frame_idx'] == 1:
                current_group.append(curr)
            else:
                if len(current_group) >= self.stability_frames:
                    groups.append(current_group)
                current_group = [curr]

        if len(current_group) >= self.stability_frames:
            groups.append(current_group)

        return groups

    def clear(self):
        """후보 초기화"""
        self.candidates = []
