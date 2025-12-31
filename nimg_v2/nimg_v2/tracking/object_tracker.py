"""
객체 추적 모듈
다중 객체 추적 및 ID 할당
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from .kalman_filter_3d import KalmanFilter3D
from ..detection.yolo_detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """추적 중인 객체"""
    track_id: int
    class_id: int
    class_name: str
    kalman_filter: KalmanFilter3D
    last_detection: Optional[Detection] = None
    age: int = 0  # 추적 프레임 수
    hits: int = 0  # 연속 탐지 수
    misses: int = 0  # 연속 미탐지 수
    confidence: float = 0.0

    # 상태 정보
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    speed: float = 0.0

    def update_state(self):
        """Kalman Filter에서 상태 업데이트"""
        state = self.kalman_filter.get_state()
        self.position = state.position
        self.velocity = state.velocity
        self.speed = np.linalg.norm(state.velocity)


class ObjectTracker:
    """
    다중 객체 추적기

    각 객체에 대해 독립적인 Kalman Filter를 유지하고,
    탐지 결과와 기존 트랙을 매칭합니다.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        distance_threshold: float = 0.5,
        dt: float = 1/30.0
    ):
        """
        Args:
            max_age: 미탐지 후 트랙 삭제까지 최대 프레임 수
            min_hits: 트랙 확정까지 필요한 최소 연속 탐지 수
            iou_threshold: 매칭용 최소 IoU
            distance_threshold: 3D 매칭용 최대 거리 (미터)
            dt: 시간 간격
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.dt = dt

        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 1
        self.frame_count = 0

        logger.info(f"ObjectTracker initialized: max_age={max_age}, min_hits={min_hits}")

    def update(
        self,
        detections: List[Detection],
        positions_3d: Optional[List[np.ndarray]] = None
    ) -> List[TrackedObject]:
        """
        추적 업데이트

        Args:
            detections: 현재 프레임의 탐지 결과 리스트
            positions_3d: 각 탐지에 대응하는 3D 위치 (옵션)

        Returns:
            확정된 트랙 리스트
        """
        self.frame_count += 1

        # 모든 트랙 예측
        for track in self.tracks.values():
            track.kalman_filter.predict()
            track.age += 1

        if len(detections) == 0:
            # 탐지가 없으면 모든 트랙 miss 처리
            self._handle_unmatched_tracks(list(self.tracks.keys()))
            return self._get_confirmed_tracks()

        # 매칭
        if positions_3d is not None and len(positions_3d) == len(detections):
            matches, unmatched_dets, unmatched_tracks = self._match_with_3d(
                detections, positions_3d
            )
        else:
            matches, unmatched_dets, unmatched_tracks = self._match_with_iou(detections)

        # 매칭된 트랙 업데이트
        for det_idx, track_id in matches:
            det = detections[det_idx]
            pos = positions_3d[det_idx] if positions_3d else np.zeros(3)

            track = self.tracks[track_id]
            track.kalman_filter.update(pos)
            track.last_detection = det
            track.hits += 1
            track.misses = 0
            track.confidence = det.confidence
            track.update_state()

        # 미매칭 탐지 → 새 트랙 생성
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            pos = positions_3d[det_idx] if positions_3d else np.zeros(3)
            self._create_track(det, pos)

        # 미매칭 트랙 처리
        self._handle_unmatched_tracks(unmatched_tracks)

        # 확정된 트랙 반환
        return self._get_confirmed_tracks()

    def _match_with_iou(
        self,
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """IoU 기반 매칭"""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        # IoU 행렬 계산
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(detections), len(track_ids)))

        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                if track.last_detection is not None:
                    iou_matrix[i, j] = self._compute_iou(det, track.last_detection)

        # 헝가리안 알고리즘으로 최적 매칭
        matches, unmatched_dets, unmatched_tracks = self._hungarian_matching(
            iou_matrix, track_ids, self.iou_threshold
        )

        return matches, unmatched_dets, unmatched_tracks

    def _match_with_3d(
        self,
        detections: List[Detection],
        positions_3d: List[np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """3D 거리 기반 매칭"""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        # 거리 행렬 계산
        track_ids = list(self.tracks.keys())
        dist_matrix = np.zeros((len(detections), len(track_ids)))

        for i, pos in enumerate(positions_3d):
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                pred_pos = track.kalman_filter.get_position()
                dist_matrix[i, j] = np.linalg.norm(pos - pred_pos)

        # 거리를 유사도로 변환 (작을수록 좋음)
        # max_dist를 넘으면 매칭 불가
        similarity_matrix = 1.0 - np.minimum(dist_matrix / self.distance_threshold, 1.0)

        matches, unmatched_dets, unmatched_tracks = self._hungarian_matching(
            similarity_matrix, track_ids, 0.5
        )

        return matches, unmatched_dets, unmatched_tracks

    def _hungarian_matching(
        self,
        cost_matrix: np.ndarray,
        track_ids: List[int],
        threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """헝가리안 알고리즘을 사용한 최적 매칭"""
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(-cost_matrix)

            matches = []
            unmatched_dets = list(range(cost_matrix.shape[0]))
            unmatched_tracks = list(range(len(track_ids)))

            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] >= threshold:
                    matches.append((row, track_ids[col]))
                    unmatched_dets.remove(row)
                    unmatched_tracks.remove(col)

            unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]
            return matches, unmatched_dets, unmatched_track_ids

        except ImportError:
            # scipy 없으면 그리디 매칭
            return self._greedy_matching(cost_matrix, track_ids, threshold)

    def _greedy_matching(
        self,
        cost_matrix: np.ndarray,
        track_ids: List[int],
        threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """그리디 매칭 (scipy 없을 때 대안)"""
        matches = []
        unmatched_dets = list(range(cost_matrix.shape[0]))
        unmatched_tracks = list(range(len(track_ids)))

        while len(unmatched_dets) > 0 and len(unmatched_tracks) > 0:
            # 최대 유사도 찾기
            max_val = -1
            max_det, max_track = -1, -1

            for det_idx in unmatched_dets:
                for track_idx in unmatched_tracks:
                    if cost_matrix[det_idx, track_idx] > max_val:
                        max_val = cost_matrix[det_idx, track_idx]
                        max_det, max_track = det_idx, track_idx

            if max_val >= threshold:
                matches.append((max_det, track_ids[max_track]))
                unmatched_dets.remove(max_det)
                unmatched_tracks.remove(max_track)
            else:
                break

        unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]
        return matches, unmatched_dets, unmatched_track_ids

    def _compute_iou(self, det1: Detection, det2: Detection) -> float:
        """두 Detection 간 IoU 계산"""
        x1 = max(det1.x, det2.x)
        y1 = max(det1.y, det2.y)
        x2 = min(det1.x2, det2.x2)
        y2 = min(det1.y2, det2.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = det1.area
        area2 = det2.area
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _create_track(self, detection: Detection, position_3d: np.ndarray):
        """새 트랙 생성"""
        track_id = self.next_track_id
        self.next_track_id += 1

        kf = KalmanFilter3D(dt=self.dt)
        kf.initialize(position_3d)

        track = TrackedObject(
            track_id=track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            kalman_filter=kf,
            last_detection=detection,
            hits=1,
            confidence=detection.confidence,
            position=position_3d.copy()
        )

        self.tracks[track_id] = track
        logger.debug(f"Created new track: ID={track_id}, class={detection.class_name}")

    def _handle_unmatched_tracks(self, track_ids: List[int]):
        """미매칭 트랙 처리"""
        tracks_to_remove = []

        for track_id in track_ids:
            if track_id not in self.tracks:
                continue

            track = self.tracks[track_id]
            track.misses += 1
            track.hits = 0

            if track.misses > self.max_age:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            logger.debug(f"Removed track: ID={track_id}")

    def _get_confirmed_tracks(self) -> List[TrackedObject]:
        """확정된 트랙 반환 (min_hits 이상 연속 탐지된 트랙)"""
        confirmed = []

        for track in self.tracks.values():
            # 충분히 연속 탐지되었거나 오래된 트랙
            if track.hits >= self.min_hits or track.age >= self.min_hits:
                track.update_state()
                confirmed.append(track)

        return confirmed

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """특정 트랙 반환"""
        return self.tracks.get(track_id)

    def get_all_tracks(self) -> List[TrackedObject]:
        """모든 트랙 반환"""
        return list(self.tracks.values())

    def get_best_track(self, by: str = 'confidence') -> Optional[TrackedObject]:
        """
        최상의 트랙 반환

        Args:
            by: 정렬 기준 ('confidence', 'hits', 'distance')
        """
        confirmed = self._get_confirmed_tracks()
        if not confirmed:
            return None

        if by == 'confidence':
            return max(confirmed, key=lambda t: t.confidence)
        elif by == 'hits':
            return max(confirmed, key=lambda t: t.hits)
        elif by == 'distance':
            return min(confirmed, key=lambda t: np.linalg.norm(t.position))
        else:
            return confirmed[0]

    def reset(self):
        """추적기 리셋"""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0
        logger.info("ObjectTracker reset")
