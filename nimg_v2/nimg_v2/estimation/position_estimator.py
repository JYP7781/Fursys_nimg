"""
3D 위치 추정 모듈
2D BBox + Depth → 3D 위치 변환
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionResult:
    """위치 추정 결과"""
    position: np.ndarray  # [x, y, z] in meters
    confidence: float     # 신뢰도 (0-1)
    valid_depth_ratio: float  # 유효 depth 비율
    depth_std: float      # depth 표준편차


class PositionEstimator:
    """
    2D BBox + Depth → 3D 위치 추정기

    카메라 내부 파라미터를 사용하여 2D 픽셀 좌표를
    3D 월드 좌표로 변환합니다.
    """

    def __init__(
        self,
        intrinsics: Dict[str, float],
        depth_min: float = 0.1,
        depth_max: float = 10.0,
        margin: int = 5,
        min_valid_ratio: float = 0.3
    ):
        """
        Args:
            intrinsics: 카메라 내부 파라미터 {'fx', 'fy', 'cx', 'cy'}
            depth_min: 최소 유효 거리 (m)
            depth_max: 최대 유효 거리 (m)
            margin: ROI 마진 (픽셀)
            min_valid_ratio: 최소 유효 depth 비율
        """
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.margin = margin
        self.min_valid_ratio = min_valid_ratio

        logger.debug(f"PositionEstimator initialized: fx={self.fx}, fy={self.fy}")

    def estimate_position(
        self,
        bbox: Tuple[int, int, int, int],
        depth: np.ndarray,
        method: str = 'median'
    ) -> Optional[PositionResult]:
        """
        객체의 3D 위치 추정

        Args:
            bbox: 2D 바운딩 박스 (x, y, width, height)
            depth: Depth 이미지 (미터 단위)
            method: depth 값 계산 방법 ('median', 'mean', 'center')

        Returns:
            PositionResult 또는 None (실패 시)
        """
        x, y, w, h = bbox

        # BBox 중심
        center_u = x + w // 2
        center_v = y + h // 2

        # ROI 추출 (마진 적용)
        x1 = max(x + self.margin, 0)
        y1 = max(y + self.margin, 0)
        x2 = min(x + w - self.margin, depth.shape[1])
        y2 = min(y + h - self.margin, depth.shape[0])

        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid ROI: bbox={bbox}")
            return None

        depth_roi = depth[y1:y2, x1:x2]

        # 유효 depth 값 필터링
        valid_mask = (depth_roi > self.depth_min) & (depth_roi < self.depth_max)
        valid_depths = depth_roi[valid_mask]

        # 유효 비율 확인
        valid_ratio = len(valid_depths) / depth_roi.size if depth_roi.size > 0 else 0.0

        if valid_ratio < self.min_valid_ratio:
            logger.debug(f"Insufficient valid depth: ratio={valid_ratio:.2f}")
            return None

        # Depth 값 계산
        if method == 'median':
            z = np.median(valid_depths)
        elif method == 'mean':
            z = np.mean(valid_depths)
        elif method == 'center':
            # 중심점 주변의 depth
            z = self._get_center_depth(depth, center_u, center_v)
            if z is None or z < self.depth_min or z > self.depth_max:
                z = np.median(valid_depths)
        else:
            z = np.median(valid_depths)

        # 2D → 3D 변환
        x_3d = (center_u - self.cx) * z / self.fx
        y_3d = (center_v - self.cy) * z / self.fy

        position = np.array([x_3d, y_3d, z])

        # 신뢰도 계산
        depth_std = np.std(valid_depths)
        confidence = self._compute_confidence(valid_ratio, depth_std, z)

        return PositionResult(
            position=position,
            confidence=confidence,
            valid_depth_ratio=valid_ratio,
            depth_std=depth_std
        )

    def _get_center_depth(
        self,
        depth: np.ndarray,
        center_u: int,
        center_v: int,
        window_size: int = 5
    ) -> Optional[float]:
        """중심점 주변의 depth 값"""
        h, w = depth.shape[:2]
        half = window_size // 2

        y1 = max(0, center_v - half)
        y2 = min(h, center_v + half + 1)
        x1 = max(0, center_u - half)
        x2 = min(w, center_u + half + 1)

        center_roi = depth[y1:y2, x1:x2]
        valid = center_roi[(center_roi > self.depth_min) & (center_roi < self.depth_max)]

        if len(valid) > 0:
            return float(np.median(valid))
        return None

    def _compute_confidence(
        self,
        valid_ratio: float,
        depth_std: float,
        distance: float
    ) -> float:
        """
        신뢰도 계산

        - 유효 비율이 높을수록 좋음
        - depth 표준편차가 낮을수록 좋음
        - 거리가 가까울수록 좋음 (depth 오차 감소)
        """
        # 유효 비율 기여 (0-0.4)
        ratio_score = min(valid_ratio, 1.0) * 0.4

        # 표준편차 기여 (0-0.3), 낮을수록 좋음
        std_score = max(0, 0.3 - depth_std * 3)

        # 거리 기여 (0-0.3), 가까울수록 좋음
        dist_score = max(0, 0.3 - distance * 0.03)

        confidence = ratio_score + std_score + dist_score
        return min(max(confidence, 0.0), 1.0)

    def pixel_to_3d(
        self,
        u: int,
        v: int,
        z: float
    ) -> np.ndarray:
        """
        단일 픽셀을 3D 좌표로 변환

        Args:
            u: 픽셀 x 좌표
            v: 픽셀 y 좌표
            z: depth 값 (미터)

        Returns:
            3D 좌표 [x, y, z]
        """
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z])

    def batch_pixel_to_3d(
        self,
        pixels: np.ndarray,
        depths: np.ndarray
    ) -> np.ndarray:
        """
        여러 픽셀을 3D 좌표로 변환

        Args:
            pixels: 픽셀 좌표 배열 (N, 2) [u, v]
            depths: depth 배열 (N,)

        Returns:
            3D 좌표 배열 (N, 3) [x, y, z]
        """
        u = pixels[:, 0]
        v = pixels[:, 1]

        x = (u - self.cx) * depths / self.fx
        y = (v - self.cy) * depths / self.fy

        return np.column_stack([x, y, depths])

    def estimate_multiple(
        self,
        bboxes: list,
        depth: np.ndarray
    ) -> list:
        """
        여러 객체의 위치 추정

        Args:
            bboxes: 바운딩 박스 리스트
            depth: Depth 이미지

        Returns:
            PositionResult 리스트
        """
        return [self.estimate_position(bbox, depth) for bbox in bboxes]


class DepthAnalyzer:
    """
    Depth 이미지 분석 유틸리티
    """

    def __init__(
        self,
        depth_min: float = 0.1,
        depth_max: float = 10.0
    ):
        self.depth_min = depth_min
        self.depth_max = depth_max

    def get_statistics(self, depth: np.ndarray) -> Dict:
        """Depth 이미지 통계"""
        valid_mask = (depth > self.depth_min) & (depth < self.depth_max)
        valid_depths = depth[valid_mask]

        if len(valid_depths) == 0:
            return {
                'valid_ratio': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }

        return {
            'valid_ratio': len(valid_depths) / depth.size,
            'mean': float(np.mean(valid_depths)),
            'std': float(np.std(valid_depths)),
            'min': float(np.min(valid_depths)),
            'max': float(np.max(valid_depths)),
            'median': float(np.median(valid_depths))
        }

    def get_roi_statistics(
        self,
        depth: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Dict:
        """ROI 영역의 depth 통계"""
        x, y, w, h = bbox
        roi = depth[y:y+h, x:x+w]
        return self.get_statistics(roi)

    def find_closest_point(
        self,
        depth: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[int, int, float]:
        """
        가장 가까운 점 찾기

        Returns:
            (u, v, depth): 픽셀 좌표 및 depth
        """
        if bbox is not None:
            x, y, w, h = bbox
            roi = depth[y:y+h, x:x+w]
            offset = (x, y)
        else:
            roi = depth
            offset = (0, 0)

        valid_mask = (roi > self.depth_min) & (roi < self.depth_max)
        if not np.any(valid_mask):
            return -1, -1, 0.0

        # 마스크 적용
        masked_roi = np.where(valid_mask, roi, np.inf)
        min_idx = np.unravel_index(np.argmin(masked_roi), masked_roi.shape)

        u = int(min_idx[1] + offset[0])
        v = int(min_idx[0] + offset[1])
        z = float(roi[min_idx])

        return u, v, z
