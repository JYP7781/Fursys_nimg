"""
Point Cloud 처리 모듈
Depth 이미지 → Point Cloud 변환 및 처리
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Open3D 임포트 (없으면 fallback)
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logger.warning("Open3D not available")


@dataclass
class PointCloudData:
    """Point Cloud 데이터"""
    points: np.ndarray      # (N, 3) xyz 좌표
    colors: Optional[np.ndarray] = None  # (N, 3) RGB 색상 (0-1)
    normals: Optional[np.ndarray] = None  # (N, 3) 법선 벡터
    num_points: int = 0

    def __post_init__(self):
        self.num_points = len(self.points) if self.points is not None else 0


class PointCloudProcessor:
    """
    Point Cloud 처리기

    Depth 이미지를 Point Cloud로 변환하고
    다양한 처리를 수행합니다.
    """

    def __init__(
        self,
        intrinsics: Dict[str, float],
        depth_min: float = 0.1,
        depth_max: float = 10.0,
        voxel_size: float = 0.005
    ):
        """
        Args:
            intrinsics: 카메라 내부 파라미터
            depth_min: 최소 유효 거리
            depth_max: 최대 유효 거리
            voxel_size: 다운샘플링 voxel 크기 (미터)
        """
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.voxel_size = voxel_size

        logger.debug("PointCloudProcessor initialized")

    def depth_to_pointcloud(
        self,
        depth: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> PointCloudData:
        """
        Depth 이미지를 Point Cloud로 변환

        Args:
            depth: Depth 이미지 (미터 단위)
            rgb: RGB 이미지 (옵션)
            mask: 처리할 영역 마스크 (옵션)

        Returns:
            PointCloudData
        """
        h, w = depth.shape

        # 유효 depth 마스크 생성
        valid_mask = (depth > self.depth_min) & (depth < self.depth_max)

        if mask is not None:
            valid_mask = valid_mask & mask

        # 유효 픽셀 좌표
        v, u = np.where(valid_mask)
        z = depth[v, u]

        # 3D 좌표 계산
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        points = np.column_stack([x, y, z])

        # 색상 추출 (있는 경우)
        colors = None
        if rgb is not None:
            colors = rgb[v, u].astype(np.float32) / 255.0
            # BGR → RGB
            if colors.shape[1] == 3:
                colors = colors[:, ::-1]

        return PointCloudData(points=points, colors=colors)

    def depth_roi_to_pointcloud(
        self,
        depth: np.ndarray,
        bbox: Tuple[int, int, int, int],
        rgb: Optional[np.ndarray] = None
    ) -> PointCloudData:
        """
        특정 ROI의 Point Cloud 생성

        Args:
            depth: 전체 Depth 이미지
            bbox: 바운딩 박스 (x, y, w, h)
            rgb: RGB 이미지 (옵션)

        Returns:
            PointCloudData
        """
        x, y, w, h = bbox

        # ROI 추출
        depth_roi = depth[y:y+h, x:x+w]
        rgb_roi = rgb[y:y+h, x:x+w] if rgb is not None else None

        # 유효 depth 마스크
        valid_mask = (depth_roi > self.depth_min) & (depth_roi < self.depth_max)

        # 유효 픽셀 좌표
        v_roi, u_roi = np.where(valid_mask)
        z = depth_roi[v_roi, u_roi]

        # 전체 이미지 좌표로 변환
        u = u_roi + x
        v = v_roi + y

        # 3D 좌표 계산
        x_3d = (u - self.cx) * z / self.fx
        y_3d = (v - self.cy) * z / self.fy

        points = np.column_stack([x_3d, y_3d, z])

        # 색상 추출
        colors = None
        if rgb_roi is not None:
            colors = rgb_roi[v_roi, u_roi].astype(np.float32) / 255.0
            if colors.shape[1] == 3:
                colors = colors[:, ::-1]

        return PointCloudData(points=points, colors=colors)

    def downsample(
        self,
        pcd_data: PointCloudData,
        voxel_size: Optional[float] = None
    ) -> PointCloudData:
        """
        Voxel 다운샘플링

        Args:
            pcd_data: 입력 Point Cloud
            voxel_size: voxel 크기 (None이면 기본값 사용)

        Returns:
            다운샘플링된 PointCloudData
        """
        if voxel_size is None:
            voxel_size = self.voxel_size

        if HAS_OPEN3D:
            return self._downsample_open3d(pcd_data, voxel_size)
        else:
            return self._downsample_numpy(pcd_data, voxel_size)

    def _downsample_open3d(
        self,
        pcd_data: PointCloudData,
        voxel_size: float
    ) -> PointCloudData:
        """Open3D를 사용한 다운샘플링"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data.points)

        if pcd_data.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(pcd_data.colors)

        pcd_down = pcd.voxel_down_sample(voxel_size)

        points = np.asarray(pcd_down.points)
        colors = np.asarray(pcd_down.colors) if pcd_data.colors is not None else None

        return PointCloudData(points=points, colors=colors)

    def _downsample_numpy(
        self,
        pcd_data: PointCloudData,
        voxel_size: float
    ) -> PointCloudData:
        """Numpy만 사용한 간단한 다운샘플링"""
        points = pcd_data.points

        # Voxel 인덱스 계산
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        # 고유 voxel 찾기
        unique_indices, inverse = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )

        # 각 voxel의 중심점 계산
        downsampled = np.zeros((len(unique_indices), 3))
        counts = np.zeros(len(unique_indices))

        for i, idx in enumerate(inverse):
            downsampled[idx] += points[i]
            counts[idx] += 1

        downsampled /= counts[:, np.newaxis]

        # 색상도 평균
        colors = None
        if pcd_data.colors is not None:
            colors_avg = np.zeros((len(unique_indices), 3))
            for i, idx in enumerate(inverse):
                colors_avg[idx] += pcd_data.colors[i]
            colors_avg /= counts[:, np.newaxis]
            colors = colors_avg

        return PointCloudData(points=downsampled, colors=colors)

    def remove_outliers(
        self,
        pcd_data: PointCloudData,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> PointCloudData:
        """
        통계적 이상치 제거

        Args:
            pcd_data: 입력 Point Cloud
            nb_neighbors: 이웃 수
            std_ratio: 표준편차 비율

        Returns:
            이상치 제거된 PointCloudData
        """
        if HAS_OPEN3D:
            return self._remove_outliers_open3d(pcd_data, nb_neighbors, std_ratio)
        else:
            return self._remove_outliers_numpy(pcd_data, std_ratio)

    def _remove_outliers_open3d(
        self,
        pcd_data: PointCloudData,
        nb_neighbors: int,
        std_ratio: float
    ) -> PointCloudData:
        """Open3D를 사용한 이상치 제거"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data.points)

        if pcd_data.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(pcd_data.colors)

        pcd_clean, inlier_idx = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )

        points = np.asarray(pcd_clean.points)
        colors = np.asarray(pcd_clean.colors) if pcd_data.colors is not None else None

        return PointCloudData(points=points, colors=colors)

    def _remove_outliers_numpy(
        self,
        pcd_data: PointCloudData,
        std_ratio: float
    ) -> PointCloudData:
        """Numpy만 사용한 이상치 제거"""
        points = pcd_data.points

        # 중심점 기준 거리
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)

        # Z-score 기반 필터링
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + std_ratio * std_dist

        mask = distances < threshold
        filtered_points = points[mask]

        colors = None
        if pcd_data.colors is not None:
            colors = pcd_data.colors[mask]

        return PointCloudData(points=filtered_points, colors=colors)

    def estimate_normals(
        self,
        pcd_data: PointCloudData,
        radius: float = 0.03,
        max_nn: int = 30
    ) -> PointCloudData:
        """
        법선 벡터 추정

        Args:
            pcd_data: 입력 Point Cloud
            radius: 검색 반경
            max_nn: 최대 이웃 수

        Returns:
            법선이 추가된 PointCloudData
        """
        if not HAS_OPEN3D:
            logger.warning("Open3D not available for normal estimation")
            return pcd_data

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data.points)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )

        # 카메라 방향으로 법선 정렬
        pcd.orient_normals_towards_camera_location(np.zeros(3))

        return PointCloudData(
            points=pcd_data.points,
            colors=pcd_data.colors,
            normals=np.asarray(pcd.normals)
        )

    def compute_bounding_box(
        self,
        pcd_data: PointCloudData
    ) -> Dict[str, np.ndarray]:
        """
        바운딩 박스 계산

        Returns:
            {'min': [x,y,z], 'max': [x,y,z], 'center': [x,y,z], 'extent': [dx,dy,dz]}
        """
        points = pcd_data.points

        if len(points) == 0:
            return {
                'min': np.zeros(3),
                'max': np.zeros(3),
                'center': np.zeros(3),
                'extent': np.zeros(3)
            }

        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        center = (min_pt + max_pt) / 2
        extent = max_pt - min_pt

        return {
            'min': min_pt,
            'max': max_pt,
            'center': center,
            'extent': extent
        }

    def compute_oriented_bounding_box(
        self,
        pcd_data: PointCloudData
    ) -> Dict:
        """
        Oriented Bounding Box (OBB) 계산

        Returns:
            {'center': [x,y,z], 'extent': [dx,dy,dz], 'rotation': 3x3 matrix}
        """
        if not HAS_OPEN3D or len(pcd_data.points) < 10:
            # Fallback: AABB 사용
            aabb = self.compute_bounding_box(pcd_data)
            return {
                'center': aabb['center'],
                'extent': aabb['extent'],
                'rotation': np.eye(3)
            }

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data.points)

        obb = pcd.get_oriented_bounding_box()

        return {
            'center': np.asarray(obb.center),
            'extent': np.asarray(obb.extent),
            'rotation': np.asarray(obb.R)
        }

    def segment_plane(
        self,
        pcd_data: PointCloudData,
        distance_threshold: float = 0.01,
        ransac_n: int = 3,
        num_iterations: int = 1000
    ) -> Tuple[np.ndarray, PointCloudData, PointCloudData]:
        """
        RANSAC 평면 분할

        Args:
            pcd_data: 입력 Point Cloud
            distance_threshold: 평면 거리 임계값
            ransac_n: RANSAC 샘플 수
            num_iterations: 반복 횟수

        Returns:
            (plane_model, inlier_cloud, outlier_cloud)
        """
        if not HAS_OPEN3D:
            logger.warning("Open3D not available for plane segmentation")
            return np.zeros(4), pcd_data, PointCloudData(points=np.array([]))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data.points)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        return (
            np.array(plane_model),
            PointCloudData(points=np.asarray(inlier_cloud.points)),
            PointCloudData(points=np.asarray(outlier_cloud.points))
        )

    def to_open3d(self, pcd_data: PointCloudData):
        """Open3D PointCloud 객체로 변환"""
        if not HAS_OPEN3D:
            raise RuntimeError("Open3D not available")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data.points)

        if pcd_data.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(pcd_data.colors)

        if pcd_data.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(pcd_data.normals)

        return pcd

    def from_open3d(self, pcd) -> PointCloudData:
        """Open3D PointCloud에서 변환"""
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None

        return PointCloudData(points=points, colors=colors, normals=normals)
