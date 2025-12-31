"""
nimg_v2: RGB + Depth + IMU 데이터를 활용한 상대적 속도변화량/각도변화량 측정 시스템
"""

__version__ = "2.0.0"
__author__ = "Fursys Image Processing Team"

from .data.data_loader import DataLoader, FrameData
from .detection.yolo_detector import YOLODetector, Detection
from .detection.item import Item, ItemList
from .tracking.kalman_filter_3d import KalmanFilter3D
from .tracking.object_tracker import ObjectTracker
from .estimation.position_estimator import PositionEstimator
from .estimation.orientation_estimator import OrientationEstimator, OrientationResult
from .estimation.point_cloud_processor import PointCloudProcessor
from .analysis.change_calculator import ChangeCalculator, ChangeResult
from .analysis.reference_manager import ReferenceManager
from .output.result_exporter import ResultExporter
from .output.visualizer import Visualizer

__all__ = [
    'DataLoader',
    'FrameData',
    'YOLODetector',
    'Detection',
    'Item',
    'ItemList',
    'KalmanFilter3D',
    'ObjectTracker',
    'PositionEstimator',
    'OrientationEstimator',
    'OrientationResult',
    'PointCloudProcessor',
    'ChangeCalculator',
    'ChangeResult',
    'ReferenceManager',
    'ResultExporter',
    'Visualizer',
]
