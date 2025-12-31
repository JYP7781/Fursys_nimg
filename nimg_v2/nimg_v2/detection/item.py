"""
Item 및 ItemList 클래스
기존 nimg의 ItemList.py를 확장하여 3D 정보 지원 추가
"""

import numpy as np
import time
from typing import List, Optional, Tuple, Dict, Any
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Item:
    """
    탐지된 객체 정보를 담는 클래스
    기존 Item 클래스를 확장하여 3D 정보 지원
    """
    name: str
    code: int = -1
    born: float = field(default_factory=time.time)

    # 2D 정보
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    x2: int = 0
    y2: int = 0
    cx: int = 0  # center x
    cy: int = 0  # center y
    confidence: float = 0.0

    # 3D 위치 정보
    pos_3d: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x, y, z] in meters

    # 3D 속도 정보
    velocity_3d: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [vx, vy, vz] in m/s
    speed: float = 0.0  # 속력 (m/s)

    # 3D 방향 정보 (Roll, Pitch, Yaw in degrees)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    orientation_confidence: float = 0.0

    # Depth 정보
    depth_center: float = 0.0  # 중심점 depth
    depth_mean: float = 0.0    # 평균 depth

    # 추적 정보
    track_id: Optional[int] = None
    age: int = 0  # 추적 프레임 수

    # 메타데이터
    desc: str = ""

    def __post_init__(self):
        """초기화 후 처리"""
        if isinstance(self.pos_3d, list):
            self.pos_3d = np.array(self.pos_3d)
        if isinstance(self.velocity_3d, list):
            self.velocity_3d = np.array(self.velocity_3d)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """바운딩 박스 (x, y, w, h)"""
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        """2D 중심점"""
        return (self.cx, self.cy)

    @property
    def position_3d(self) -> np.ndarray:
        """3D 위치"""
        return self.pos_3d

    @property
    def orientation(self) -> Tuple[float, float, float]:
        """방향 (roll, pitch, yaw) in degrees"""
        return (self.roll, self.pitch, self.yaw)

    def set_rect(self, x: int, y: int, w: int, h: int):
        """2D 바운딩 박스 설정"""
        self.x = int(x)
        self.y = int(y)
        self.width = int(w)
        self.height = int(h)
        self.x2 = int(x + w)
        self.y2 = int(y + h)
        self.cx = int(x + w / 2)
        self.cy = int(y + h / 2)

    def set_rect_xyxy(self, x1: int, y1: int, x2: int, y2: int):
        """2D 바운딩 박스 설정 (xyxy 형식)"""
        self.set_rect(x1, y1, x2 - x1, y2 - y1)

    def set_position_3d(self, x: float, y: float, z: float):
        """3D 위치 설정"""
        self.pos_3d = np.array([x, y, z])

    def set_velocity_3d(self, vx: float, vy: float, vz: float):
        """3D 속도 설정"""
        self.velocity_3d = np.array([vx, vy, vz])
        self.speed = np.linalg.norm(self.velocity_3d)

    def set_orientation(self, roll: float, pitch: float, yaw: float, confidence: float = 1.0):
        """방향 설정 (degrees)"""
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.orientation_confidence = confidence

    def set_depth(self, center: float, mean: Optional[float] = None):
        """Depth 정보 설정"""
        self.depth_center = center
        self.depth_mean = mean if mean is not None else center

    def check_inside(self, x1: int, x2: int, y1: int, y2: int) -> bool:
        """중심점이 영역 내에 있는지 확인"""
        return x1 < self.cx < x2 and y1 < self.cy < y2

    def distance_to(self, other: 'Item') -> float:
        """다른 Item과의 3D 거리"""
        return np.linalg.norm(self.pos_3d - other.pos_3d)

    def distance_2d_to(self, other: 'Item') -> float:
        """다른 Item과의 2D 거리 (픽셀)"""
        return np.sqrt((self.cx - other.cx) ** 2 + (self.cy - other.cy) ** 2)

    def iou(self, other: 'Item') -> float:
        """다른 Item과의 IoU"""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = self.width * self.height
        area2 = other.width * other.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'name': self.name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center_2d': self.center,
            'position_3d': self.pos_3d.tolist(),
            'velocity_3d': self.velocity_3d.tolist(),
            'speed': self.speed,
            'orientation': {
                'roll': self.roll,
                'pitch': self.pitch,
                'yaw': self.yaw,
                'confidence': self.orientation_confidence
            },
            'depth': {
                'center': self.depth_center,
                'mean': self.depth_mean
            },
            'track_id': self.track_id,
            'age': self.age
        }

    def get_info_str(self) -> str:
        """정보 문자열 반환"""
        return (f"{self.name} conf={self.confidence:.2f} "
                f"pos=({self.pos_3d[0]:.3f}, {self.pos_3d[1]:.3f}, {self.pos_3d[2]:.3f}) "
                f"vel={self.speed:.3f}m/s "
                f"orient=(R:{self.roll:.1f}, P:{self.pitch:.1f}, Y:{self.yaw:.1f})")


class ItemList:
    """
    Item 객체들의 리스트를 관리하는 클래스
    """

    def __init__(self):
        self.items: List[Item] = []

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Item:
        return self.items[idx]

    def __iter__(self):
        return iter(self.items)

    def append(self, item: Item):
        """아이템 추가"""
        self.items.append(item)

    def extend(self, items: List[Item]):
        """여러 아이템 추가"""
        self.items.extend(items)

    def clear(self):
        """리스트 초기화"""
        self.items = []

    def size(self) -> int:
        """아이템 수 반환"""
        return len(self.items)

    def is_empty(self) -> bool:
        """비어있는지 확인"""
        return len(self.items) == 0

    def get_by_name(self, name: str) -> Optional[Item]:
        """이름으로 아이템 찾기"""
        for item in self.items:
            if item.name == name:
                return item
        return None

    def get_by_track_id(self, track_id: int) -> Optional[Item]:
        """트랙 ID로 아이템 찾기"""
        for item in self.items:
            if item.track_id == track_id:
                return item
        return None

    def filter_by_confidence(self, min_conf: float) -> 'ItemList':
        """신뢰도로 필터링"""
        result = ItemList()
        for item in self.items:
            if item.confidence >= min_conf:
                result.append(item)
        return result

    def filter_by_name(self, names: List[str]) -> 'ItemList':
        """이름으로 필터링"""
        result = ItemList()
        for item in self.items:
            if item.name in names:
                result.append(item)
        return result

    def get_best_item(self, by: str = 'confidence') -> Optional[Item]:
        """
        최상의 아이템 반환

        Args:
            by: 정렬 기준 ('confidence', 'area', 'depth')
        """
        if self.is_empty():
            return None

        if by == 'confidence':
            return max(self.items, key=lambda x: x.confidence)
        elif by == 'area':
            return max(self.items, key=lambda x: x.width * x.height)
        elif by == 'depth':
            return min(self.items, key=lambda x: x.depth_center if x.depth_center > 0 else float('inf'))
        else:
            return self.items[0]

    def get_most_common_name(self) -> Optional[str]:
        """가장 많이 등장하는 이름 반환"""
        if self.is_empty():
            return None

        names = [item.name for item in self.items]
        counter = Counter(names)
        return counter.most_common(1)[0][0]

    def sort_by(self, key: str = 'confidence', reverse: bool = True) -> 'ItemList':
        """
        정렬된 새 리스트 반환

        Args:
            key: 정렬 키 ('confidence', 'x', 'y', 'depth', 'area')
            reverse: 내림차순 여부
        """
        result = ItemList()

        if key == 'confidence':
            sorted_items = sorted(self.items, key=lambda x: x.confidence, reverse=reverse)
        elif key == 'x':
            sorted_items = sorted(self.items, key=lambda x: x.cx, reverse=reverse)
        elif key == 'y':
            sorted_items = sorted(self.items, key=lambda x: x.cy, reverse=reverse)
        elif key == 'depth':
            sorted_items = sorted(self.items, key=lambda x: x.depth_center, reverse=reverse)
        elif key == 'area':
            sorted_items = sorted(self.items, key=lambda x: x.width * x.height, reverse=reverse)
        else:
            sorted_items = self.items

        result.items = sorted_items
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        if self.is_empty():
            return {'count': 0}

        confidences = [item.confidence for item in self.items]
        depths = [item.depth_center for item in self.items if item.depth_center > 0]
        speeds = [item.speed for item in self.items]

        return {
            'count': len(self.items),
            'confidence': {
                'mean': np.mean(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'depth': {
                'mean': np.mean(depths) if depths else 0,
                'min': np.min(depths) if depths else 0,
                'max': np.max(depths) if depths else 0
            },
            'speed': {
                'mean': np.mean(speeds),
                'min': np.min(speeds),
                'max': np.max(speeds)
            },
            'names': dict(Counter([item.name for item in self.items]))
        }

    def to_list_dict(self) -> List[Dict[str, Any]]:
        """딕셔너리 리스트로 변환"""
        return [item.to_dict() for item in self.items]

    def list_info(self) -> str:
        """전체 정보 문자열 반환"""
        info = f"ItemList: {len(self.items)} items\n"
        for i, item in enumerate(self.items):
            info += f"  [{i}] {item.get_info_str()}\n"
        return info
