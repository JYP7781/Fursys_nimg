# nimg 시스템 구현 설계 가이드

**작성일**: 2025-12-17
**기반 문서**: comprehensive_improvement_research_2025.md
**목적**: RGB + Depth + IMU 데이터를 활용한 상대적 속도변화량/각도변화량 측정 시스템 구현

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [기존 코드베이스 분석](#2-기존-코드베이스-분석)
3. [테스트 데이터 구조](#3-테스트-데이터-구조)
4. [새 시스템 아키텍처 설계](#4-새-시스템-아키텍처-설계)
5. [핵심 모듈 설계](#5-핵심-모듈-설계)
6. [테스트 파이프라인 설계](#6-테스트-파이프라인-설계)
7. [구현 단계별 가이드](#7-구현-단계별-가이드)
8. [예상 결과 및 검증 방법](#8-예상-결과-및-검증-방법)

---

## 1. 시스템 개요

### 1.1 목표

기준 이미지(Reference Frame) 한 장을 선택하고, 이후 프레임들에서 다음을 측정:

| 측정 항목 | 설명 | 단위 |
|----------|------|------|
| **상대적 속도변화량** | 기준 프레임 대비 객체의 3D 이동 속도 | m/s |
| **상대적 각도변화량** | 기준 프레임 대비 객체의 Roll/Pitch/Yaw 변화 | degrees (°) |

### 1.2 입력 데이터

```
테스트 데이터 폴더:
├── 20251208_155531_output/
│   ├── color_000000.png ~ color_012394.png  (RGB 이미지)
│   ├── depth_000000.png ~ depth_012394.png  (Depth 이미지)
│   └── imu_data.csv                          (IMU 데이터)
│
├── 20251208_161246_output/
│   └── (동일 구조)
│
└── models/
    └── class187_image85286_v12x_250epochs.pt (품종인식 AI 모델, 115MB, YOLOv12x)
```

### 1.3 처리 흐름 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                    전체 처리 흐름                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [기준 프레임 선택] (Frame 0)                                    │
│         │                                                        │
│         ▼                                                        │
│  [기준 상태 추출]                                                │
│    • 품종 탐지 (YOLO)                                           │
│    • 3D 위치 계산                                               │
│    • 3D 방향(PCA/OBB) 계산                                      │
│    • Kalman Filter 초기화                                       │
│         │                                                        │
│         ▼                                                        │
│  [후속 프레임 처리] (Frame 1, 2, 3, ...)                         │
│    ┌─────────────────────────────────────────────────┐          │
│    │ For each frame:                                 │          │
│    │   • 품종 탐지                                   │          │
│    │   • 3D 위치 계산                               │          │
│    │   • Kalman Filter 업데이트 → 속도 추정         │          │
│    │   • PCA/OBB → 현재 방향                        │          │
│    │   • 기준 대비 변화량 계산                      │          │
│    │   • 결과 저장                                   │          │
│    └─────────────────────────────────────────────────┘          │
│         │                                                        │
│         ▼                                                        │
│  [결과 출력]                                                     │
│    • CSV/JSON 파일                                              │
│    • 시각화 그래프                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 기존 코드베이스 분석

### 2.1 현재 디렉토리 구조

```
src/nimg/
├── nimg/
│   ├── nimg.py                    # 메인 ROS2 노드 (nimg_x86 클래스)
│   ├── submodules/
│   │   ├── nodeL515.py            # dSensor 클래스 - L515 카메라 처리
│   │   ├── detectProcessor.py     # 객체 탐지 및 각도 측정 (현재)
│   │   ├── detect.py              # YOLOv5 기반 Detector 클래스
│   │   ├── lineDetector.py        # Hough 변환 라인 검출
│   │   ├── orbProcessor.py        # ORB 특징점 처리
│   │   ├── ItemList.py            # Item, ItemList 클래스
│   │   └── ...
│   ├── models/                    # YOLOv5 모델 정의
│   └── utils/                     # 유틸리티 함수
│
└── models/
    └── class187_image85286_v12x_250epochs.pt  # 새 품종인식 모델
```

### 2.2 현재 시스템의 핵심 문제점

| 문제 영역 | 현재 상태 | 개선 방향 |
|----------|----------|----------|
| **속도 측정** | 미구현 (`speed = 0`) | 3D Kalman Filter로 추정 |
| **Pitch 각도** | 단순 depth 차이 계산 (±15-20° 오차) | PCA/OBB 기반 3D 방향 추정 |
| **Yaw 각도** | 2D Hough 변환 (±10-15° 오차) | PCA/OBB 기반 3D 방향 추정 |
| **Roll 각도** | 미구현 | PCA/OBB 기반 3D 방향 추정 |
| **Point Cloud** | 파일 저장용으로만 사용 | 실시간 3D 분석에 활용 |

### 2.3 재사용 가능한 모듈

| 모듈 | 파일 | 재사용 가능 여부 | 비고 |
|------|------|-----------------|------|
| `Detector` | detect.py | ✅ 가능 | 모델 경로만 변경 |
| `Item`, `ItemList` | ItemList.py | ✅ 가능 | 확장하여 3D 정보 추가 |
| `orbProcessor` | orbProcessor.py | △ 보조용 | 객체 재식별에만 사용 |
| `lineDetector` | lineDetector.py | ❌ 대체 필요 | PCA/OBB로 대체 |
| `detectProcessor` | detectProcessor.py | △ 수정 필요 | 대폭 리팩토링 필요 |

---

## 3. 테스트 데이터 구조

### 3.1 데이터셋 상세 분석

#### 20251208_155531_output 폴더

| 데이터 타입 | 파일 형식 | 개수 | 비고 |
|------------|----------|------|------|
| RGB 이미지 | color_XXXXXX.png | ~12,395개 | PNG 포맷 |
| Depth 이미지 | depth_XXXXXX.png | ~12,395개 | 16-bit PNG |
| IMU 데이터 | imu_data.csv | 1개 | ~24,790 행 |
| 비디오 | *_video_rgb.mp4 | 1개 | 640MB |

#### IMU 데이터 구조

```csv
timestamp,type,x,y,z
1765176931633.066,accel,0.019613,-9.630129,-1.520030
1765176931637.96,gyro,-0.001595,0.001595,-0.002129
...
```

| 필드 | 설명 | 단위 |
|------|------|------|
| `timestamp` | Unix 시간 (밀리초) | ms |
| `type` | 센서 타입 (`accel` 또는 `gyro`) | - |
| `x, y, z` | 3축 값 | m/s² (accel), rad/s (gyro) |

### 3.2 데이터 동기화 전략

```python
# 프레임 번호와 IMU 데이터 동기화 방법
# 가정: 30 FPS, 프레임 번호 = 시간순 인덱스

def sync_imu_with_frame(frame_idx, imu_df, fps=30):
    """프레임 인덱스에 해당하는 IMU 데이터 조회"""
    frame_time = frame_idx / fps  # 프레임 시간 (초)

    # 해당 시간 범위의 IMU 데이터 필터링
    start_time = frame_time
    end_time = frame_time + (1 / fps)

    # 가속도계와 자이로스코프 데이터 분리
    accel_data = imu_df[(imu_df['type'] == 'accel')]
    gyro_data = imu_df[(imu_df['type'] == 'gyro')]

    return accel_data, gyro_data
```

### 3.3 Depth 이미지 해석

```python
import cv2
import numpy as np

def load_depth_image(depth_path, depth_scale=0.001):
    """Depth 이미지 로드 및 미터 단위 변환"""
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 16-bit PNG → 미터 단위
    depth_meters = depth_raw.astype(np.float32) * depth_scale

    return depth_meters
```

---

## 4. 새 시스템 아키텍처 설계

### 4.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Image Processing System v2.0                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Data Layer                                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ RGB Loader  │  │Depth Loader │  │ IMU Loader  │  │ Model Loader│  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Processing Layer                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │   │
│  │  │  Detector    │  │   Tracker    │  │  Estimator   │                │   │
│  │  │  (YOLO)      │  │ (Kalman 3D)  │  │ (PCA/OBB)    │                │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Analysis Layer                                  │   │
│  │  ┌──────────────────────┐  ┌──────────────────────┐                  │   │
│  │  │ VelocityChangeCalc   │  │  AngleChangeCalc     │                  │   │
│  │  │ (상대적 속도변화량)   │  │  (상대적 각도변화량)  │                  │   │
│  │  └──────────────────────┘  └──────────────────────┘                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Output Layer                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │  CSV Export │  │ Visualizer  │  │ Report Gen  │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 새 디렉토리 구조

```
src/nimg_v2/
├── nimg_v2/
│   ├── __init__.py
│   ├── config/
│   │   ├── camera_intrinsics.yaml    # D455 카메라 내부 파라미터
│   │   └── processing_config.yaml    # 처리 설정
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py            # RGB/Depth/IMU 데이터 로더
│   │   └── frame_synchronizer.py     # 프레임-IMU 동기화
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py          # YOLO 기반 객체 탐지
│   │   └── item.py                   # Item/ItemList 확장
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── kalman_filter_3d.py       # 3D Kalman Filter (CA 모델)
│   │   └── object_tracker.py         # 객체 추적기
│   │
│   ├── estimation/
│   │   ├── __init__.py
│   │   ├── position_estimator.py     # 3D 위치 추정
│   │   ├── velocity_estimator.py     # 속도 추정
│   │   ├── orientation_estimator.py  # PCA/OBB 방향 추정
│   │   └── point_cloud_processor.py  # Point Cloud 처리
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── change_calculator.py      # 변화량 계산
│   │   └── reference_manager.py      # 기준 프레임 관리
│   │
│   ├── output/
│   │   ├── __init__.py
│   │   ├── result_exporter.py        # 결과 내보내기
│   │   └── visualizer.py             # 시각화
│   │
│   └── main.py                        # 메인 실행 파일
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_kalman_filter.py
│   ├── test_orientation_estimator.py
│   └── test_full_pipeline.py
│
└── requirements.txt
```

---

## 5. 핵심 모듈 설계

### 5.1 데이터 로더 (data_loader.py)

```python
import os
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class FrameData:
    """단일 프레임의 모든 데이터"""
    frame_idx: int
    rgb: np.ndarray
    depth: np.ndarray
    timestamp: float
    accel: Optional[np.ndarray] = None  # [ax, ay, az]
    gyro: Optional[np.ndarray] = None   # [gx, gy, gz]

class DataLoader:
    """RGB, Depth, IMU 데이터 통합 로더"""

    def __init__(self, data_dir: str, depth_scale: float = 0.001):
        self.data_dir = data_dir
        self.depth_scale = depth_scale

        # 파일 목록 로드
        self.color_files = sorted([f for f in os.listdir(data_dir)
                                   if f.startswith('color_')])
        self.depth_files = sorted([f for f in os.listdir(data_dir)
                                   if f.startswith('depth_')])

        # IMU 데이터 로드
        imu_path = os.path.join(data_dir, 'imu_data.csv')
        if os.path.exists(imu_path):
            self.imu_df = pd.read_csv(imu_path)
        else:
            self.imu_df = None

        self.num_frames = min(len(self.color_files), len(self.depth_files))

    def __len__(self):
        return self.num_frames

    def load_frame(self, idx: int) -> FrameData:
        """특정 인덱스의 프레임 데이터 로드"""
        if idx >= self.num_frames:
            raise IndexError(f"Frame index {idx} out of range")

        # RGB 로드
        rgb_path = os.path.join(self.data_dir, self.color_files[idx])
        rgb = cv2.imread(rgb_path)

        # Depth 로드
        depth_path = os.path.join(self.data_dir, self.depth_files[idx])
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth_raw.astype(np.float32) * self.depth_scale

        # IMU 데이터 (프레임별 평균)
        accel, gyro = None, None
        if self.imu_df is not None:
            accel, gyro = self._get_imu_for_frame(idx)

        return FrameData(
            frame_idx=idx,
            rgb=rgb,
            depth=depth,
            timestamp=idx / 30.0,  # 30 FPS 가정
            accel=accel,
            gyro=gyro
        )

    def _get_imu_for_frame(self, idx: int, fps: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
        """프레임에 해당하는 IMU 데이터 조회"""
        # 프레임 시간 범위
        t_start = idx / fps
        t_end = (idx + 1) / fps

        # 가속도계 평균
        accel_df = self.imu_df[self.imu_df['type'] == 'accel']
        accel = np.array([
            accel_df['x'].mean(),
            accel_df['y'].mean(),
            accel_df['z'].mean()
        ])

        # 자이로스코프 평균
        gyro_df = self.imu_df[self.imu_df['type'] == 'gyro']
        gyro = np.array([
            gyro_df['x'].mean(),
            gyro_df['y'].mean(),
            gyro_df['z'].mean()
        ])

        return accel, gyro
```

### 5.2 3D Kalman Filter (kalman_filter_3d.py)

```python
import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Tuple

class KalmanFilter3D:
    """
    Constant Acceleration (CA) 모델 기반 3D Kalman Filter

    상태 벡터: [x, y, z, vx, vy, vz, ax, ay, az]
    측정 벡터: [x, y, z]
    """

    def __init__(self, dt: float = 1/30.0, process_noise: float = 0.1):
        self.dt = dt

        # Kalman Filter 초기화
        self.kf = KalmanFilter(dim_x=9, dim_z=3)

        # 상태 전이 행렬 F (CA 모델)
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],           # x
            [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],           # y
            [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],           # z
            [0, 0, 0, 1, 0, 0, dt, 0, 0],                   # vx
            [0, 0, 0, 0, 1, 0, 0, dt, 0],                   # vy
            [0, 0, 0, 0, 0, 1, 0, 0, dt],                   # vz
            [0, 0, 0, 0, 0, 0, 1, 0, 0],                    # ax
            [0, 0, 0, 0, 0, 0, 0, 1, 0],                    # ay
            [0, 0, 0, 0, 0, 0, 0, 0, 1],                    # az
        ])

        # 측정 행렬 H (위치만 측정)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
        ])

        # 프로세스 노이즈 Q
        q = process_noise
        self.kf.Q = np.eye(9) * q
        self.kf.Q[6:9, 6:9] *= 10  # 가속도는 더 큰 노이즈

        # 측정 노이즈 R (초기값, 적응형으로 업데이트)
        self.kf.R = np.eye(3) * 0.01

        # 초기 상태 공분산
        self.kf.P = np.eye(9) * 1.0

        self.initialized = False

    def initialize(self, position: np.ndarray):
        """초기 위치로 필터 초기화"""
        self.kf.x = np.zeros(9)
        self.kf.x[0:3] = position
        self.initialized = True

    def update_measurement_noise(self, distance: float):
        """거리 기반 적응형 측정 노이즈 업데이트 (D455 특성)"""
        # D455 baseline = 95mm 기준 오차 모델
        base_error = 0.005  # 1m에서 5mm
        error = base_error * (distance ** 2)
        error = np.clip(error, 0.002, 0.1)  # 2mm ~ 100mm

        self.kf.R = np.eye(3) * (error ** 2)

    def predict_and_update(self, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        예측 및 업데이트

        Returns:
            position: 추정된 3D 위치 [x, y, z]
            velocity: 추정된 3D 속도 [vx, vy, vz]
            acceleration: 추정된 3D 가속도 [ax, ay, az]
        """
        if not self.initialized:
            self.initialize(position)
            return position, np.zeros(3), np.zeros(3)

        # 거리 기반 적응형 노이즈
        distance = np.linalg.norm(position)
        self.update_measurement_noise(distance)

        # 예측 및 업데이트
        self.kf.predict()
        self.kf.update(position)

        # 상태 추출
        est_position = self.kf.x[0:3]
        est_velocity = self.kf.x[3:6]
        est_acceleration = self.kf.x[6:9]

        return est_position, est_velocity, est_acceleration

    def get_state(self) -> dict:
        """현재 상태 반환"""
        return {
            'position': self.kf.x[0:3].copy(),
            'velocity': self.kf.x[3:6].copy(),
            'acceleration': self.kf.x[6:9].copy()
        }
```

### 5.3 방향 추정기 (orientation_estimator.py)

```python
import numpy as np
import open3d as o3d
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class OrientationResult:
    """방향 추정 결과"""
    roll: float   # X축 회전 (degrees)
    pitch: float  # Y축 회전 (degrees)
    yaw: float    # Z축 회전 (degrees)
    center: np.ndarray  # 객체 중심점
    axes: np.ndarray    # 주축 방향 벡터 (3x3)
    confidence: float   # 추정 신뢰도

class OrientationEstimator:
    """PCA/OBB 기반 3D 방향 추정기"""

    def __init__(self, min_points: int = 100):
        self.min_points = min_points

    def estimate_from_depth(
        self,
        depth: np.ndarray,
        bbox: Tuple[int, int, int, int],  # x, y, w, h
        intrinsics: dict
    ) -> Optional[OrientationResult]:
        """
        Depth 이미지에서 3D 방향 추정

        Args:
            depth: Depth 이미지 (미터 단위)
            bbox: 2D 바운딩 박스 (x, y, width, height)
            intrinsics: 카메라 내부 파라미터 {'fx', 'fy', 'cx', 'cy'}

        Returns:
            OrientationResult: 방향 추정 결과
        """
        x, y, w, h = bbox

        # ROI 추출
        depth_roi = depth[y:y+h, x:x+w]

        # 유효 깊이 마스크
        valid_mask = (depth_roi > 0.1) & (depth_roi < 10.0)

        if np.sum(valid_mask) < self.min_points:
            return None

        # 2D → 3D 변환
        points_3d = self._depth_to_pointcloud(
            depth_roi, valid_mask, x, y, intrinsics
        )

        if len(points_3d) < self.min_points:
            return None

        # PCA 분석
        return self._pca_orientation(points_3d)

    def estimate_from_pointcloud(
        self,
        points: np.ndarray
    ) -> Optional[OrientationResult]:
        """Point Cloud에서 직접 방향 추정"""
        if len(points) < self.min_points:
            return None
        return self._pca_orientation(points)

    def _depth_to_pointcloud(
        self,
        depth_roi: np.ndarray,
        mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        intrinsics: dict
    ) -> np.ndarray:
        """Depth ROI를 3D Point Cloud로 변환"""
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        # 유효 픽셀 좌표
        v, u = np.where(mask)
        z = depth_roi[v, u]

        # 전체 이미지 좌표로 변환
        u_full = u + offset_x
        v_full = v + offset_y

        # 3D 좌표 계산
        x = (u_full - cx) * z / fx
        y = (v_full - cy) * z / fy

        return np.column_stack([x, y, z])

    def _pca_orientation(self, points: np.ndarray) -> OrientationResult:
        """PCA 기반 방향 추정"""
        # Open3D Point Cloud 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 이상치 제거
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        points_clean = np.asarray(pcd.points)

        if len(points_clean) < self.min_points:
            points_clean = points

        # 중심점 계산
        center = np.mean(points_clean, axis=0)

        # 공분산 행렬 및 PCA
        centered = points_clean - center
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 큰 순서로 정렬
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 주축에서 Roll, Pitch, Yaw 추출
        roll, pitch, yaw = self._axes_to_euler(eigenvectors)

        # 신뢰도 계산 (eigenvalue ratio)
        total = np.sum(eigenvalues)
        confidence = eigenvalues[0] / total if total > 0 else 0

        return OrientationResult(
            roll=np.degrees(roll),
            pitch=np.degrees(pitch),
            yaw=np.degrees(yaw),
            center=center,
            axes=eigenvectors,
            confidence=confidence
        )

    def _axes_to_euler(self, axes: np.ndarray) -> Tuple[float, float, float]:
        """주축 벡터에서 오일러 각도 추출"""
        # Z축 (가장 짧은 축 = 두께 방향)이 세 번째 열
        z_axis = axes[:, 2]

        # 기준 Z축 [0, 0, 1]과의 비교
        pitch = np.arcsin(-z_axis[0])  # X 성분에서 pitch
        roll = np.arctan2(z_axis[1], z_axis[2])  # Y, Z 성분에서 roll

        # Yaw는 X-Y 평면 투영에서 계산
        x_axis = axes[:, 0]
        yaw = np.arctan2(x_axis[1], x_axis[0])

        return roll, pitch, yaw
```

### 5.4 변화량 계산기 (change_calculator.py)

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .orientation_estimator import OrientationResult

@dataclass
class ChangeResult:
    """기준 대비 변화량"""
    # 위치 변화
    position_change: np.ndarray  # [dx, dy, dz] (meters)

    # 속도 변화
    velocity: np.ndarray         # [vx, vy, vz] (m/s)
    speed: float                 # 속력 (m/s)

    # 각도 변화
    roll_change: float           # degrees
    pitch_change: float          # degrees
    yaw_change: float            # degrees

    # 메타데이터
    frame_idx: int
    timestamp: float
    confidence: float

class ChangeCalculator:
    """기준 프레임 대비 변화량 계산"""

    def __init__(self):
        self.reference_position: Optional[np.ndarray] = None
        self.reference_orientation: Optional[OrientationResult] = None
        self.reference_set = False

    def set_reference(
        self,
        position: np.ndarray,
        orientation: OrientationResult
    ):
        """기준 프레임 설정"""
        self.reference_position = position.copy()
        self.reference_orientation = orientation
        self.reference_set = True

    def calculate_change(
        self,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        current_orientation: OrientationResult,
        frame_idx: int,
        timestamp: float
    ) -> ChangeResult:
        """현재 프레임과 기준 프레임 간의 변화량 계산"""
        if not self.reference_set:
            raise ValueError("Reference frame not set")

        # 위치 변화
        position_change = current_position - self.reference_position

        # 속도 (Kalman Filter에서 직접 추정)
        speed = np.linalg.norm(current_velocity)

        # 각도 변화
        roll_change = current_orientation.roll - self.reference_orientation.roll
        pitch_change = current_orientation.pitch - self.reference_orientation.pitch
        yaw_change = current_orientation.yaw - self.reference_orientation.yaw

        # -180 ~ 180 범위로 정규화
        roll_change = self._normalize_angle(roll_change)
        pitch_change = self._normalize_angle(pitch_change)
        yaw_change = self._normalize_angle(yaw_change)

        # 신뢰도 (두 프레임의 기하평균)
        confidence = np.sqrt(
            self.reference_orientation.confidence *
            current_orientation.confidence
        )

        return ChangeResult(
            position_change=position_change,
            velocity=current_velocity,
            speed=speed,
            roll_change=roll_change,
            pitch_change=pitch_change,
            yaw_change=yaw_change,
            frame_idx=frame_idx,
            timestamp=timestamp,
            confidence=confidence
        )

    def _normalize_angle(self, angle: float) -> float:
        """각도를 -180 ~ 180 범위로 정규화"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
```

### 5.5 위치 추정기 (position_estimator.py)

```python
import numpy as np
from typing import Tuple, Optional

class PositionEstimator:
    """2D BBox + Depth → 3D 위치 추정"""

    def __init__(self, intrinsics: dict):
        """
        Args:
            intrinsics: {'fx', 'fy', 'cx', 'cy'}
        """
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']

    def estimate_position(
        self,
        bbox: Tuple[int, int, int, int],  # x, y, w, h
        depth: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        객체의 3D 위치 추정

        Returns:
            position: [x, y, z] in meters
            confidence: depth 신뢰도 (유효 픽셀 비율)
        """
        x, y, w, h = bbox

        # BBox 중심
        center_u = x + w // 2
        center_v = y + h // 2

        # ROI에서 중앙 영역의 depth 값 추출 (노이즈 감소)
        margin = 5
        x1, y1 = max(x + margin, 0), max(y + margin, 0)
        x2, y2 = min(x + w - margin, depth.shape[1]), min(y + h - margin, depth.shape[0])

        depth_roi = depth[y1:y2, x1:x2]

        # 유효 depth 값 필터링
        valid_mask = (depth_roi > 0.1) & (depth_roi < 10.0)
        valid_depths = depth_roi[valid_mask]

        if len(valid_depths) == 0:
            return np.array([0.0, 0.0, 0.0]), 0.0

        # 중앙값 사용 (이상치에 강건)
        z = np.median(valid_depths)

        # 2D → 3D 변환
        x_3d = (center_u - self.cx) * z / self.fx
        y_3d = (center_v - self.cy) * z / self.fy

        position = np.array([x_3d, y_3d, z])

        # 신뢰도 (유효 픽셀 비율)
        confidence = len(valid_depths) / depth_roi.size

        return position, confidence
```

---

## 6. 테스트 파이프라인 설계

### 6.1 메인 파이프라인 (main.py)

```python
import os
import numpy as np
import pandas as pd
from typing import List, Optional
from dataclasses import asdict

# 모듈 임포트
from data.data_loader import DataLoader, FrameData
from detection.yolo_detector import YOLODetector
from tracking.kalman_filter_3d import KalmanFilter3D
from estimation.position_estimator import PositionEstimator
from estimation.orientation_estimator import OrientationEstimator
from analysis.change_calculator import ChangeCalculator, ChangeResult

class ImageProcessingPipeline:
    """영상처리 메인 파이프라인"""

    def __init__(
        self,
        model_path: str,
        intrinsics: dict,
        reference_frame_idx: int = 0
    ):
        # 모델 및 추정기 초기화
        self.detector = YOLODetector(model_path)
        self.position_estimator = PositionEstimator(intrinsics)
        self.orientation_estimator = OrientationEstimator()
        self.kalman_filter = KalmanFilter3D()
        self.change_calculator = ChangeCalculator()

        self.reference_frame_idx = reference_frame_idx
        self.intrinsics = intrinsics
        self.results: List[ChangeResult] = []

    def process_dataset(
        self,
        data_dir: str,
        max_frames: Optional[int] = None
    ) -> pd.DataFrame:
        """전체 데이터셋 처리"""
        # 데이터 로더 초기화
        loader = DataLoader(data_dir)
        num_frames = len(loader) if max_frames is None else min(len(loader), max_frames)

        print(f"Processing {num_frames} frames from {data_dir}")

        for idx in range(num_frames):
            frame = loader.load_frame(idx)

            if idx == self.reference_frame_idx:
                # 기준 프레임 처리
                self._process_reference_frame(frame)
            else:
                # 후속 프레임 처리
                result = self._process_frame(frame)
                if result is not None:
                    self.results.append(result)

            if idx % 100 == 0:
                print(f"  Processed {idx}/{num_frames} frames")

        # 결과를 DataFrame으로 변환
        return self._results_to_dataframe()

    def _process_reference_frame(self, frame: FrameData):
        """기준 프레임 처리"""
        # 1. 객체 탐지
        detections = self.detector.detect(frame.rgb)
        if len(detections) == 0:
            raise ValueError(f"No object detected in reference frame {frame.frame_idx}")

        # 가장 신뢰도 높은 객체 선택
        best_det = max(detections, key=lambda d: d.confidence)
        bbox = (best_det.x, best_det.y, best_det.width, best_det.height)

        # 2. 3D 위치 추정
        position, _ = self.position_estimator.estimate_position(bbox, frame.depth)

        # 3. 방향 추정
        orientation = self.orientation_estimator.estimate_from_depth(
            frame.depth, bbox, self.intrinsics
        )

        if orientation is None:
            raise ValueError("Failed to estimate orientation for reference frame")

        # 4. 기준 상태 설정
        self.change_calculator.set_reference(position, orientation)
        self.kalman_filter.initialize(position)

        print(f"Reference frame set: position={position}, "
              f"orientation=(R:{orientation.roll:.2f}, P:{orientation.pitch:.2f}, Y:{orientation.yaw:.2f})")

    def _process_frame(self, frame: FrameData) -> Optional[ChangeResult]:
        """개별 프레임 처리"""
        # 1. 객체 탐지
        detections = self.detector.detect(frame.rgb)
        if len(detections) == 0:
            return None

        best_det = max(detections, key=lambda d: d.confidence)
        bbox = (best_det.x, best_det.y, best_det.width, best_det.height)

        # 2. 3D 위치 추정
        position, pos_conf = self.position_estimator.estimate_position(bbox, frame.depth)

        if pos_conf < 0.3:  # 신뢰도 낮으면 스킵
            return None

        # 3. Kalman Filter 업데이트 → 속도 추정
        est_position, velocity, acceleration = self.kalman_filter.predict_and_update(position)

        # 4. 방향 추정
        orientation = self.orientation_estimator.estimate_from_depth(
            frame.depth, bbox, self.intrinsics
        )

        if orientation is None:
            return None

        # 5. 변화량 계산
        result = self.change_calculator.calculate_change(
            current_position=est_position,
            current_velocity=velocity,
            current_orientation=orientation,
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp
        )

        return result

    def _results_to_dataframe(self) -> pd.DataFrame:
        """결과를 DataFrame으로 변환"""
        data = []
        for r in self.results:
            row = {
                'frame_idx': r.frame_idx,
                'timestamp': r.timestamp,
                'dx': r.position_change[0],
                'dy': r.position_change[1],
                'dz': r.position_change[2],
                'vx': r.velocity[0],
                'vy': r.velocity[1],
                'vz': r.velocity[2],
                'speed': r.speed,
                'roll_change': r.roll_change,
                'pitch_change': r.pitch_change,
                'yaw_change': r.yaw_change,
                'confidence': r.confidence
            }
            data.append(row)

        return pd.DataFrame(data)


def main():
    """메인 실행"""
    # 설정
    MODEL_PATH = "./models/class187_image85286_v12x_250epochs.pt"
    DATA_DIR = "./20251208_155531_output"
    OUTPUT_DIR = "./output"

    # D455 카메라 내부 파라미터 (예시 값, 실제 캘리브레이션 필요)
    INTRINSICS = {
        'fx': 385.0,  # focal length x
        'fy': 385.0,  # focal length y
        'cx': 320.0,  # principal point x
        'cy': 240.0   # principal point y
    }

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 파이프라인 실행
    pipeline = ImageProcessingPipeline(
        model_path=MODEL_PATH,
        intrinsics=INTRINSICS,
        reference_frame_idx=0
    )

    results_df = pipeline.process_dataset(DATA_DIR, max_frames=1000)

    # 결과 저장
    output_path = os.path.join(OUTPUT_DIR, "velocity_angle_changes.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # 요약 통계
    print("\n=== Summary Statistics ===")
    print(f"Total frames processed: {len(results_df)}")
    print(f"Average speed: {results_df['speed'].mean():.4f} m/s")
    print(f"Max speed: {results_df['speed'].max():.4f} m/s")
    print(f"Roll change range: {results_df['roll_change'].min():.2f}° ~ {results_df['roll_change'].max():.2f}°")
    print(f"Pitch change range: {results_df['pitch_change'].min():.2f}° ~ {results_df['pitch_change'].max():.2f}°")
    print(f"Yaw change range: {results_df['yaw_change'].min():.2f}° ~ {results_df['yaw_change'].max():.2f}°")


if __name__ == "__main__":
    main()
```

### 6.2 YOLO 탐지기 (yolo_detector.py)

```python
import torch
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List
from ultralytics import YOLO  # 또는 기존 YOLOv5 사용

@dataclass
class Detection:
    """탐지 결과"""
    class_id: int
    class_name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    x2: int
    y2: int

class YOLODetector:
    """YOLO 기반 객체 탐지기"""

    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect(self, image: np.ndarray) -> List[Detection]:
        """이미지에서 객체 탐지"""
        results = self.model(image, conf=self.conf_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=self.model.names[cls_id],
                    confidence=conf,
                    x=int(x1),
                    y=int(y1),
                    width=int(x2 - x1),
                    height=int(y2 - y1),
                    x2=int(x2),
                    y2=int(y2)
                ))

        return detections
```

---

## 7. 구현 단계별 가이드

### 7.1 Phase 1: 기본 인프라 구축

| 작업 | 설명 | 산출물 |
|------|------|--------|
| 디렉토리 구조 생성 | 새 모듈 디렉토리 생성 | `src/nimg_v2/` |
| 데이터 로더 구현 | RGB, Depth, IMU 로드 | `data_loader.py` |
| 설정 파일 작성 | 카메라 내부 파라미터 | `camera_intrinsics.yaml` |
| 기본 테스트 | 데이터 로딩 테스트 | `test_data_loader.py` |

### 7.2 Phase 2: 탐지 및 위치 추정

| 작업 | 설명 | 산출물 |
|------|------|--------|
| YOLO 탐지기 구현 | 새 모델 통합 | `yolo_detector.py` |
| 위치 추정기 구현 | 2D→3D 변환 | `position_estimator.py` |
| 기본 테스트 | 탐지 및 위치 테스트 | `test_detection.py` |

### 7.3 Phase 3: 속도 추정 (Kalman Filter)

| 작업 | 설명 | 산출물 |
|------|------|--------|
| Kalman Filter 구현 | CA 모델 기반 | `kalman_filter_3d.py` |
| 적응형 노이즈 추가 | D455 특성 반영 | (통합) |
| 단위 테스트 | 속도 추정 정확도 | `test_kalman_filter.py` |

### 7.4 Phase 4: 방향 추정 (PCA/OBB)

| 작업 | 설명 | 산출물 |
|------|------|--------|
| 방향 추정기 구현 | PCA 기반 | `orientation_estimator.py` |
| Point Cloud 처리 | 노이즈 제거 | (통합) |
| 단위 테스트 | 각도 추정 정확도 | `test_orientation.py` |

### 7.5 Phase 5: 통합 및 테스트

| 작업 | 설명 | 산출물 |
|------|------|--------|
| 변화량 계산기 구현 | 기준 프레임 대비 | `change_calculator.py` |
| 메인 파이프라인 통합 | 전체 흐름 | `main.py` |
| 결과 내보내기 | CSV/시각화 | `result_exporter.py` |
| 전체 테스트 | 데이터셋 처리 | `test_full_pipeline.py` |

---

## 8. 예상 결과 및 검증 방법

### 8.1 출력 데이터 형식

#### CSV 출력 예시

```csv
frame_idx,timestamp,dx,dy,dz,vx,vy,vz,speed,roll_change,pitch_change,yaw_change,confidence
0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.95
1,0.033,0.001,-0.002,0.003,0.03,-0.06,0.09,0.115,0.5,-1.2,0.3,0.92
2,0.067,0.003,-0.005,0.008,0.06,-0.09,0.15,0.189,1.1,-2.5,0.8,0.91
...
```

### 8.2 검증 방법

#### A. 속도 추정 검증

```python
# 알려진 속도로 이동하는 객체 시뮬레이션
def validate_velocity_estimation():
    """속도 추정 검증"""
    # 1. 일정 속도로 이동하는 가상 데이터 생성
    true_velocity = np.array([0.1, 0.0, 0.05])  # m/s
    dt = 1/30.0

    kf = KalmanFilter3D(dt=dt)

    errors = []
    for t in range(100):
        # 실제 위치 (노이즈 추가)
        true_position = true_velocity * t * dt
        measured_position = true_position + np.random.normal(0, 0.01, 3)

        # Kalman Filter 업데이트
        _, est_velocity, _ = kf.predict_and_update(measured_position)

        # 오차 계산
        error = np.linalg.norm(est_velocity - true_velocity)
        errors.append(error)

    print(f"Velocity RMSE: {np.sqrt(np.mean(np.array(errors)**2)):.4f} m/s")
```

#### B. 각도 추정 검증

```python
# 알려진 자세의 평면 Point Cloud로 검증
def validate_orientation_estimation():
    """각도 추정 검증"""
    estimator = OrientationEstimator()

    # 45도 기울어진 평면 생성
    true_pitch = 45.0
    n_points = 1000

    # X-Y 평면에 점 생성 후 회전
    x = np.random.uniform(-0.5, 0.5, n_points)
    y = np.random.uniform(-0.5, 0.5, n_points)
    z = np.zeros(n_points)

    # Pitch 회전 적용
    angle = np.radians(true_pitch)
    z_rot = -x * np.sin(angle) + z * np.cos(angle)
    x_rot = x * np.cos(angle) + z * np.sin(angle)

    points = np.column_stack([x_rot, y, z_rot])

    # 추정
    result = estimator.estimate_from_pointcloud(points)

    print(f"True pitch: {true_pitch}°, Estimated pitch: {result.pitch:.2f}°")
    print(f"Error: {abs(true_pitch - result.pitch):.2f}°")
```

### 8.3 예상 성능 지표

| 메트릭 | 예상 값 | 비고 |
|--------|---------|------|
| **속도 추정 오차** | ±5-10% | CA Kalman Filter |
| **위치 추정 오차** | D455 depth 오차에 의존 | 1m에서 ~5mm |
| **Roll 각도 오차** | ±3-5° | PCA/OBB |
| **Pitch 각도 오차** | ±3-5° | PCA/OBB |
| **Yaw 각도 오차** | ±2-4° | PCA/OBB |
| **처리 FPS** | 10-30 FPS | 하드웨어 의존 |

### 8.4 시각화 예시

```python
import matplotlib.pyplot as plt

def visualize_results(df: pd.DataFrame, output_path: str):
    """결과 시각화"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 1. 속도 변화
    axes[0].plot(df['timestamp'], df['speed'], 'b-', label='Speed')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Speed (m/s)')
    axes[0].set_title('Object Speed Over Time')
    axes[0].legend()
    axes[0].grid(True)

    # 2. 위치 변화
    axes[1].plot(df['timestamp'], df['dx'], 'r-', label='ΔX')
    axes[1].plot(df['timestamp'], df['dy'], 'g-', label='ΔY')
    axes[1].plot(df['timestamp'], df['dz'], 'b-', label='ΔZ')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Position Change (m)')
    axes[1].set_title('Position Change from Reference')
    axes[1].legend()
    axes[1].grid(True)

    # 3. 각도 변화
    axes[2].plot(df['timestamp'], df['roll_change'], 'r-', label='Roll')
    axes[2].plot(df['timestamp'], df['pitch_change'], 'g-', label='Pitch')
    axes[2].plot(df['timestamp'], df['yaw_change'], 'b-', label='Yaw')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Angle Change (°)')
    axes[2].set_title('Orientation Change from Reference')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
```

---

## 부록: 의존성 목록 (requirements.txt)

```
# Core
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
torch>=1.10.0

# YOLO
ultralytics>=8.0.0

# Kalman Filter
filterpy>=1.4.5

# Point Cloud Processing
open3d>=0.15.0

# Visualization
matplotlib>=3.4.0

# Optional
scipy>=1.7.0
pyyaml>=5.4.0
```

---

*작성: 2025-12-17*
*기반 문서: comprehensive_improvement_research_2025.md*
*목적: RGB + Depth + IMU 데이터 기반 상대적 속도/각도 변화량 측정 시스템 구현*
