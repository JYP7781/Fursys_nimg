# nimg 코드베이스 개선 권장사항 (D455 + Jetson Orin Nano Super 환경)

**작성일**: 2025-12-04
**대상 하드웨어**: Intel RealSense D455 + NVIDIA Jetson Orin Nano Super
**목적**: Depth Image 및 3D Point Cloud를 활용한 속도/각도 측정 정확도 개선

---

## 목차

1. [하드웨어 사양 분석](#1-하드웨어-사양-분석)
2. [D455 vs L515 비교 및 마이그레이션 가이드](#2-d455-vs-l515-비교-및-마이그레이션-가이드)
3. [Jetson Orin Nano Super 성능 분석](#3-jetson-orin-nano-super-성능-분석)
4. [최적화된 AI 모델 추천](#4-최적화된-ai-모델-추천)
5. [구현 권장사항](#5-구현-권장사항)
6. [예상 성능 및 제약사항](#6-예상-성능-및-제약사항)
7. [코드 마이그레이션 가이드](#7-코드-마이그레이션-가이드)
8. [참고 자료](#8-참고-자료)

---

## 1. 하드웨어 사양 분석

### 1.1 Intel RealSense D455 상세 사양

| 항목 | 사양 |
|------|------|
| **센서 타입** | Active Stereo (적외선 프로젝터 + 스테레오 센서) |
| **Depth 해상도** | 최대 1280 x 720 @ 90fps |
| **RGB 해상도** | 1920 x 1080 @ 30fps (Global Shutter) |
| **Depth 정확도** | < 2% @ 4m |
| **최적 동작 범위** | 0.4m ~ 6m (실외 가능) |
| **최대 범위** | ~20m (환경 의존) |
| **베이스라인** | 95mm (D435 대비 2배) |
| **FOV (Depth)** | 86° x 57° |
| **FOV (RGB)** | 86° x 57° |
| **IMU** | 6축 (가속도계 + 자이로스코프) |
| **USB** | USB 3.1 Gen 1 Type-C |
| **SDK** | Intel RealSense SDK 2.0 (pyrealsense2) |

### 1.2 NVIDIA Jetson Orin Nano Super 상세 사양

| 항목 | 사양 |
|------|------|
| **AI 성능** | 67 TOPS (INT8) |
| **GPU** | 1024 CUDA cores (Ampere), 32 Tensor Cores |
| **GPU 클럭** | 1020 MHz |
| **CPU** | 6-core Arm Cortex-A78AE @ 1.7 GHz |
| **메모리** | 8GB LPDDR5 (128-bit) |
| **메모리 대역폭** | 102 GB/s |
| **스토리지** | NVMe SSD 지원 |
| **TDP** | 7W ~ 25W (설정 가능) |
| **JetPack** | 6.x (CUDA 12.6, TensorRT 10.3) |
| **가격** | $249 USD |

### 1.3 하드웨어 조합의 장점

1. **실내/실외 호환**: D455는 L515와 달리 실외에서도 사용 가능
2. **장거리 지원**: 최대 6m까지 정확한 depth 측정 (L515는 실내 9m)
3. **IMU 내장**: 6DoF 관성 데이터로 카메라 움직임 보정 가능
4. **에너지 효율**: Orin Nano Super는 67 TOPS를 25W 이내에서 제공
5. **비용 효율**: 총 하드웨어 비용 ~$600 (D455 $349 + Orin Nano Super $249)

---

## 2. D455 vs L515 비교 및 마이그레이션 가이드

### 2.1 핵심 차이점

| 특성 | L515 (LiDAR) | D455 (Stereo) | 영향 |
|------|-------------|---------------|------|
| **기술** | Time-of-Flight | Active Stereo | 측정 원리 다름 |
| **Depth 정확도 (1m)** | 2.5-5mm | ~5-10mm | D455가 약간 낮음 |
| **Depth 정확도 (4m)** | N/A (실내 한정) | < 2% (~80mm) | 장거리에서 D455 유리 |
| **실외 사용** | 불가 | 가능 | D455 더 유연함 |
| **반사면/투명면** | 노이즈 많음 | 텍스처 필요 | 상황별 차이 |
| **제품 상태** | 단종 (2022) | 현재 판매 중 | D455가 장기 지원 |
| **IMU** | 없음 | 있음 (6축) | D455가 움직임 보정 가능 |

### 2.2 D455 특성에 따른 고려사항

#### 2.2.1 Depth 정확도
```
D455 Depth 오차 공식:
Z_error ≈ Z² / (baseline × focal_length)

베이스라인 95mm로 확장되어:
- 1m에서: ~5mm (0.5%) 오차
- 2m에서: ~20mm (1%) 오차
- 4m에서: ~80mm (2%) 오차
```

**대응 전략**:
- 작업 거리를 1-3m로 최적화
- Kalman Filter로 노이즈 필터링 강화
- 시간 축 평균화로 정확도 향상

#### 2.2.2 텍스처 의존성
Active Stereo는 텍스처가 없는 단색 표면에서 성능 저하가 발생할 수 있습니다.

**대응 전략**:
- 적외선 프로젝터 패턴으로 대부분 해결됨
- 고반사 표면에서는 RGB 정보 가중치 증가
- Post-processing 필터 적용 (Spatial, Temporal)

### 2.3 코드 마이그레이션 체크리스트

#### nodeL515.py → nodeD455.py 변경사항

```python
# 변경 전 (L515)
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# 변경 후 (D455)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# IMU 스트림 추가 (D455 전용)
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
```

#### D455 전용 필터 설정

```python
# D455 최적화 Post-Processing 필터 체인
decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 2)  # 해상도 1/2

spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)

temporal = rs.temporal_filter()
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal.set_option(rs.option.filter_smooth_delta, 20)

hole_filling = rs.hole_filling_filter()

# 필터 적용 순서
def post_process_depth_frame(depth_frame):
    depth_frame = decimation.process(depth_frame)
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)
    return depth_frame
```

#### IMU 데이터 활용

```python
def process_imu(self, accel_frame, gyro_frame):
    """D455 IMU 데이터 처리"""
    accel_data = accel_frame.as_motion_frame().get_motion_data()
    gyro_data = gyro_frame.as_motion_frame().get_motion_data()

    # 가속도 (m/s²)
    accel = np.array([accel_data.x, accel_data.y, accel_data.z])

    # 각속도 (rad/s)
    gyro = np.array([gyro_data.x, gyro_data.y, gyro_data.z])

    return accel, gyro

# IMU로 카메라 움직임 감지 → depth 보정에 활용
def is_camera_stable(self, accel, gyro, threshold=0.1):
    """카메라가 정지 상태인지 확인"""
    accel_magnitude = np.linalg.norm(accel - np.array([0, 9.81, 0]))  # 중력 제거
    gyro_magnitude = np.linalg.norm(gyro)

    return accel_magnitude < threshold and gyro_magnitude < 0.05
```

---

## 3. Jetson Orin Nano Super 성능 분석

### 3.1 AI 추론 벤치마크

| 모델 | Precision | FPS (TensorRT) | 메모리 사용 |
|------|-----------|----------------|-------------|
| YOLOv5s | FP16 | ~80 | ~500MB |
| YOLOv5m | FP16 | ~45 | ~700MB |
| YOLOv8n | FP16 | ~120 | ~400MB |
| YOLOv8s | FP16 | ~60 | ~600MB |
| YOLOv8m | FP16 | ~35 | ~900MB |
| YOLO11n | FP16 | ~100+ | ~450MB |
| YOLO11s | FP16 | ~55 | ~650MB |

**참고**: [Ultralytics YOLO11 on Jetson Orin Nano Super](https://www.ultralytics.com/blog/ultralytics-yolo11-on-nvidia-jetson-orin-nano-super-fast-and-efficient)

### 3.2 3D 처리 성능 예상

| 처리 | 예상 시간 | 비고 |
|------|-----------|------|
| Point Cloud 생성 (640x480) | ~10ms | pyrealsense2 |
| Voxel Downsampling (5mm) | ~5ms | Open3D |
| ICP Registration | ~30-50ms | Open3D, CPU 기반 |
| OBB/PCA 계산 | ~3ms | NumPy/Open3D |
| Kalman Filter 업데이트 | ~0.5ms | FilterPy |

### 3.3 메모리 예산 분석

```
총 가용 메모리: 8GB

예상 메모리 사용:
- 시스템/OS: ~1.5GB
- CUDA/TensorRT: ~1GB
- YOLO 모델 (FP16): ~0.6GB
- RGB Frame (1280x720x3): ~2.7MB
- Depth Frame (1280x720x2): ~1.8MB
- Point Cloud (약 100K points): ~1.2MB
- 버퍼/기타: ~1GB
----------------------------------------
총 예상 사용: ~5-6GB
여유 메모리: ~2-3GB
```

### 3.4 전력 모드 설정

```bash
# Jetson Orin Nano Super 전력 모드 설정

# 최대 성능 모드 (25W) - 권장
sudo nvpmodel -m 0
sudo jetson_clocks

# 전력 효율 모드 (15W)
sudo nvpmodel -m 1

# 현재 모드 확인
nvpmodel -q
```

---

## 4. 최적화된 AI 모델 추천

### 4.1 Jetson Orin Nano Super에 적합한 모델

Orin Nano Super의 67 TOPS 성능과 8GB 메모리를 고려할 때, **경량 모델** 위주로 선정해야 합니다.

#### 4.1.1 2D 객체 탐지 (현재 시스템 개선)

| 모델 | 추천도 | 이유 |
|------|--------|------|
| **YOLOv8n/YOLOv8s** | ★★★★★ | 최적 균형, TensorRT 완벽 지원 |
| YOLO11n/YOLO11s | ★★★★★ | 2024 최신, 높은 효율 |
| YOLOv5s | ★★★★☆ | 안정적, 검증됨 |
| EfficientDet-Lite | ★★★☆☆ | 경량, TFLite 필요 |

**권장**: **YOLOv8s** 또는 **YOLO11s** (TensorRT FP16)
- 60+ FPS 실시간 처리
- ~600MB 메모리 사용
- 높은 정확도 유지

#### 4.1.2 3D 객체 탐지/추적 (고급 기능)

**중요**: CenterPoint, PointPillars 같은 full 3D detector는 Orin Nano Super에서 **권장하지 않음**

| 모델 | 적합성 | 이유 |
|------|--------|------|
| CenterPoint (Full) | ❌ | 메모리/연산 과다 |
| PointPillars (Full) | ❌ | 메모리/연산 과다 |
| **PointPillars (Lite)** | △ | TAO Toolkit 경량 버전 가능 |

**대안 접근법**: 2D YOLO + Depth 기반 3D 위치 추정
```python
# 2D 탐지 → 3D 위치 변환 (경량 접근법)
def bbox_2d_to_3d(bbox_2d, depth_frame, intrinsics):
    """2D 바운딩 박스를 3D 위치로 변환"""
    x1, y1, x2, y2 = bbox_2d
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    # 중심점 depth
    z = depth_frame[int(cy), int(cx)] * 0.001  # mm to m

    # 카메라 내부 파라미터
    fx, fy = intrinsics.fx, intrinsics.fy
    ppx, ppy = intrinsics.ppx, intrinsics.ppy

    # 3D 위치 계산
    x = (cx - ppx) * z / fx
    y = (cy - ppy) * z / fy

    return np.array([x, y, z])
```

#### 4.1.3 자세 추정

| 방법 | 적합성 | 성능 | 추천도 |
|------|--------|------|--------|
| **OBB + PCA** (Open3D) | ★★★★★ | ~5ms | 최우선 |
| ICP Registration | ★★★★☆ | ~40ms | 정밀 필요시 |
| Depth 기반 기울기 계산 | ★★★★☆ | ~2ms | 간단한 경우 |
| 6DoF DL 모델 | ★★☆☆☆ | 느림 | 메모리 부족 |

**권장**: **OBB/PCA 기반 각도 측정** (즉시 적용 가능, 경량)

```python
import open3d as o3d
import numpy as np

def estimate_orientation_pca(points):
    """PCA 기반 객체 방향 추정 (경량)"""
    # 중심 계산
    center = np.mean(points, axis=0)
    centered = points - center

    # 공분산 행렬
    cov = np.cov(centered.T)

    # 고유값 분해
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 주축 방향 (가장 큰 고유값에 대응)
    idx = np.argsort(eigenvalues)[::-1]
    principal_axes = eigenvectors[:, idx]

    # Euler 각도 추출
    R = principal_axes
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.degrees([roll, pitch, yaw])
```

### 4.2 추천 솔루션 조합

```
[최종 추천 파이프라인 - Jetson Orin Nano Super]

┌─────────────────────────────────────────────────────────────┐
│                    Intel RealSense D455                      │
│            RGB (1280x720) + Depth + IMU                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Pre-processing (~15ms)                         │
│  • Post-processing Filters (Spatial, Temporal, Hole-fill)   │
│  • RGB-Depth Alignment                                       │
│  • IMU 데이터 수집                                           │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│   YOLO11s/YOLOv8s       │     │  Point Cloud Generation     │
│   (TensorRT FP16)       │     │  (Object ROI Only)          │
│   ~60 FPS, ~15ms        │     │  ~10ms                      │
└─────────────────────────┘     └─────────────────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│   2D → 3D Position      │     │  PCA/OBB Orientation        │
│   (Depth Lookup)        │     │  ~5ms                       │
│   ~2ms                  │     └─────────────────────────────┘
└─────────────────────────┘                   │
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                3D Kalman Filter                              │
│    • 위치, 속도, 가속도 추정                                 │
│    • 노이즈 필터링                                           │
│    • ~1ms                                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Output: Position, Velocity, Orientation         │
│              총 지연시간: ~45-50ms (~20-22 FPS)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 구현 권장사항

### 5.1 Phase 1: 기반 코드 개선 (즉시 적용)

#### 5.1.1 파일 I/O 제거
```python
# 현재 (비효율적)
cv2.imwrite('source.png', src_img)
result = self.dProcessor.detect('source.png')

# 개선 (메모리 기반)
class Detector:
    def detect_from_array(self, img_array):
        """이미지 배열을 직접 입력으로 받는 메서드"""
        # 전처리
        img = self.preprocess(img_array)
        # 추론
        pred = self.model(img)
        # 후처리
        results = self.postprocess(pred)
        return results
```

#### 5.1.2 D455 센서 클래스 생성
```python
# nimg/submodules/nodeD455.py
import pyrealsense2 as rs
import numpy as np

class D455Sensor:
    def __init__(self, width=1280, height=720, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 스트림 설정
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
        self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        # 필터 초기화
        self._init_filters()

        # Point Cloud
        self.pc = rs.pointcloud()

    def _init_filters(self):
        """D455 최적화 필터 설정"""
        self.align = rs.align(rs.stream.color)

        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 2)

        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)

        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)

        self.hole_filling = rs.hole_filling_filter()

    def start(self):
        profile = self.pipeline.start(self.config)
        depth_sensor = profile.get_device().first_depth_sensor()

        # Auto-exposure 활성화
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

        # Intrinsics 저장
        depth_stream = profile.get_stream(rs.stream.depth)
        self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

    def get_frames(self):
        """프레임 획득 및 전처리"""
        frames = self.pipeline.wait_for_frames()

        # Align
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        # IMU
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        # Depth 필터링
        depth_frame = self._filter_depth(depth_frame)

        # NumPy 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return {
            'color': color_image,
            'depth': depth_image,
            'depth_frame': depth_frame,
            'color_frame': color_frame,
            'accel': accel_frame,
            'gyro': gyro_frame
        }

    def _filter_depth(self, depth_frame):
        """Post-processing 필터 체인"""
        depth_frame = self.decimation.process(depth_frame)
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)
        return depth_frame

    def get_point_cloud(self, depth_frame, color_frame, roi=None):
        """Point Cloud 생성 (ROI 지원)"""
        self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)

        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

        # ROI 필터링 (선택적)
        if roi is not None:
            mask = self._get_roi_mask(vtx, roi)
            vtx = vtx[mask]
            tex = tex[mask]

        return vtx, tex

    def stop(self):
        self.pipeline.stop()
```

### 5.2 Phase 2: 3D 측정 구현

#### 5.2.1 3D Kalman Filter 통합
```python
# nimg/submodules/tracker3d.py
from filterpy.kalman import KalmanFilter
import numpy as np

class Object3DTracker:
    def __init__(self, dt=1/30.0):
        self.dt = dt
        self.tracks = {}
        self.next_id = 0

    def create_kalman_filter(self):
        """9-state Kalman Filter: [x,y,z, vx,vy,vz, ax,ay,az]"""
        kf = KalmanFilter(dim_x=9, dim_z=3)

        # State transition (constant acceleration)
        dt = self.dt
        kf.F = np.array([
            [1, 0, 0, dt, 0,  0,  0.5*dt**2, 0,         0        ],
            [0, 1, 0, 0,  dt, 0,  0,         0.5*dt**2, 0        ],
            [0, 0, 1, 0,  0,  dt, 0,         0,         0.5*dt**2],
            [0, 0, 0, 1,  0,  0,  dt,        0,         0        ],
            [0, 0, 0, 0,  1,  0,  0,         dt,        0        ],
            [0, 0, 0, 0,  0,  1,  0,         0,         dt       ],
            [0, 0, 0, 0,  0,  0,  1,         0,         0        ],
            [0, 0, 0, 0,  0,  0,  0,         1,         0        ],
            [0, 0, 0, 0,  0,  0,  0,         0,         1        ]
        ])

        # Measurement matrix (position only)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])

        # Measurement noise (D455 accuracy: ~10mm at 1m)
        kf.R = np.eye(3) * 0.01**2

        # Process noise
        kf.Q = np.eye(9) * 0.1

        return kf

    def update(self, position_3d):
        """단일 객체 추적 업데이트"""
        if 0 not in self.tracks:
            # 첫 번째 탐지: 새 트랙 생성
            kf = self.create_kalman_filter()
            kf.x[:3] = position_3d.reshape(3, 1)
            self.tracks[0] = {'kf': kf, 'age': 0}

        track = self.tracks[0]
        track['kf'].predict()
        track['kf'].update(position_3d)
        track['age'] += 1

        return {
            'position': track['kf'].x[:3].flatten(),
            'velocity': track['kf'].x[3:6].flatten(),
            'acceleration': track['kf'].x[6:9].flatten()
        }

    def get_speed(self):
        """현재 속도 크기 반환"""
        if 0 in self.tracks:
            velocity = self.tracks[0]['kf'].x[3:6].flatten()
            return np.linalg.norm(velocity)
        return 0.0
```

#### 5.2.2 PCA 기반 각도 측정
```python
# nimg/submodules/orientation_estimator.py
import numpy as np

class OrientationEstimator:
    def __init__(self):
        self.prev_orientation = None

    def estimate_from_points(self, points):
        """Point Cloud에서 객체 방향 추정"""
        if len(points) < 10:
            return None

        # 중심 계산
        center = np.mean(points, axis=0)
        centered = points - center

        # 공분산 행렬
        cov = np.cov(centered.T)

        # 고유값 분해
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 주축 정렬 (큰 순서)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # 회전 행렬 → Euler 각도
        R = eigenvectors
        euler = self._rotation_to_euler(R)

        # Smoothing (이전 값과 평균)
        if self.prev_orientation is not None:
            euler = 0.7 * euler + 0.3 * self.prev_orientation
        self.prev_orientation = euler

        return {
            'roll': euler[0],
            'pitch': euler[1],
            'yaw': euler[2],
            'center': center,
            'dimensions': np.sqrt(eigenvalues) * 2  # 대략적 크기
        }

    def _rotation_to_euler(self, R):
        """회전 행렬 → Euler 각도 (도)"""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        return np.degrees([roll, pitch, yaw])

    def estimate_from_depth(self, depth_roi, intrinsics):
        """Depth 이미지에서 간단한 기울기 추정 (경량)"""
        h, w = depth_roi.shape

        # 상단/하단 평균 depth
        top_mean = np.nanmean(depth_roi[:h//3, :])
        bottom_mean = np.nanmean(depth_roi[2*h//3:, :])

        # 좌측/우측 평균 depth
        left_mean = np.nanmean(depth_roi[:, :w//3])
        right_mean = np.nanmean(depth_roi[:, 2*w//3:])

        # 기울기 계산 (간단한 근사)
        pitch_angle = np.arctan2(bottom_mean - top_mean, h * 0.001)  # mm to m
        roll_angle = np.arctan2(right_mean - left_mean, w * 0.001)

        return {
            'pitch': np.degrees(pitch_angle),
            'roll': np.degrees(roll_angle),
            'yaw': 0  # 2D depth에서는 yaw 추정 어려움
        }
```

### 5.3 Phase 3: TensorRT 최적화

#### 5.3.1 YOLO 모델 TensorRT 변환
```bash
# Jetson Orin Nano Super에서 실행

# 1. Ultralytics 설치
pip install ultralytics

# 2. YOLO 모델 다운로드 및 TensorRT 변환
python3 << 'EOF'
from ultralytics import YOLO

# YOLOv8s 모델 로드
model = YOLO("yolov8s.pt")

# TensorRT FP16으로 내보내기
model.export(format="engine", half=True, device=0)

print("TensorRT 엔진 생성 완료: yolov8s.engine")
EOF

# 3. 또는 YOLO11 사용
python3 << 'EOF'
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.export(format="engine", half=True, device=0)
EOF
```

#### 5.3.2 TensorRT 추론 코드
```python
# nimg/submodules/detector_tensorrt.py
from ultralytics import YOLO
import cv2
import numpy as np

class YOLODetectorTRT:
    def __init__(self, model_path='yolov8s.engine'):
        """TensorRT 엔진 로드"""
        self.model = YOLO(model_path)

    def detect(self, image, conf_threshold=0.5):
        """객체 탐지 실행"""
        results = self.model(image, conf=conf_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = result.names[cls]

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class': cls,
                    'name': name
                })

        return detections
```

---

## 6. 예상 성능 및 제약사항

### 6.1 예상 성능

| 항목 | 현재 시스템 | 개선 후 (D455 + Orin Nano Super) |
|------|-------------|-----------------------------------|
| **객체 탐지 FPS** | ~15 (파일 I/O) | 55-60 (TensorRT) |
| **Depth 정확도** | L515 ~5mm@1m | D455 ~10mm@1m |
| **속도 측정** | 미구현 | ±5-10% 정확도 |
| **Pitch 각도** | ±15-20° | ±3-5° |
| **Yaw 각도** | ±10-15° | ±3-5° |
| **Roll 각도** | 미구현 | ±5-7° |
| **추적 성공률** | ~70% | >90% |
| **총 파이프라인 FPS** | ~10 | 20-25 |
| **지연 시간** | ~100ms | ~40-50ms |
| **전력 소비** | N/A | 15-25W |

### 6.2 D455 특성으로 인한 제약사항

| 제약사항 | 영향 | 대응 방안 |
|----------|------|-----------|
| 스테레오 기반 | 근거리(<0.4m) 정확도 저하 | 작업 거리 0.5m 이상 유지 |
| 텍스처 의존성 | 단색 표면 depth 노이즈 | IR 프로젝터 + 필터링 |
| 장거리 오차 증가 | 4m 이상에서 2%+ 오차 | 작업 범위 3m 이내 권장 |
| 햇빛 간섭 | 강한 직사광에서 성능 저하 | 실내 또는 그늘 환경 |

### 6.3 Orin Nano Super 제약사항

| 제약사항 | 영향 | 대응 방안 |
|----------|------|-----------|
| 8GB 메모리 | 대형 모델 실행 불가 | 경량 모델 사용 (YOLOv8s/YOLO11s) |
| CPU 6코어 | 멀티쓰레드 병렬 처리 한계 | GPU 활용 최대화 |
| TensorRT 호환성 | 일부 연산 미지원 | 지원 모델 사용 |
| 열 관리 | 장시간 고부하시 쓰로틀링 | 방열판/팬 추가 |

---

## 7. 코드 마이그레이션 가이드

### 7.1 파일 구조 변경

```
nimg/
├── nimg.py                          # 수정: D455 사용
├── submodules/
│   ├── nodeL515.py                  # 유지 (하위 호환)
│   ├── nodeD455.py                  # 신규: D455 전용 센서 클래스
│   ├── detectProcessor.py           # 수정: Point Cloud 활용
│   ├── detect.py                    # 유지
│   ├── detector_tensorrt.py         # 신규: TensorRT 탐지기
│   ├── tracker3d.py                 # 신규: 3D Kalman Filter
│   ├── orientation_estimator.py     # 신규: 각도 측정
│   └── ...
└── config/
    └── d455_config.yaml             # 신규: D455 설정
```

### 7.2 설정 파일 예시

```yaml
# config/d455_config.yaml

camera:
  type: "D455"
  depth:
    width: 1280
    height: 720
    fps: 30
  color:
    width: 1280
    height: 720
    fps: 30
  imu:
    accel_fps: 200
    gyro_fps: 200

filters:
  decimation:
    enabled: true
    magnitude: 2
  spatial:
    enabled: true
    magnitude: 2
    smooth_alpha: 0.5
    smooth_delta: 20
  temporal:
    enabled: true
    smooth_alpha: 0.4
    smooth_delta: 20
  hole_filling:
    enabled: true

detection:
  model: "yolov8s.engine"
  conf_threshold: 0.5
  input_size: 640

tracking:
  kalman:
    process_noise: 0.1
    measurement_noise: 0.01  # D455 정확도 반영
  max_age: 30
  min_hits: 3

roi:
  x: 0
  y: 0
  width: 1280
  height: 720
  min_depth: 0.4
  max_depth: 3.0
```

### 7.3 메인 노드 수정 예시

```python
# nimg/nimg.py 수정 버전

from nimg.submodules.nodeD455 import D455Sensor
from nimg.submodules.detector_tensorrt import YOLODetectorTRT
from nimg.submodules.tracker3d import Object3DTracker
from nimg.submodules.orientation_estimator import OrientationEstimator

class nimg_x86(Node):
    def __init__(self):
        super().__init__('nimg_x86')

        # 설정 로드
        self.config = self.load_config('config/d455_config.yaml')

        # D455 센서 초기화
        self.sensor = D455Sensor(
            width=self.config['camera']['depth']['width'],
            height=self.config['camera']['depth']['height'],
            fps=self.config['camera']['depth']['fps']
        )

        # TensorRT 탐지기
        self.detector = YOLODetectorTRT(self.config['detection']['model'])

        # 3D 추적기
        self.tracker = Object3DTracker(dt=1/30.0)

        # 각도 추정기
        self.orientation_est = OrientationEstimator()

    def process_frame(self):
        """메인 처리 루프"""
        frames = self.sensor.get_frames()

        # 1. 객체 탐지
        detections = self.detector.detect(
            frames['color'],
            conf_threshold=self.config['detection']['conf_threshold']
        )

        if len(detections) == 0:
            return None

        # 2. 가장 신뢰도 높은 탐지 선택
        det = max(detections, key=lambda x: x['confidence'])
        bbox = det['bbox']

        # 3. 2D → 3D 위치 변환
        position_3d = self.bbox_to_3d(bbox, frames['depth'])

        # 4. Kalman Filter 업데이트 (속도 추정)
        state = self.tracker.update(position_3d)

        # 5. 각도 추정
        points = self.get_object_points(frames, bbox)
        orientation = self.orientation_est.estimate_from_points(points)

        return {
            'class': det['name'],
            'confidence': det['confidence'],
            'position': state['position'],
            'velocity': state['velocity'],
            'speed': np.linalg.norm(state['velocity']),
            'orientation': orientation
        }
```

---

## 8. 참고 자료

### 8.1 하드웨어 문서

- **Intel RealSense D455**
  - [Product Page](https://www.intelrealsense.com/depth-camera-d455/)
  - [Specifications](https://www.intel.com/content/www/us/en/products/sku/205847/intel-realsense-depth-camera-d455/specifications.html)
  - [D455 vs L515 Comparison Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8622561/)

- **NVIDIA Jetson Orin Nano Super**
  - [Product Page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
  - [Developer Blog](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/)
  - [JetPack 6 Documentation](https://docs.nvidia.com/jetson/)

### 8.2 소프트웨어 및 라이브러리

- **Intel RealSense SDK 2.0**: https://github.com/IntelRealSense/librealsense
- **pyrealsense2 Python Wrapper**: https://pypi.org/project/pyrealsense2/
- **Ultralytics YOLO**: https://docs.ultralytics.com/guides/nvidia-jetson/
- **NVIDIA TensorRT**: https://developer.nvidia.com/tensorrt
- **Open3D**: https://www.open3d.org/
- **FilterPy**: https://filterpy.readthedocs.io/

### 8.3 벤치마크 및 성능 자료

- [NVIDIA Jetson Benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks)
- [Ultralytics YOLO11 on Jetson Orin Nano Super](https://www.ultralytics.com/blog/ultralytics-yolo11-on-nvidia-jetson-orin-nano-super-fast-and-efficient)
- [Run Your 3D Object Detector on NVIDIA Jetson Platforms](https://www.mdpi.com/1424-8220/23/8/4005)
- [Benchmarking Deep Learning Models on Edge Devices](https://arxiv.org/html/2409.16808v1)

### 8.4 관련 연구 논문

1. [Metrological Characterization of D415, D455, L515 RealSense Devices](https://www.mdpi.com/1424-8220/21/22/7770)
2. [A comprehensive survey of lightweight object detection models for edge devices](https://link.springer.com/article/10.1007/s10462-024-10877-1)
3. [Real-Time Object Detection and Tracking for Local Dynamic Map Generation](https://www.mdpi.com/2079-9292/13/5/811)

---

## 결론

### D455 + Jetson Orin Nano Super 환경에서의 핵심 권장사항

1. **경량 모델 사용 필수**: YOLOv8s 또는 YOLO11s (TensorRT FP16)
2. **2D+Depth 접근법**: 2D 탐지 → Depth 기반 3D 위치 계산 (CenterPoint 대신)
3. **필터링 강화**: D455의 스테레오 노이즈를 필터 체인으로 보완
4. **PCA/OBB 각도 측정**: 딥러닝 6DoF 대신 고전적 방법 (메모리 절약)
5. **Kalman Filter 필수**: 속도 추론 및 노이즈 필터링
6. **IMU 활용**: D455 내장 IMU로 카메라 움직임 보정

### 예상 최종 성능

| 메트릭 | 목표치 |
|--------|--------|
| 탐지 FPS | 55-60 |
| 총 파이프라인 FPS | 20-25 |
| 속도 측정 정확도 | ±5-10% |
| 각도 측정 정확도 | ±3-7° |
| 추적 성공률 | >90% |
| 지연 시간 | <50ms |
| 전력 소비 | 15-25W |

이 환경은 L515 + 고성능 PC 대비 일부 정확도 손실이 있지만, **실내/실외 호환성**, **장기 지원**, **비용 효율성** 측면에서 우수한 선택입니다.
