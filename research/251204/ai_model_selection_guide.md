# AI 모델 선택 가이드 - D455 + Jetson Orin Nano Super 환경

**작성일**: 2025-12-04
**목적**: 속도/각도 측정 정확도 향상을 위한 최적 AI 모델 선택

---

## 목차

1. [하드웨어 제약 분석](#1-하드웨어-제약-분석)
2. [2D 객체 탐지 모델](#2-2d-객체-탐지-모델)
3. [3D 객체 탐지/추적 모델](#3-3d-객체-탐지추적-모델)
4. [속도 추정 모델](#4-속도-추정-모델)
5. [자세 추정 모델](#5-자세-추정-모델)
6. [추천 조합](#6-추천-조합)
7. [모델별 구현 가이드](#7-모델별-구현-가이드)

---

## 1. 하드웨어 제약 분석

### 1.1 Jetson Orin Nano Super 스펙

| 항목 | 사양 | 제약 영향 |
|------|------|----------|
| **AI 성능** | 67 TOPS (INT8) | 중형 모델까지 실시간 가능 |
| **GPU** | 1024 CUDA cores (Ampere) | TensorRT 최적화 필수 |
| **메모리** | 8GB LPDDR5 | 대형 모델 불가 |
| **TDP** | 25W (MAXN) | 열 관리 고려 |
| **JetPack** | 6.1+ | CUDA 12.6, TensorRT 10.3 |

### 1.2 메모리 예산 분석

```
총 가용 메모리: 8GB

필수 할당:
├── 시스템/OS: 1.5GB
├── CUDA/TensorRT: 1.0GB
├── 프레임 버퍼 (RGB+Depth): 0.1GB
└── Point Cloud 버퍼: 0.3GB
──────────────────────────────
기본 사용량: 2.9GB

모델용 가용 메모리: ~5.1GB
```

### 1.3 모델 크기 제한

| 모델 크기 | 메모리 (FP16) | 적합성 |
|-----------|---------------|--------|
| Nano (~3M params) | ~300MB | ★★★★★ |
| Small (~10M params) | ~500-700MB | ★★★★★ |
| Medium (~25M params) | ~1GB | ★★★★☆ |
| Large (~50M params) | ~2GB | ★★☆☆☆ |
| XLarge (~100M+ params) | ~4GB+ | ❌ 부적합 |

---

## 2. 2D 객체 탐지 모델

### 2.1 YOLO 시리즈 비교

| 모델 | FPS (TensorRT FP16) | mAP50 | 메모리 | 추천도 |
|------|---------------------|-------|--------|--------|
| **YOLO11n** | 100+ | 39.5 | ~400MB | ★★★★★ |
| **YOLO11s** | 55-60 | 47.0 | ~600MB | ★★★★★ |
| YOLO11m | 30-35 | 51.5 | ~1GB | ★★★★☆ |
| YOLOv8n | 120+ | 37.3 | ~350MB | ★★★★☆ |
| YOLOv8s | 60-65 | 44.9 | ~550MB | ★★★★☆ |
| YOLOv5s | 70-80 | 42.5 | ~500MB | ★★★★☆ |

### 2.2 최우선 추천: YOLO11s

**선택 이유**:
- 55-60 FPS로 실시간 처리 충분
- mAP50 47.0으로 높은 정확도
- ~600MB 메모리로 여유 있음
- TensorRT 10.3 완벽 지원
- 2024 최신 아키텍처

**구현**:
```python
from ultralytics import YOLO

# 모델 로드 및 TensorRT 변환
model = YOLO("yolo11s.pt")
model.export(format="engine", half=True, device=0)

# 추론
detector = YOLO("yolo11s.engine")

def detect(image):
    results = detector(image, conf=0.5, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                'bbox': box.xyxy[0].cpu().numpy(),
                'confidence': float(box.conf[0]),
                'class': int(box.cls[0]),
                'name': r.names[int(box.cls[0])]
            })
    return detections
```

### 2.3 대안: YOLOv8n (속도 우선)

FPS가 가장 중요한 경우:
- 120+ FPS 가능
- 정확도는 다소 낮음 (mAP50 37.3)

### 2.4 커스텀 모델 학습 권장

현재 `best.pt`가 있다면 동일 데이터로 YOLO11s 학습 권장:

```bash
yolo train data=your_dataset.yaml model=yolo11s.pt epochs=100 imgsz=640
```

---

## 3. 3D 객체 탐지/추적 모델

### 3.1 Full 3D 탐지기 (권장하지 않음)

| 모델 | 메모리 | FPS | Jetson 적합성 |
|------|--------|-----|---------------|
| CenterPoint (Full) | ~4GB | ~10 | ❌ 메모리 초과 |
| PointPillars (Full) | ~3GB | ~15 | ❌ 메모리 부족 |
| VoxelNet | ~5GB+ | ~5 | ❌ 불가 |

**결론**: Jetson Orin Nano Super에서 full 3D detector는 부적합

### 3.2 권장 접근법: 2D YOLO + Depth 기반 3D 변환

```python
def bbox_2d_to_3d(bbox_2d, depth_frame, intrinsics):
    """2D 탐지를 3D 위치로 변환"""
    x1, y1, x2, y2 = map(int, bbox_2d)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # 중심점 depth (median 사용으로 노이즈 감소)
    roi = depth_frame[max(0,cy-5):cy+5, max(0,cx-5):cx+5]
    z = np.median(roi[roi > 0]) * 0.001  # mm to m

    if z == 0:
        return None

    # 카메라 좌표계 3D 위치
    x = (cx - intrinsics.ppx) * z / intrinsics.fx
    y = (cy - intrinsics.ppy) * z / intrinsics.fy

    return np.array([x, y, z])
```

### 3.3 3D 추적: HVTrack 또는 경량 대안

#### HVTrack (ECCV 2024) - 고급 옵션

**특징**:
- High Temporal Variation 대응
- Point Cloud 형상 변화에 강건
- 복잡한 장면에서 유사 객체 구분

**제약**:
- 메모리 사용량 높음 (~2GB)
- 실시간 어려움 (~15 FPS)

**권장**: Phase 4 이후 고려

#### 경량 대안: 3D Kalman Filter + Hungarian Algorithm

```python
from scipy.optimize import linear_sum_assignment

class Simple3DTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_age = 10

    def update(self, detections_3d):
        """간단한 3D 추적"""
        if not self.tracks:
            for det in detections_3d:
                self._create_track(det)
            return

        # 비용 행렬 (3D 거리)
        cost = np.zeros((len(self.tracks), len(detections_3d)))
        track_ids = list(self.tracks.keys())

        for i, tid in enumerate(track_ids):
            pred = self.tracks[tid]['kf'].x[:3].flatten()
            for j, det in enumerate(detections_3d):
                cost[i, j] = np.linalg.norm(pred - det['position'])

        # Hungarian 매칭
        row_ind, col_ind = linear_sum_assignment(cost)

        # 매칭된 트랙 업데이트
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < 0.5:  # 50cm threshold
                self._update_track(track_ids[i], detections_3d[j])
```

---

## 4. 속도 추정 모델

### 4.1 방법 비교

| 방법 | 정확도 | 연산량 | 적합성 |
|------|--------|--------|--------|
| **Kalman Filter** | 중-상 | 매우 낮음 | ★★★★★ |
| Optical Flow + Depth | 중 | 낮음 | ★★★★☆ |
| Scene Flow (DL) | 상 | 높음 | ★★★☆☆ |
| CenterPoint 속도 헤드 | 상 | 매우 높음 | ★★☆☆☆ |

### 4.2 최우선 추천: Constant Acceleration Kalman Filter

```python
from filterpy.kalman import KalmanFilter
import numpy as np

def create_ca_kalman_filter(dt=1/30.0):
    """Constant Acceleration 모델 Kalman Filter"""
    kf = KalmanFilter(dim_x=9, dim_z=3)

    # 상태: [x, y, z, vx, vy, vz, ax, ay, az]
    # 상태 전이 행렬 (Constant Acceleration)
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

    # 측정 행렬 (위치만 측정)
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0]
    ])

    # D455 측정 노이즈 (1m에서 ~5mm)
    kf.R = np.eye(3) * 0.005**2

    # 프로세스 노이즈
    q = 0.1
    kf.Q = np.eye(9) * q

    return kf

class VelocityEstimator:
    def __init__(self, dt=1/30.0):
        self.kf = create_ca_kalman_filter(dt)
        self.initialized = False

    def update(self, position_3d):
        if not self.initialized:
            self.kf.x[:3] = position_3d.reshape(3, 1)
            self.initialized = True
            return np.zeros(3), 0.0

        self.kf.predict()
        self.kf.update(position_3d)

        velocity = self.kf.x[3:6].flatten()
        speed = np.linalg.norm(velocity)

        return velocity, speed
```

### 4.3 고급 옵션: Scene Flow (FlowFormer)

**권장 상황**:
- 다중 객체 속도 필요
- 높은 정확도 필요
- 충분한 연산 여유 (20+ FPS 목표)

**제약**:
- 추가 모델 로드 필요 (~1GB)
- 처리 시간 증가 (~30ms)

```python
# FlowFormer 통합 (개념적 코드)
class SceneFlowVelocity:
    def __init__(self, model_path):
        self.model = load_flowformer(model_path)
        self.prev_pcd = None

    def estimate(self, current_pcd, dt):
        if self.prev_pcd is None:
            self.prev_pcd = current_pcd
            return None

        # Scene Flow 추정
        flow = self.model(self.prev_pcd, current_pcd)

        # 평균 속도 계산
        velocity = np.mean(flow, axis=0) / dt

        self.prev_pcd = current_pcd
        return velocity
```

---

## 5. 자세 추정 모델

### 5.1 방법 비교

| 방법 | Roll | Pitch | Yaw | 연산량 | 적합성 |
|------|------|-------|-----|--------|--------|
| **PCA/OBB** | ✓ | ✓ | ✓ | 매우 낮음 | ★★★★★ |
| Depth 기울기 | △ | ✓ | △ | 매우 낮음 | ★★★★☆ |
| ICP Registration | ✓ | ✓ | ✓ | 중간 | ★★★☆☆ |
| DL 6DoF Pose | ✓ | ✓ | ✓ | 높음 | ★★☆☆☆ |

### 5.2 최우선 추천: PCA 기반 OBB

```python
import numpy as np
import open3d as o3d

class OrientationEstimator:
    def __init__(self):
        self.prev_orientation = None
        self.alpha = 0.7  # smoothing factor

    def estimate(self, points):
        """Point Cloud에서 6DoF 방향 추정"""
        if len(points) < 10:
            return None

        # 중심 계산
        center = np.mean(points, axis=0)
        centered = points - center

        # PCA
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 주축 정렬 (큰 순서)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # 회전 행렬 → Euler 각도
        R = eigenvectors
        euler = self._rotation_to_euler(R)

        # Temporal Smoothing
        if self.prev_orientation is not None:
            euler = self.alpha * euler + (1 - self.alpha) * self.prev_orientation
        self.prev_orientation = euler

        return {
            'roll': euler[0],
            'pitch': euler[1],
            'yaw': euler[2],
            'center': center,
            'dimensions': np.sqrt(eigenvalues) * 2
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
```

### 5.3 간단한 대안: Depth 기반 기울기

Roll/Pitch만 필요한 경우:

```python
def estimate_tilt_from_depth(depth_roi):
    """Depth ROI에서 기울기 추정 (Pitch, Roll)"""
    h, w = depth_roi.shape

    # Pitch: 상단/하단 depth 차이
    top = np.nanmean(depth_roi[:h//3, :])
    bottom = np.nanmean(depth_roi[2*h//3:, :])
    pitch = np.arctan2(bottom - top, h * 0.001)  # 대략적 계산

    # Roll: 좌측/우측 depth 차이
    left = np.nanmean(depth_roi[:, :w//3])
    right = np.nanmean(depth_roi[:, 2*w//3:])
    roll = np.arctan2(right - left, w * 0.001)

    return np.degrees(pitch), np.degrees(roll)
```

### 5.4 고급 옵션: ICP Registration

템플릿 매칭이 필요한 경우:

```python
import open3d as o3d

def estimate_pose_icp(source_pcd, target_template):
    """ICP 기반 정밀 자세 추정"""
    # 다운샘플링
    voxel_size = 0.005  # 5mm
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_template.voxel_down_sample(voxel_size)

    # Normal 계산
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )

    # Point-to-Plane ICP
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, 0.02, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )

    return result.transformation
```

---

## 6. 추천 조합

### 6.1 기본 구성 (권장)

**목표**: 30 FPS, 안정적 성능

| 기능 | 모델/방법 | 예상 시간 |
|------|----------|----------|
| 객체 탐지 | YOLO11s (TensorRT FP16) | 15-18ms |
| 3D 변환 | Depth Lookup | 2ms |
| 속도 추정 | CA Kalman Filter | 1ms |
| 각도 추정 | PCA/OBB | 3-5ms |

**총 처리 시간**: ~25ms (40 FPS)

### 6.2 속도 우선 구성

**목표**: 50+ FPS

| 기능 | 모델/방법 | 예상 시간 |
|------|----------|----------|
| 객체 탐지 | YOLO11n (TensorRT FP16) | 8-10ms |
| 3D 변환 | Depth Lookup | 2ms |
| 속도 추정 | CV Kalman Filter | 0.5ms |
| 각도 추정 | Depth 기울기 | 1ms |

**총 처리 시간**: ~13ms (70+ FPS)

### 6.3 정확도 우선 구성

**목표**: 높은 측정 정확도

| 기능 | 모델/방법 | 예상 시간 |
|------|----------|----------|
| 객체 탐지 | YOLO11m (TensorRT FP16) | 25-30ms |
| 3D 변환 | ROI Point Cloud | 5ms |
| 속도 추정 | CA Kalman + Multi-sweep | 3ms |
| 각도 추정 | PCA + ICP Refinement | 10-15ms |

**총 처리 시간**: ~50ms (20 FPS)

---

## 7. 모델별 구현 가이드

### 7.1 YOLO11 설치 및 최적화

```bash
# JetPack 6.1+ 환경에서

# 1. Ultralytics 설치
pip install ultralytics

# 2. CUDA 확인
python3 -c "import torch; print(torch.cuda.is_available())"

# 3. TensorRT 엔진 생성
python3 << 'EOF'
from ultralytics import YOLO

# 모델 다운로드 및 변환
model = YOLO("yolo11s.pt")
model.export(
    format="engine",
    half=True,        # FP16
    device=0,
    workspace=4,      # GB
    batch=1,
    dynamic=False,
    simplify=True
)
print("TensorRT 엔진 생성 완료!")
EOF
```

### 7.2 FilterPy 설치

```bash
pip install filterpy
```

### 7.3 Open3D 설치 (Jetson용)

```bash
# Jetson 전용 빌드 필요
pip install open3d --no-cache-dir

# 또는 소스 빌드
git clone https://github.com/isl-org/Open3D.git
cd Open3D
mkdir build && cd build
cmake -DBUILD_CUDA_MODULE=ON ..
make -j4
```

### 7.4 통합 코드 예시

```python
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import numpy as np
import pyrealsense2 as rs

class IntegratedMeasurement:
    def __init__(self):
        # YOLO 탐지기
        self.detector = YOLO("yolo11s.engine")

        # Kalman Filter
        self.velocity_est = VelocityEstimator(dt=1/30.0)

        # 방향 추정기
        self.orientation_est = OrientationEstimator()

        # D455 센서
        self.setup_d455()

    def setup_d455(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # 카메라 내부 파라미터
        profile = self.pipeline.get_active_profile()
        self.intrinsics = profile.get_stream(rs.stream.depth)\
                                 .as_video_stream_profile()\
                                 .get_intrinsics()

    def process_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 1. YOLO 탐지
        detections = self.detector(color_image, conf=0.5, verbose=False)

        results = []
        for det in detections[0].boxes:
            bbox = det.xyxy[0].cpu().numpy()

            # 2. 3D 위치
            pos_3d = bbox_2d_to_3d(bbox, depth_image, self.intrinsics)
            if pos_3d is None:
                continue

            # 3. 속도 추정
            velocity, speed = self.velocity_est.update(pos_3d)

            # 4. Point Cloud 추출 및 방향 추정
            points = self.extract_object_points(depth_frame, bbox)
            orientation = self.orientation_est.estimate(points)

            results.append({
                'bbox': bbox,
                'position': pos_3d,
                'velocity': velocity,
                'speed': speed,
                'orientation': orientation
            })

        return results
```

---

## 결론

### 최종 추천

1. **2D 탐지**: **YOLO11s** (TensorRT FP16)
   - 균형 잡힌 속도/정확도
   - 55-60 FPS 가능

2. **속도 추정**: **Constant Acceleration Kalman Filter**
   - 즉시 구현 가능
   - 노이즈 필터링 내장
   - 가속도까지 추정

3. **방향 추정**: **PCA/OBB**
   - 경량, 빠름 (~5ms)
   - 6DoF 모두 측정
   - Open3D로 쉽게 구현

4. **3D 변환**: **Depth Lookup + Median 필터**
   - 단순하고 효과적
   - 노이즈 감소

### 피해야 할 모델

- CenterPoint (Full): 메모리 초과
- PointPillars (Full): 성능 부족
- 대형 Transformer: 메모리/속도 문제
- 복잡한 Scene Flow 모델: 실시간 불가

---

**참고 자료**:
- [YOLO11 Guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- [FilterPy Documentation](https://filterpy.readthedocs.io/)
- [Open3D Tutorials](https://www.open3d.org/docs/latest/tutorial/)
- [Intel RealSense SDK](https://dev.intelrealsense.com/docs)
