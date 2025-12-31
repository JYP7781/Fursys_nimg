# nimg 고급 개선 방안 - Intel RealSense D455 + Jetson Orin Nano Super

**작성일**: 2025-12-04
**환경**: Intel RealSense D455 + NVIDIA Jetson Orin Nano Super
**목적**: Depth Image 및 3D Point Cloud를 활용한 속도/각도 측정 정확도 개선

---

## 목차

1. [현재 코드 분석 요약](#1-현재-코드-분석-요약)
2. [추가 개선 가능 영역](#2-추가-개선-가능-영역)
3. [최신 AI 모델 및 기술 추천](#3-최신-ai-모델-및-기술-추천)
4. [D455 특화 최적화](#4-d455-특화-최적화)
5. [Jetson Orin Nano Super 최적화](#5-jetson-orin-nano-super-최적화)
6. [구현 우선순위 및 로드맵](#6-구현-우선순위-및-로드맵)
7. [예상 성능 지표](#7-예상-성능-지표)
8. [참고 자료](#8-참고-자료)

---

## 1. 현재 코드 분석 요약

### 1.1 코드베이스 구조

```
nimg/
├── nimg.py                    # 메인 ROS2 노드 (nimg_x86 클래스)
├── submodules/
│   ├── nodeL515.py            # dSensor 클래스 - L515 카메라 처리
│   ├── detectProcessor.py     # 객체 탐지 및 각도 측정
│   ├── detect.py              # YOLOv5 탐지기
│   ├── lineDetector.py        # Hough 변환 라인 검출
│   └── ItemList.py            # 탐지 객체 관리
├── models/                    # YOLOv5 모델 정의
└── utils/                     # 유틸리티 함수
```

### 1.2 주요 클래스 및 메서드

| 클래스 | 파일 위치 | 핵심 메서드 |
|--------|----------|------------|
| `nimg_x86` | nimg.py:46-247 | `spin()`, `start()`, `stop()` |
| `dSensor` | nodeL515.py:63-628 | `run()`, `move_check()`, `post_process_depth_frame()` |
| `detectProcessor` | detectProcessor.py:25-329 | `processImage()`, `depthProcess()`, `countingPattern()` |
| `Detector` | detect.py:56-236 | `detect()` |

### 1.3 기존 연구 문서에서 확인된 핵심 문제점

1. **속도 측정 미구현**: `speed = 0`으로 초기화만 됨
2. **단순 Depth 기반 각도 추정**: 상단/하단 depth 평균 차이만 계산
3. **Point Cloud 미활용**: 파일 저장용으로만 사용
4. **객체 추적 부재**: 프레임 간 연결 없음
5. **파일 I/O 병목**: `cv2.imwrite()`로 이미지 저장 후 탐지

---

## 2. 추가 개선 가능 영역

기존 연구 문서에서 다루지 않았거나 추가로 개선 가능한 영역:

### 2.1 IMU 데이터 미활용 (D455 전용)

D455는 6축 IMU(가속도계 + 자이로스코프)를 내장하고 있으나 현재 코드에서 활용되지 않음.

**개선 방안**:
```python
# D455 IMU 스트림 활성화
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

# IMU 데이터 융합
def process_imu_data(accel_frame, gyro_frame):
    """카메라 움직임 보정을 위한 IMU 데이터 처리"""
    accel = accel_frame.as_motion_frame().get_motion_data()
    gyro = gyro_frame.as_motion_frame().get_motion_data()

    # 카메라 안정성 확인
    accel_magnitude = np.sqrt(accel.x**2 + accel.y**2 + accel.z**2)
    is_stable = abs(accel_magnitude - 9.81) < 0.1

    return is_stable, accel, gyro
```

### 2.2 적응형 Kalman Filter 미적용

현재 시스템은 정적 노이즈 모델만 사용. D455의 depth 오차는 거리에 따라 변화함.

**개선 방안**: 거리 기반 측정 노이즈 조정
```python
def adaptive_measurement_noise(distance_m):
    """D455 거리 기반 적응형 측정 노이즈"""
    # D455 오차: Z_error ≈ Z² / (baseline × focal_length)
    # 베이스라인 95mm 기준
    base_error = 0.005  # 1m에서 5mm
    error = base_error * (distance_m ** 2)

    return np.eye(3) * (error ** 2)
```

### 2.3 Multi-Sweep Point Cloud 미사용

연속 프레임의 Point Cloud를 누적하여 노이즈 감소 및 정확도 향상 가능.

**개선 방안**:
```python
class MultiSweepPointCloud:
    def __init__(self, num_sweeps=3):
        self.num_sweeps = num_sweeps
        self.sweep_buffer = []

    def add_sweep(self, pcd):
        self.sweep_buffer.append(pcd)
        if len(self.sweep_buffer) > self.num_sweeps:
            self.sweep_buffer.pop(0)

    def get_accumulated_cloud(self):
        if len(self.sweep_buffer) < 2:
            return self.sweep_buffer[-1] if self.sweep_buffer else None

        # 모션 보정 및 누적
        accumulated = self.sweep_buffer[0]
        for pcd in self.sweep_buffer[1:]:
            accumulated = self.merge_clouds_with_motion_compensation(accumulated, pcd)

        return accumulated
```

### 2.4 ROI 기반 Point Cloud 처리 미최적화

현재 전체 Point Cloud를 처리하나, 탐지된 객체 영역만 처리하면 성능 향상 가능.

**개선 방안**:
```python
def extract_object_pointcloud(pcd, bbox_2d, depth_image, camera_intrinsics):
    """2D 바운딩 박스 영역의 Point Cloud만 추출"""
    x1, y1, x2, y2 = bbox_2d

    # 해당 영역의 depth 값
    roi_depth = depth_image[y1:y2, x1:x2]

    # 유효 depth 포인트만 추출
    valid_mask = roi_depth > 0

    # 3D 좌표 계산
    fx, fy = camera_intrinsics.fx, camera_intrinsics.fy
    cx, cy = camera_intrinsics.ppx, camera_intrinsics.ppy

    # ... 변환 로직
    return object_pcd
```

### 2.5 Temporal Smoothing 미적용

프레임별 각도/속도 측정값의 급격한 변화를 완화하는 시간적 평활화 미적용.

**개선 방안**:
```python
class TemporalSmoother:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.prev_value = None

    def smooth(self, current_value):
        if self.prev_value is None:
            self.prev_value = current_value
            return current_value

        smoothed = self.alpha * current_value + (1 - self.alpha) * self.prev_value
        self.prev_value = smoothed
        return smoothed
```

---

## 3. 최신 AI 모델 및 기술 추천

### 3.1 2D 객체 탐지: YOLO11 (2024 최신)

Ultralytics YOLO11은 Jetson Orin Nano Super에서 최적의 성능을 제공합니다.

| 모델 | TensorRT FP16 | 메모리 | 추천도 |
|------|---------------|--------|--------|
| **YOLO11n** | ~100+ FPS | ~400MB | ★★★★★ |
| **YOLO11s** | ~55 FPS | ~600MB | ★★★★★ |
| YOLO11m | ~30 FPS | ~1GB | ★★★☆☆ |

**설치 및 변환**:
```bash
# Ultralytics 설치
pip install ultralytics

# TensorRT 엔진 변환
python3 << 'EOF'
from ultralytics import YOLO
model = YOLO("yolo11s.pt")
model.export(format="engine", half=True, device=0)
EOF
```

**참고**: [YOLO11 Jetson Orin Nano Super](https://www.ultralytics.com/blog/ultralytics-yolo11-on-nvidia-jetson-orin-nano-super-fast-and-efficient)

### 3.2 3D 객체 추적: HVTrack (ECCV 2024)

높은 시간적 변동이 있는 Point Cloud에서 효과적인 3D Single Object Tracking.

**핵심 모듈**:
1. **Relative-Pose-Aware Memory (RPM)**: Point Cloud 형상 변화 처리
2. **Base-Expansion Feature Cross-Attention (BEA)**: 유사 객체 구분
3. **Contextual Point Guided Self-Attention (CPA)**: 포인트별 특징 강화

**참고**: [3D Single-Object Tracking in Point Clouds with High Temporal Variation](https://arxiv.org/html/2408.02049v3)

### 3.3 Scene Flow 추정: FlowFormer / SeFlow (2024)

Point Cloud 간 3D 움직임 벡터 추정.

| 모델 | 특징 | 적합성 |
|------|------|--------|
| **FlowFormer** | Transformer 기반, 경량 | ★★★★★ |
| **SeFlow** | Self-supervised, 자율주행 특화 | ★★★★☆ |
| FH-Net | Fast Hierarchical, 실시간 | ★★★★☆ |

**FlowFormer 특징**:
- 경량 feature enhancement 모듈
- Seed points guided attention
- 글로벌 컨텍스트 인코딩

**참고**: [FlowFormer: 3D scene flow estimation for point clouds with transformers](https://www.sciencedirect.com/science/article/abs/pii/S0950705123007918)

### 3.4 6DoF Pose 추정: Depth 기반 경량 방법

Jetson Orin Nano Super 환경에 적합한 경량 자세 추정 방법.

**추천 접근법**:

1. **OBB + PCA (즉시 적용 가능)**:
   - Open3D 활용
   - ~5ms 처리 시간
   - Roll, Pitch, Yaw 모두 측정

2. **Lightweight Depth-based Pose Estimation**:
   - Compressive sensing 활용
   - 82% 측정값만으로 정확한 추정
   - Edge device 최적화

```python
import open3d as o3d
import numpy as np

def estimate_6dof_pose_obb(points):
    """OBB 기반 6DoF 자세 추정"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Oriented Bounding Box
    obb = pcd.get_oriented_bounding_box()

    # 중심 및 회전
    center = np.asarray(obb.center)
    R = np.asarray(obb.R)

    # Euler 각도 추출
    euler = rotation_matrix_to_euler(R)

    return {
        'position': center,
        'roll': euler[0],
        'pitch': euler[1],
        'yaw': euler[2],
        'dimensions': np.asarray(obb.extent)
    }
```

### 3.5 GRU-Kalman Filter (2024 최신)

기존 Kalman Filter의 한계를 극복하는 학습 기반 필터.

**장점**:
- 비선형 움직임 모델링
- 데이터 기반 노이즈 학습
- 수동 모델 설계 불필요

**참고**: [3D Multi-Object Tracking with Semi-Supervised GRU-Kalman Filter](https://arxiv.org/html/2411.08433v1)

### 3.6 Motion Model 선택 가이드

| 움직임 패턴 | 추천 모델 | 이유 |
|-------------|----------|------|
| 일정 속도 | Constant Velocity (CV) | 단순, 효과적 |
| 가속/감속 | Constant Acceleration (CA) | 가속도 추정 포함 |
| 불규칙 | Constant Jerk (CJ) | 급격한 변화 대응 |
| 복합 | Adaptive Multi-Model | 상황별 자동 선택 |

---

## 4. D455 특화 최적화

### 4.1 Depth 정확도 향상 설정

D455의 최적 설정:

```python
import pyrealsense2 as rs

def configure_d455_optimal():
    pipeline = rs.pipeline()
    config = rs.config()

    # 최적 해상도: 848x480 (최고 정확도)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

    # IMU 스트림
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    profile = pipeline.start(config)

    # Depth 센서 설정
    depth_sensor = profile.get_device().first_depth_sensor()

    # High Accuracy 프리셋
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset,
                                rs.rs400_visual_preset.high_accuracy)

    # 레이저 파워 최대화
    if depth_sensor.supports(rs.option.laser_power):
        depth_sensor.set_option(rs.option.laser_power, 360)  # 최대값

    return pipeline
```

**참고**: [Tuning depth cameras for best performance](https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance)

### 4.2 Post-Processing 필터 체인 (D455 최적화)

```python
def create_d455_filter_chain():
    """D455 최적화 필터 체인"""

    # 1. Decimation (해상도 감소)
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)

    # 2. Spatial (공간 필터링) - D455 특화 설정
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 0)

    # 3. Temporal (시간 필터링)
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    # 4. Hole Filling
    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 1)  # 최근접 픽셀

    # 5. Threshold (거리 제한)
    threshold = rs.threshold_filter()
    threshold.set_option(rs.option.min_distance, 0.4)  # D455 최소 거리
    threshold.set_option(rs.option.max_distance, 3.0)  # 작업 거리

    return [decimation, threshold, spatial, temporal, hole_filling]

def apply_filters(depth_frame, filters):
    for f in filters:
        depth_frame = f.process(depth_frame)
    return depth_frame
```

### 4.3 자동 캘리브레이션

D455는 온칩 자동 캘리브레이션 지원:

```python
def auto_calibrate_d455(device):
    """D455 자동 캘리브레이션"""
    # Calibration Health 확인
    calibration = device.first_depth_sensor().query_stream_profile_calibration()

    # On-Chip Calibration 실행 (15초 소요)
    auto_calibrated = device.run_on_chip_calibration(
        timeout=15000,
        json_content=""
    )

    if auto_calibrated:
        print("캘리브레이션 성공")

    return auto_calibrated
```

---

## 5. Jetson Orin Nano Super 최적화

### 5.1 Super Mode 활성화

```bash
# Super Mode (MAXN) 활성화 - 최대 성능
sudo nvpmodel -m 2
sudo jetson_clocks

# 현재 모드 확인
nvpmodel -q
```

**성능 향상**: AI 워크로드에서 최대 1.7배 성능 향상

### 5.2 TensorRT 최적화

```python
from ultralytics import YOLO

def optimize_for_tensorrt():
    """YOLO 모델 TensorRT 최적화"""
    model = YOLO("yolo11s.pt")

    # TensorRT FP16 엔진 생성
    model.export(
        format="engine",
        half=True,  # FP16
        device=0,
        workspace=4,  # GB
        batch=1
    )

    return YOLO("yolo11s.engine")
```

### 5.3 메모리 최적화

Jetson Orin Nano Super 8GB 메모리 활용:

```python
# 메모리 예산
MEMORY_BUDGET = {
    'system': 1.5,      # GB
    'cuda_tensorrt': 1.0,
    'yolo_model': 0.6,
    'frame_buffers': 0.1,
    'point_cloud': 0.5,
    'available': 4.3    # 여유 메모리
}

# Unified Memory 활용
import torch
torch.cuda.set_per_process_memory_fraction(0.7)  # 70% 제한
```

### 5.4 멀티스레드 파이프라인

```python
import threading
from queue import Queue

class OptimizedPipeline:
    def __init__(self):
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)

    def start(self):
        # Thread 1: 카메라 캡처
        t1 = threading.Thread(target=self.capture_thread, daemon=True)

        # Thread 2: YOLO 추론 (GPU)
        t2 = threading.Thread(target=self.inference_thread, daemon=True)

        # Thread 3: Point Cloud 처리
        t3 = threading.Thread(target=self.pointcloud_thread, daemon=True)

        t1.start()
        t2.start()
        t3.start()
```

### 5.5 DeepStream SDK 활용 (고급)

대규모 비디오 분석 시 DeepStream SDK 활용:

```bash
# DeepStream 설치 (JetPack 6.1)
sudo apt-get install deepstream-6.4
```

**참고**: [YOLO11 with DeepStream on Jetson](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson/)

---

## 6. 구현 우선순위 및 로드맵

### Phase 1: 기반 개선 (1-2주)

**우선순위: 매우 높음**

| 작업 | 예상 시간 | 영향도 |
|------|----------|--------|
| 파일 I/O 제거 | 1일 | 높음 |
| D455 센서 클래스 생성 | 2일 | 높음 |
| 필터 체인 적용 | 1일 | 중간 |
| IMU 데이터 통합 | 1일 | 중간 |

### Phase 2: 3D 측정 구현 (2-3주)

**우선순위: 높음**

| 작업 | 예상 시간 | 영향도 |
|------|----------|--------|
| 3D Kalman Filter 통합 | 2일 | 매우 높음 |
| PCA/OBB 각도 측정 | 2일 | 높음 |
| ROI 기반 Point Cloud | 1일 | 중간 |
| Temporal Smoothing | 1일 | 중간 |

### Phase 3: AI 모델 적용 (2-3주)

**우선순위: 중간-높음**

| 작업 | 예상 시간 | 영향도 |
|------|----------|--------|
| YOLO11 TensorRT 변환 | 1일 | 높음 |
| YOLO11 통합 테스트 | 2일 | 높음 |
| 적응형 노이즈 모델 | 2일 | 중간 |
| Multi-Sweep PC | 2일 | 중간 |

### Phase 4: 고급 최적화 (선택사항)

**우선순위: 낮음-중간**

| 작업 | 예상 시간 | 영향도 |
|------|----------|--------|
| GRU-Kalman Filter | 5일 | 높음 |
| Scene Flow 모델 | 5일 | 높음 |
| HVTrack 통합 | 7일 | 높음 |
| DeepStream 파이프라인 | 3일 | 중간 |

---

## 7. 예상 성능 지표

### 7.1 기존 vs 개선 비교

| 항목 | 현재 시스템 | Phase 1-2 완료 | Phase 3 완료 |
|------|-------------|----------------|--------------|
| **탐지 FPS** | ~15 | ~40 | 55-60 |
| **파이프라인 FPS** | ~10 | ~25 | ~30 |
| **속도 측정** | 미구현 | ±8-10% | ±5-8% |
| **Pitch 각도** | ±15-20° | ±5-7° | ±3-5° |
| **Yaw 각도** | ±10-15° | ±4-6° | ±2-4° |
| **Roll 각도** | 미구현 | ±5-8° | ±3-5° |
| **지연 시간** | ~100ms | ~50ms | ~35-40ms |
| **메모리 사용** | N/A | ~4GB | ~5GB |
| **전력 소비** | N/A | ~18W | ~22W |

### 7.2 D455 거리별 정확도

| 거리 | Depth 오차 | 속도 오차 | 각도 오차 |
|------|----------|----------|----------|
| 0.5m | ~2.5mm | ±3% | ±2° |
| 1.0m | ~5mm | ±5% | ±3° |
| 2.0m | ~20mm | ±7% | ±4° |
| 3.0m | ~45mm | ±10% | ±5° |

### 7.3 처리 시간 분석

```
[Phase 3 완료 후 예상 처리 시간]

┌────────────────────────┬──────────────┐
│ 처리 단계              │ 시간 (ms)    │
├────────────────────────┼──────────────┤
│ 프레임 캡처            │ 2            │
│ Depth 필터링           │ 3            │
│ YOLO11 추론 (TensorRT) │ 15-18        │
│ Point Cloud 생성       │ 5            │
│ ROI 추출              │ 2            │
│ PCA/OBB 계산          │ 3            │
│ Kalman Filter         │ 1            │
│ 후처리/발행           │ 2            │
├────────────────────────┼──────────────┤
│ 총 처리 시간          │ 33-36ms      │
│ 예상 FPS              │ 28-30        │
└────────────────────────┴──────────────┘
```

---

## 8. 참고 자료

### 8.1 공식 문서

- **Intel RealSense D455**: [Product Page](https://www.intelrealsense.com/depth-camera-d455/)
- **D455 Specifications**: [Intel Specs](https://www.intel.com/content/www/us/en/products/sku/205847/intel-realsense-depth-camera-d455/specifications.html)
- **Tuning Depth Cameras**: [Best Performance Guide](https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance)
- **Jetson Orin Nano Super**: [NVIDIA Product Page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
- **Jetson Developer Blog**: [Super Boost Announcement](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/)

### 8.2 AI/ML 모델

- **YOLO11 on Jetson**: [Ultralytics Guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- **YOLO11 Jetson Performance**: [Ultralytics Blog](https://www.ultralytics.com/blog/ultralytics-yolo11-on-nvidia-jetson-orin-nano-super-fast-and-efficient)
- **HVTrack (ECCV 2024)**: [arXiv Paper](https://arxiv.org/html/2408.02049v3)
- **FlowFormer**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705123007918)
- **GRU-Kalman Filter**: [arXiv Paper](https://arxiv.org/html/2411.08433v1)

### 8.3 기술 문서

- **3D Deep Learning Survey**: [PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12196975/)
- **Kalman Filter for MOT**: [Motion Dynamics Paper](https://arxiv.org/html/2505.07254v1)
- **Scene Flow Survey**: [Computer Graphics Forum](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14795)
- **Edge AI Pose Estimation**: [Expert Systems Journal](https://www.sciencedirect.com/science/article/pii/S0957417424009552)

### 8.4 오픈소스 라이브러리

- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics
- **Open3D**: https://www.open3d.org/
- **FilterPy**: https://github.com/rlabbe/filterpy
- **Intel RealSense SDK**: https://github.com/IntelRealSense/librealsense
- **Awesome Point Cloud Scene Flow**: https://github.com/MaxChanger/awesome-point-cloud-scene-flow

---

## 결론

### 핵심 개선 사항 요약

1. **즉시 적용 가능**:
   - 파일 I/O 제거 → 메모리 기반 처리
   - D455 필터 체인 적용
   - 3D Kalman Filter 통합 (FilterPy)
   - PCA/OBB 기반 6DoF 각도 측정

2. **단기 적용 (1-2주)**:
   - YOLO11 TensorRT 변환
   - IMU 데이터 융합
   - ROI 기반 Point Cloud 처리

3. **중장기 적용 (선택사항)**:
   - GRU-Kalman Filter
   - Scene Flow 모델 (FlowFormer)
   - HVTrack 3D 추적기

### 예상 최종 성능

| 메트릭 | 목표치 |
|--------|--------|
| 탐지 FPS | 55-60 |
| 파이프라인 FPS | 28-30 |
| 속도 측정 정확도 | ±5-8% |
| 각도 측정 정확도 | ±2-5° |
| 추적 성공률 | >92% |
| 지연 시간 | <40ms |
| 전력 소비 | <25W |

D455 + Jetson Orin Nano Super 조합은 **비용 효율성**과 **실내/실외 호환성**에서 우수하며,
제안된 개선 사항 적용 시 산업용 수준의 3D 측정 시스템 구현이 가능합니다.
