# FoundationPose 기반 6DoF 자세 추정 구현 가이드
## 도장물 속도/각도 측정 시스템 업데이트

**작성일**: 2025-12-31
**기반 논문**: FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects (CVPR 2024 Highlight)
**목적**: 기존 ai_based_velocity_angle_measurement_research.md 구현 계획을 FoundationPose 기반으로 수정
**환경**: Intel RealSense D455 + NVIDIA Jetson Orin Nano Super

---

## 목차

1. [개요](#1-개요)
2. [FoundationPose 핵심 개념](#2-foundationpose-핵심-개념)
3. [기존 계획 대비 변경사항](#3-기존-계획-대비-변경사항)
4. [환경 설정](#4-환경-설정)
5. [Model-Based 설정 (CAD 모델 사용)](#5-model-based-설정-cad-모델-사용)
6. [Model-Free 설정 (참조 이미지 사용)](#6-model-free-설정-참조-이미지-사용)
7. [속도/각도 측정 통합](#7-속도각도-측정-통합)
8. [Jetson 최적화](#8-jetson-최적화)
9. [구현 로드맵](#9-구현-로드맵)
10. [참고 자료](#10-참고-자료)

---

## 1. 개요

### 1.1 배경

기존 `ai_based_velocity_angle_measurement_research.md` 문서에서는 다음과 같은 AI 기반 접근법을 권장했습니다:

| 측정 항목 | 기존 권장 | 새로운 권장 (FoundationPose) |
|----------|----------|----------------------------|
| **속도 측정** | KalmanNet + RAFT-Small | FoundationPose 추적 + Kalman Filter |
| **각도 측정** | EfficientPose | FoundationPose (6DoF 직접 추정) |
| **객체 추적** | ByteTrack | FoundationPose 통합 추적 |

### 1.2 왜 FoundationPose인가?

FoundationPose는 NVIDIA에서 개발한 **통합 6DoF 자세 추정 및 추적 모델**로, 다음과 같은 장점이 있습니다:

| 특징 | 설명 | 기존 방법 대비 장점 |
|------|------|-------------------|
| **통합 프레임워크** | 자세 추정 + 추적을 단일 모델로 | 파이프라인 단순화 |
| **Zero-Shot 일반화** | 새로운 객체에 미세조정 없이 적용 | Domain Shift 문제 해결 |
| **Model-Free 지원** | CAD 없이 참조 이미지만으로 동작 | 유연한 적용 가능 |
| **고정밀 6DoF** | 위치 + 방향 동시 추정 | 별도 각도 추정 불필요 |
| **실시간 추적** | 추적 모드에서 120+ FPS | 고속 처리 가능 |

### 1.3 프로젝트 적용 효과

```
기존 계획:
[YOLO 탐지] → [ByteTrack] → [EfficientPose] → [KalmanNet] → [속도/각도]
                   ↓              ↓               ↓
              복잡한 파이프라인, 다중 모델 관리, Domain Shift 취약

새로운 계획:
[YOLO 탐지] → [FoundationPose (통합)] → [속도/각도]
                       ↓
              단순화된 파이프라인, 강건한 일반화, 통합 6DoF
```

---

## 2. FoundationPose 핵심 개념

### 2.1 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      FoundationPose 아키텍처                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   [입력]                                                                 │
│   ┌──────────────┐    ┌──────────────┐                                  │
│   │   RGB 이미지  │    │  Depth 이미지 │  ← RGBD 입력                   │
│   └──────┬───────┘    └──────┬───────┘                                  │
│          │                   │                                           │
│          └─────────┬─────────┘                                           │
│                    ▼                                                     │
│   ┌─────────────────────────────────────────────────────────────┐       │
│   │              객체 정보 (둘 중 하나)                          │       │
│   │  ┌─────────────────┐     ┌─────────────────────────┐       │       │
│   │  │ Model-Based:    │ OR  │ Model-Free:             │       │       │
│   │  │ CAD 모델        │     │ ~16개 참조 이미지       │       │       │
│   │  │ (텍스처 메시)   │     │ + Neural Object Field  │       │       │
│   │  └─────────────────┘     └─────────────────────────┘       │       │
│   └────────────────────────────────┬────────────────────────────┘       │
│                                    │                                     │
│                                    ▼                                     │
│   ┌─────────────────────────────────────────────────────────────┐       │
│   │                     자세 가설 생성                           │       │
│   │  • 정이십면체 샘플링으로 N_s개 시점                         │       │
│   │  • 평면 내 회전 N_i개로 증강                                │       │
│   │  → 총 N_s × N_i 개의 초기 자세 가설                        │       │
│   └────────────────────────────────┬────────────────────────────┘       │
│                                    │                                     │
│                                    ▼                                     │
│   ┌─────────────────────────────────────────────────────────────┐       │
│   │                     자세 정제 네트워크                        │       │
│   │  • Transformer 기반 아키텍처                                │       │
│   │  • 렌더링 vs 관측 비교                                      │       │
│   │  • 반복적 업데이트 (이동 Δt, 회전 ΔR)                      │       │
│   └────────────────────────────────┬────────────────────────────┘       │
│                                    │                                     │
│                                    ▼                                     │
│   ┌─────────────────────────────────────────────────────────────┐       │
│   │                     자세 선택 (순위 결정)                    │       │
│   │  • 계층적 비교 전략                                         │       │
│   │  • 대조 학습 기반 점수 계산                                 │       │
│   │  • 최고 점수 자세 선택                                      │       │
│   └────────────────────────────────┬────────────────────────────┘       │
│                                    │                                     │
│                                    ▼                                     │
│   [출력]                                                                 │
│   ┌──────────────────────────────────────────────────────────┐          │
│   │  6DoF 자세: [R | t] ∈ SE(3)                              │          │
│   │  • 이동 (Translation): t = [tx, ty, tz]                  │          │
│   │  • 회전 (Rotation): R (3×3 행렬 또는 쿼터니언)           │          │
│   │  → Roll, Pitch, Yaw 직접 추출 가능                       │          │
│   └──────────────────────────────────────────────────────────┘          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 두 가지 설정 비교

#### Model-Based (CAD 모델 사용)

```yaml
model_based:
  입력:
    - RGBD 이미지
    - 객체 CAD 모델 (텍스처 메시)

  장점:
    - 더 정확한 자세 추정
    - 빠른 렌더링
    - 안정적인 추적

  단점:
    - CAD 모델 필요
    - CAD와 실제 객체 차이 시 성능 저하

  적합한 경우:
    - 정밀 CAD 모델이 있는 경우
    - 제조업 환경
    - 표준화된 객체
```

#### Model-Free (참조 이미지 사용)

```yaml
model_free:
  입력:
    - RGBD 이미지
    - ~16개 참조 이미지 (다양한 시점)

  장점:
    - CAD 모델 불필요
    - 새로운 객체에 쉽게 적용
    - 실제 외관 그대로 사용

  단점:
    - 참조 이미지 촬영 필요
    - Neural Object Field 학습 시간
    - 약간 낮은 정확도

  적합한 경우:
    - CAD 모델이 없는 경우
    - 다양한 객체 처리
    - 빠른 프로토타입
```

### 2.3 Neural Object Field (신경 객체 필드)

Model-Free 설정에서 핵심이 되는 기술:

```python
"""
Neural Object Field 구조:

1. 기하학 함수 Ω: x → s
   - 입력: 3D 점 x ∈ ℝ³
   - 출력: 부호 있는 거리 값 s ∈ ℝ (SDF)
   - 목적: 객체 형상 표현

2. 외관 함수 Φ: (f, n, d) → c
   - 입력: 기하학 특징 f, 법선 n, 시선 방향 d
   - 출력: 색상 c ∈ ℝ³
   - 목적: 텍스처/외관 표현

장점:
- NeRF 대비 더 정확한 깊이 렌더링
- 효율적인 새로운 뷰 합성
- ~16개 참조 이미지로 충분한 품질
"""
```

### 2.4 성능 벤치마크

| 데이터셋 | 메트릭 | 기존 SOTA | FoundationPose | 개선율 |
|----------|--------|----------|----------------|--------|
| **YCB-Video** | ADD-S AUC | 88.4% (FS6D) | **97.4%** | +9.0% |
| **LINEMOD** | ADD-0.1d | 91.5% (FS6D+ICP) | **99.9%** | +8.4% |
| **YCBInEOAT** | ADD AUC | 92.66% (se(3)-TrackNet) | **93.09%** | +0.43% |

---

## 3. 기존 계획 대비 변경사항

### 3.1 측정 파이프라인 변경

#### 기존 계획 (ai_based_velocity_angle_measurement_research.md)

```
속도 측정:
  [YOLO 탐지] → [ByteTrack] → [RAFT-Small] → [KalmanNet] → 속도

각도 측정:
  [YOLO 탐지] → [Point Cloud 추출] → [EfficientPose] → Roll/Pitch/Yaw

문제점:
  - 다중 모델 관리의 복잡성
  - Domain Shift에 취약
  - 모듈 간 오류 전파
```

#### 새로운 계획 (FoundationPose 기반)

```
통합 측정:
  [YOLO 탐지] → [FoundationPose] → [6DoF 자세] → [후처리] → 속도/각도
                      │
                      └─→ [추적 모드] → 실시간 연속 추정

장점:
  - 단일 모델로 통합
  - 강건한 일반화 (새 객체 즉시 적용)
  - End-to-end 6DoF 추정
```

### 3.2 모듈별 변경사항

| 모듈 | 기존 계획 | 새 계획 | 변경 이유 |
|------|----------|---------|----------|
| **객체 탐지** | YOLO11s | YOLO11s (유지) | 탐지 성능 우수 |
| **각도 추정** | EfficientPose | FoundationPose | 통합 6DoF |
| **속도 추정** | RAFT-Small + KalmanNet | FoundationPose 추적 + CA Kalman | 단순화 |
| **추적** | ByteTrack | FoundationPose 내장 추적 | 통합 |

### 3.3 정확도 비교 예상

| 측정 항목 | 기존 계획 (예상) | 새 계획 (예상) | 근거 |
|----------|-----------------|---------------|------|
| **속도** | ±3-6% | ±2-5% | 6DoF 추적의 정밀도 |
| **Roll** | ±1-3° | ±1-2° | FoundationPose 정확도 |
| **Pitch** | ±1-3° | ±1-2° | FoundationPose 정확도 |
| **Yaw** | ±1-2° | ±0.5-1° | FoundationPose 정확도 |

---

## 4. 환경 설정

### 4.1 하드웨어 요구사항

| 구성요소 | 최소 사양 | 권장 사양 | 현재 환경 |
|----------|----------|----------|----------|
| **GPU** | NVIDIA GTX 1080 | RTX 3090+ | Jetson Orin Nano Super |
| **VRAM** | 8GB | 12GB+ | 8GB (공유) |
| **RAM** | 16GB | 32GB+ | 8GB |
| **Storage** | 50GB | 100GB+ | 충분 |

### 4.2 Conda 환경 설정

```bash
# 환경 생성
conda create -n foundationpose python=3.9 -y
conda activate foundationpose

# 기본 패키지 설치
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Eigen3 설치 (3.4.0)
conda install eigen=3.4.0 -c conda-forge

# 필수 패키지 설치
pip install -r requirements.txt

# NVDiffRast 설치
pip install git+https://github.com/NVlabs/nvdiffrast

# Kaolin 설치 (Model-Free 설정 필요)
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D 설치
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# 확장 빌드
bash build_all_conda.sh
```

### 4.3 Docker 설정 (권장)

```bash
# Docker 이미지 Pull
cd docker/
docker pull wenbowen123/foundationpose
docker tag wenbowen123/foundationpose foundationpose

# 또는 RTX 4090 등 최신 GPU용
docker pull shingarey/foundationpose_custom_cuda121:latest

# Docker 실행
docker run --gpus all -it \
    -v /path/to/data:/workspace/data \
    -v /path/to/models:/workspace/models \
    foundationpose bash
```

### 4.4 Isaac ROS 설정 (Jetson용)

```bash
# Isaac ROS FoundationPose 설치
mkdir -p ~/workspaces/isaac_ros-dev/src
cd ~/workspaces/isaac_ros-dev/src

# Isaac ROS Common
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git

# Isaac ROS FoundationPose
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation.git

# 빌드
cd ~/workspaces/isaac_ros-dev
colcon build --symlink-install

# 환경 소스
source install/setup.bash
```

---

## 5. Model-Based 설정 (CAD 모델 사용)

### 5.1 CAD 모델 준비

#### 지원 형식

```yaml
supported_formats:
  - obj (권장)
  - ply
  - stl (텍스처 없음)
  - glb/gltf

requirements:
  - 텍스처 포함 권장
  - 미터 단위 스케일
  - 원점이 객체 중심
```

#### CAD 모델 전처리

```python
import trimesh
import numpy as np

def prepare_cad_model(input_path, output_path, target_scale=1.0):
    """CAD 모델 전처리"""

    # 모델 로드
    mesh = trimesh.load(input_path)

    # 중심 정렬
    mesh.vertices -= mesh.center_mass

    # 스케일 조정 (미터 단위로)
    current_size = mesh.extents.max()
    scale_factor = target_scale / current_size
    mesh.vertices *= scale_factor

    # 저장
    mesh.export(output_path)

    print(f"Model prepared: {output_path}")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Size: {mesh.extents}")

    return mesh
```

### 5.2 Model-Based 추론

```python
import sys
sys.path.append('/path/to/FoundationPose')

from estimater import FoundationPose
import cv2
import numpy as np

class ModelBasedPoseEstimator:
    """Model-Based FoundationPose 추정기"""

    def __init__(self, mesh_path, model_dir, device='cuda:0'):
        """
        Args:
            mesh_path: CAD 모델 경로 (.obj)
            model_dir: 사전학습 모델 디렉토리
            device: 추론 디바이스
        """
        self.estimator = FoundationPose(
            model_dir=model_dir,
            device=device
        )

        # CAD 모델 로드
        self.estimator.load_mesh(mesh_path)

        print(f"Model-Based Estimator initialized")
        print(f"  Mesh: {mesh_path}")

    def estimate_pose(self, rgb, depth, mask, intrinsics):
        """
        단일 프레임 자세 추정

        Args:
            rgb: RGB 이미지 [H, W, 3]
            depth: Depth 이미지 [H, W] (미터 단위)
            mask: 객체 마스크 [H, W] (binary)
            intrinsics: 카메라 내부 파라미터 {'fx', 'fy', 'cx', 'cy'}

        Returns:
            pose: 4x4 변환 행렬 [R|t]
            confidence: 추정 신뢰도
        """
        # 전처리
        rgb = rgb.astype(np.float32) / 255.0
        depth = depth.astype(np.float32)

        # 자세 추정
        pose, confidence = self.estimator.estimate(
            rgb=rgb,
            depth=depth,
            mask=mask,
            K=np.array([
                [intrinsics['fx'], 0, intrinsics['cx']],
                [0, intrinsics['fy'], intrinsics['cy']],
                [0, 0, 1]
            ])
        )

        return pose, confidence

    def track_pose(self, rgb, depth, prev_pose, intrinsics):
        """
        추적 모드 자세 추정 (이전 자세 활용)

        Args:
            rgb, depth: 현재 프레임
            prev_pose: 이전 프레임 자세
            intrinsics: 카메라 파라미터

        Returns:
            pose: 업데이트된 자세
            confidence: 신뢰도
        """
        pose, confidence = self.estimator.track(
            rgb=rgb.astype(np.float32) / 255.0,
            depth=depth.astype(np.float32),
            prev_pose=prev_pose,
            K=np.array([
                [intrinsics['fx'], 0, intrinsics['cx']],
                [0, intrinsics['fy'], intrinsics['cy']],
                [0, 0, 1]
            ])
        )

        return pose, confidence
```

### 5.3 실행 예시

```python
# 초기화
estimator = ModelBasedPoseEstimator(
    mesh_path='models/painting_object.obj',
    model_dir='models/foundationpose_weights',
    device='cuda:0'
)

# 카메라 파라미터 (D455)
intrinsics = {
    'fx': 383.883,
    'fy': 383.883,
    'cx': 320.499,
    'cy': 237.913
}

# 단일 프레임 추정
pose, confidence = estimator.estimate_pose(
    rgb=rgb_image,
    depth=depth_image,
    mask=object_mask,
    intrinsics=intrinsics
)

print(f"Pose: {pose}")
print(f"Confidence: {confidence}")
```

---

## 6. Model-Free 설정 (참조 이미지 사용)

### 6.1 참조 이미지 촬영 가이드

#### 촬영 요구사항

```yaml
reference_images:
  count: 16개 (최소 12개, 최대 32개)

  viewpoints:
    - 정면 (0°)
    - 측면 좌/우 (±30°, ±60°, ±90°)
    - 상단 (45°, 90°)
    - 하단 (필요시)

  coverage:
    - 객체의 모든 면이 최소 2개 이상의 이미지에 포함
    - 시점 간 중첩 영역 확보

  quality:
    - 선명한 초점
    - 적절한 조명 (그림자 최소화)
    - 배경과 구분 가능
```

#### 촬영 설정 다이어그램

```
                    Top View (90°)
                         │
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    60° Left ─────── [Object] ─────── 60° Right
         │               │               │
         │               │               │
    30° Left            │           30° Right
         │               │               │
         └───────────────┼───────────────┘
                         │
                    Front View (0°)

권장 촬영 위치:
• 수평면: 0°, ±30°, ±60°, ±90°, ±120°, ±150°, 180° (13개)
• 상단: 45°에서 3개
• 총 16개 이미지
```

### 6.2 참조 이미지 데이터 구조

```
reference_images/
├── painting_object/
│   ├── rgb/
│   │   ├── 000.png
│   │   ├── 001.png
│   │   ├── ...
│   │   └── 015.png
│   ├── depth/
│   │   ├── 000.png
│   │   ├── 001.png
│   │   ├── ...
│   │   └── 015.png
│   ├── mask/
│   │   ├── 000.png
│   │   ├── 001.png
│   │   ├── ...
│   │   └── 015.png
│   └── poses.json  # 각 이미지의 카메라 자세 (선택)
```

### 6.3 Neural Object Field 학습

```python
import sys
sys.path.append('/path/to/FoundationPose')

from bundlesdf.run_nerf import train_neural_field

def train_object_field(ref_dir, output_dir, config=None):
    """
    참조 이미지로부터 Neural Object Field 학습

    Args:
        ref_dir: 참조 이미지 디렉토리
        output_dir: 출력 디렉토리
        config: 학습 설정 (None이면 기본값)
    """
    default_config = {
        'num_iterations': 5000,
        'learning_rate': 0.001,
        'batch_size': 1024,
        'sdf_lambda': 0.1,
        'eikonal_lambda': 0.1,
        'truncation_distance': 0.02,
    }

    if config:
        default_config.update(config)

    # Neural Field 학습 실행
    train_neural_field(
        ref_view_dir=ref_dir,
        output_dir=output_dir,
        **default_config
    )

    print(f"Neural Object Field trained and saved to: {output_dir}")
```

#### CLI 실행

```bash
# Neural Object Field 학습
python bundlesdf/run_nerf.py \
    --ref_view_dir reference_images/painting_object \
    --dataset_flag custom \
    --output_dir models/neural_fields/painting_object

# 학습 시간: ~30초 ~ 2분 (이미지 수와 해상도에 따라)
```

### 6.4 Model-Free 추론

```python
class ModelFreePoseEstimator:
    """Model-Free FoundationPose 추정기"""

    def __init__(self, neural_field_dir, model_dir, device='cuda:0'):
        """
        Args:
            neural_field_dir: 학습된 Neural Field 디렉토리
            model_dir: 사전학습 모델 디렉토리
            device: 추론 디바이스
        """
        self.estimator = FoundationPose(
            model_dir=model_dir,
            device=device
        )

        # Neural Field에서 메시 추출
        self.estimator.load_reconstructed_mesh(
            neural_field_dir,
            use_reconstructed_mesh=True
        )

        print(f"Model-Free Estimator initialized")
        print(f"  Neural Field: {neural_field_dir}")

    def estimate_pose(self, rgb, depth, mask, intrinsics):
        """Model-Based와 동일한 인터페이스"""
        rgb = rgb.astype(np.float32) / 255.0
        depth = depth.astype(np.float32)

        pose, confidence = self.estimator.estimate(
            rgb=rgb,
            depth=depth,
            mask=mask,
            K=np.array([
                [intrinsics['fx'], 0, intrinsics['cx']],
                [0, intrinsics['fy'], intrinsics['cy']],
                [0, 0, 1]
            ])
        )

        return pose, confidence
```

### 6.5 참조 이미지 촬영 스크립트

```python
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json

class ReferenceImageCapture:
    """참조 이미지 촬영 도구"""

    def __init__(self, output_dir, num_images=16):
        self.output_dir = output_dir
        self.num_images = num_images
        self.captured_count = 0

        # 디렉토리 생성
        os.makedirs(f"{output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)
        os.makedirs(f"{output_dir}/mask", exist_ok=True)

        # RealSense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        # 정렬 객체
        self.align = rs.align(rs.stream.color)

        print(f"Reference Image Capture initialized")
        print(f"  Output: {output_dir}")
        print(f"  Target images: {num_images}")
        print(f"  Press 'c' to capture, 'q' to quit")

    def capture_frame(self):
        """현재 프레임 촬영"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # 저장
        idx = self.captured_count
        cv2.imwrite(f"{self.output_dir}/rgb/{idx:03d}.png", rgb)
        cv2.imwrite(f"{self.output_dir}/depth/{idx:03d}.png", depth)

        # 마스크 (수동 또는 자동 생성 필요)
        # 여기서는 placeholder로 전체 화면 마스크 생성
        mask = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.uint8) * 255
        cv2.imwrite(f"{self.output_dir}/mask/{idx:03d}.png", mask)

        self.captured_count += 1
        print(f"Captured image {self.captured_count}/{self.num_images}")

        return self.captured_count >= self.num_images

    def run(self):
        """촬영 루프 실행"""
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()

                rgb = np.asanyarray(color_frame.get_data())

                # 가이드라인 표시
                h, w = rgb.shape[:2]
                cv2.line(rgb, (w//2, 0), (w//2, h), (0, 255, 0), 1)
                cv2.line(rgb, (0, h//2), (w, h//2), (0, 255, 0), 1)
                cv2.putText(rgb, f"Captured: {self.captured_count}/{self.num_images}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Reference Capture", rgb)

                key = cv2.waitKey(1)
                if key == ord('c'):
                    if self.capture_frame():
                        print("All images captured!")
                        break
                elif key == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

# 사용
capture = ReferenceImageCapture(
    output_dir="reference_images/painting_object",
    num_images=16
)
capture.run()
```

---

## 7. 속도/각도 측정 통합

### 7.1 6DoF 자세에서 각도 추출

```python
from scipy.spatial.transform import Rotation
import numpy as np

def pose_to_euler(pose_matrix):
    """
    4x4 자세 행렬에서 오일러 각도 추출

    Args:
        pose_matrix: 4x4 변환 행렬 [R|t]

    Returns:
        translation: [tx, ty, tz] 미터 단위
        euler_angles: [roll, pitch, yaw] 도 단위
    """
    # 이동 추출
    translation = pose_matrix[:3, 3]

    # 회전 추출
    rotation_matrix = pose_matrix[:3, :3]

    # scipy로 오일러 각도 변환
    r = Rotation.from_matrix(rotation_matrix)
    euler = r.as_euler('xyz', degrees=True)

    roll, pitch, yaw = euler

    return translation, (roll, pitch, yaw)


def compute_angle_change(prev_pose, curr_pose):
    """
    두 자세 간의 각도 변화 계산

    Args:
        prev_pose: 이전 자세 (4x4)
        curr_pose: 현재 자세 (4x4)

    Returns:
        delta_angles: [Δroll, Δpitch, Δyaw] 도 단위
    """
    _, prev_euler = pose_to_euler(prev_pose)
    _, curr_euler = pose_to_euler(curr_pose)

    delta_angles = np.array(curr_euler) - np.array(prev_euler)

    # -180 ~ 180 범위로 정규화
    delta_angles = np.where(delta_angles > 180, delta_angles - 360, delta_angles)
    delta_angles = np.where(delta_angles < -180, delta_angles + 360, delta_angles)

    return delta_angles
```

### 7.2 속도 계산

```python
class VelocityEstimator:
    """FoundationPose 기반 속도 추정기"""

    def __init__(self, dt=1/30.0):
        """
        Args:
            dt: 프레임 간 시간 간격 (초)
        """
        self.dt = dt
        self.prev_pose = None
        self.velocity_buffer = []
        self.buffer_size = 5  # 이동 평균용

    def update(self, current_pose):
        """
        새 자세로 속도 업데이트

        Args:
            current_pose: 현재 4x4 자세 행렬

        Returns:
            velocity: [vx, vy, vz] m/s
            speed: 속력 m/s
            angular_velocity: [ωx, ωy, ωz] °/s
        """
        if self.prev_pose is None:
            self.prev_pose = current_pose.copy()
            return None, None, None

        # 위치 변화
        prev_pos = self.prev_pose[:3, 3]
        curr_pos = current_pose[:3, 3]
        displacement = curr_pos - prev_pos

        # 선속도
        velocity = displacement / self.dt
        speed = np.linalg.norm(velocity)

        # 각속도
        delta_angles = compute_angle_change(self.prev_pose, current_pose)
        angular_velocity = delta_angles / self.dt

        # 이동 평균 필터
        self.velocity_buffer.append(velocity)
        if len(self.velocity_buffer) > self.buffer_size:
            self.velocity_buffer.pop(0)

        smoothed_velocity = np.mean(self.velocity_buffer, axis=0)
        smoothed_speed = np.linalg.norm(smoothed_velocity)

        # 상태 업데이트
        self.prev_pose = current_pose.copy()

        return smoothed_velocity, smoothed_speed, angular_velocity
```

### 7.3 Kalman Filter 통합

```python
from filterpy.kalman import KalmanFilter
import numpy as np

class PoseKalmanFilter:
    """6DoF 자세를 위한 Kalman Filter"""

    def __init__(self, dt=1/30.0):
        """
        상태 벡터: [x, y, z, vx, vy, vz, roll, pitch, yaw, ωx, ωy, ωz]
        측정 벡터: [x, y, z, roll, pitch, yaw]
        """
        self.dt = dt

        # 12차원 상태, 6차원 측정
        self.kf = KalmanFilter(dim_x=12, dim_z=6)

        # 상태 전이 행렬 (등속 모델)
        self.kf.F = np.eye(12)
        self.kf.F[0, 3] = dt  # x += vx * dt
        self.kf.F[1, 4] = dt  # y += vy * dt
        self.kf.F[2, 5] = dt  # z += vz * dt
        self.kf.F[6, 9] = dt   # roll += ωx * dt
        self.kf.F[7, 10] = dt  # pitch += ωy * dt
        self.kf.F[8, 11] = dt  # yaw += ωz * dt

        # 측정 행렬
        self.kf.H = np.zeros((6, 12))
        self.kf.H[0, 0] = 1  # x
        self.kf.H[1, 1] = 1  # y
        self.kf.H[2, 2] = 1  # z
        self.kf.H[3, 6] = 1  # roll
        self.kf.H[4, 7] = 1  # pitch
        self.kf.H[5, 8] = 1  # yaw

        # 노이즈 공분산
        self.kf.R *= 0.1     # 측정 노이즈
        self.kf.Q *= 0.01    # 프로세스 노이즈
        self.kf.P *= 1.0     # 초기 불확실성

    def predict(self):
        """예측 단계"""
        self.kf.predict()
        return self.get_state()

    def update(self, pose_matrix):
        """업데이트 단계"""
        translation, euler = pose_to_euler(pose_matrix)
        measurement = np.array([
            translation[0], translation[1], translation[2],
            euler[0], euler[1], euler[2]
        ])
        self.kf.update(measurement)
        return self.get_state()

    def get_state(self):
        """현재 상태 반환"""
        state = self.kf.x.flatten()
        return {
            'position': state[0:3],
            'velocity': state[3:6],
            'orientation': state[6:9],  # roll, pitch, yaw
            'angular_velocity': state[9:12],
            'speed': np.linalg.norm(state[3:6])
        }
```

### 7.4 통합 파이프라인

```python
class IntegratedMeasurementSystem:
    """FoundationPose 기반 통합 속도/각도 측정 시스템"""

    def __init__(self, estimator, use_kalman=True, fps=30):
        """
        Args:
            estimator: ModelBasedPoseEstimator 또는 ModelFreePoseEstimator
            use_kalman: Kalman Filter 사용 여부
            fps: 프레임 레이트
        """
        self.estimator = estimator
        self.use_kalman = use_kalman
        self.dt = 1.0 / fps

        if use_kalman:
            self.kalman = PoseKalmanFilter(dt=self.dt)
        else:
            self.velocity_estimator = VelocityEstimator(dt=self.dt)

        self.tracking_mode = False
        self.prev_pose = None

    def process_frame(self, rgb, depth, mask, intrinsics):
        """
        단일 프레임 처리

        Args:
            rgb: RGB 이미지
            depth: Depth 이미지
            mask: 객체 마스크
            intrinsics: 카메라 파라미터

        Returns:
            result: {
                'pose': 4x4 변환 행렬,
                'translation': [x, y, z],
                'euler': [roll, pitch, yaw],
                'velocity': [vx, vy, vz],
                'speed': float,
                'angular_velocity': [ωx, ωy, ωz],
                'confidence': float
            }
        """
        # 자세 추정
        if self.tracking_mode and self.prev_pose is not None:
            pose, confidence = self.estimator.track_pose(
                rgb, depth, self.prev_pose, intrinsics
            )
        else:
            pose, confidence = self.estimator.estimate_pose(
                rgb, depth, mask, intrinsics
            )
            self.tracking_mode = True

        self.prev_pose = pose

        # 각도 추출
        translation, euler = pose_to_euler(pose)

        # 속도 계산
        if self.use_kalman:
            self.kalman.predict()
            state = self.kalman.update(pose)
            velocity = state['velocity']
            speed = state['speed']
            angular_velocity = state['angular_velocity']
        else:
            velocity, speed, angular_velocity = self.velocity_estimator.update(pose)
            if velocity is None:
                velocity = np.zeros(3)
                speed = 0
                angular_velocity = np.zeros(3)

        return {
            'pose': pose,
            'translation': translation,
            'euler': {
                'roll': euler[0],
                'pitch': euler[1],
                'yaw': euler[2]
            },
            'velocity': velocity,
            'speed': speed,
            'angular_velocity': angular_velocity,
            'confidence': confidence
        }

    def reset_tracking(self):
        """추적 리셋 (객체 재탐지 필요 시)"""
        self.tracking_mode = False
        self.prev_pose = None
        if self.use_kalman:
            self.kalman = PoseKalmanFilter(dt=self.dt)
```

---

## 8. Jetson 최적화

### 8.1 성능 특성

| 모드 | 예상 FPS | GPU 사용률 | 메모리 사용 |
|------|----------|-----------|-----------|
| **자세 추정** | 0.7-1.0 FPS | 100% | ~4GB |
| **추적 모드** | 30-40 FPS | 60-80% | ~2GB |
| **혼합 모드** | 15-25 FPS | 70-90% | ~3GB |

### 8.2 최적화 전략

#### TensorRT 변환

```python
# FoundationPose 모델의 TensorRT 변환
import torch
from torch2trt import torch2trt

def convert_to_tensorrt(model, input_shape, fp16=True):
    """
    PyTorch 모델을 TensorRT로 변환

    Args:
        model: PyTorch 모델
        input_shape: 입력 텐서 shape
        fp16: FP16 모드 사용

    Returns:
        trt_model: TensorRT 최적화된 모델
    """
    x = torch.randn(input_shape).cuda()

    if fp16:
        model = model.half()
        x = x.half()

    trt_model = torch2trt(
        model,
        [x],
        fp16_mode=fp16,
        max_workspace_size=1 << 30  # 1GB
    )

    return trt_model
```

#### 추적 모드 우선 사용

```python
class OptimizedMeasurementSystem(IntegratedMeasurementSystem):
    """Jetson 최적화된 측정 시스템"""

    def __init__(self, *args, tracking_recovery_threshold=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracking_recovery_threshold = tracking_recovery_threshold
        self.consecutive_low_conf = 0
        self.max_low_conf = 5

    def process_frame(self, rgb, depth, mask, intrinsics):
        """최적화된 프레임 처리"""

        # 추적 모드 우선 (더 빠름)
        if self.tracking_mode and self.prev_pose is not None:
            pose, confidence = self.estimator.track_pose(
                rgb, depth, self.prev_pose, intrinsics
            )

            # 신뢰도 확인
            if confidence < self.tracking_recovery_threshold:
                self.consecutive_low_conf += 1
            else:
                self.consecutive_low_conf = 0

            # 연속 저신뢰도 시 재초기화
            if self.consecutive_low_conf >= self.max_low_conf:
                print("Tracking lost, re-initializing...")
                self.reset_tracking()
                pose, confidence = self.estimator.estimate_pose(
                    rgb, depth, mask, intrinsics
                )
                self.tracking_mode = True
        else:
            # 초기 자세 추정 (느림)
            pose, confidence = self.estimator.estimate_pose(
                rgb, depth, mask, intrinsics
            )
            self.tracking_mode = True

        self.prev_pose = pose

        # 이하 동일...
        translation, euler = pose_to_euler(pose)

        if self.use_kalman:
            self.kalman.predict()
            state = self.kalman.update(pose)
            velocity = state['velocity']
            speed = state['speed']
            angular_velocity = state['angular_velocity']
        else:
            velocity, speed, angular_velocity = self.velocity_estimator.update(pose)
            if velocity is None:
                velocity = np.zeros(3)
                speed = 0
                angular_velocity = np.zeros(3)

        return {
            'pose': pose,
            'translation': translation,
            'euler': {'roll': euler[0], 'pitch': euler[1], 'yaw': euler[2]},
            'velocity': velocity,
            'speed': speed,
            'angular_velocity': angular_velocity,
            'confidence': confidence
        }
```

### 8.3 Isaac ROS 활용

```yaml
# Isaac ROS FoundationPose Launch 설정
isaac_ros_foundationpose:
  node_name: foundationpose_node
  parameters:
    mesh_file: "models/painting_object.obj"
    refine_iterations: 3
    score_threshold: 0.3

  # Jetson 최적화
  jetson_optimization:
    use_tensorrt: true
    fp16_mode: true
    max_batch_size: 1

  # 추적 설정
  tracking:
    enabled: true
    recovery_threshold: 0.3
```

### 8.4 전력 모드 설정

```bash
# Jetson Orin Nano Super 전력 모드 설정

# 최고 성능 모드 (25W)
sudo nvpmodel -m 0
sudo jetson_clocks

# 균형 모드 (15W)
sudo nvpmodel -m 1

# 확인
nvpmodel -q
```

---

## 9. 구현 로드맵

### 9.1 Phase 1: 환경 구축 (1주)

```
□ FoundationPose 환경 설정 (Conda 또는 Docker)
□ RealSense D455 연동 테스트
□ 기본 추론 테스트 (공개 데이터셋)
□ Isaac ROS 설정 (Jetson용)
```

### 9.2 Phase 2: 객체 모델링 (1주)

```
□ 도장물 CAD 모델 준비 또는
□ 참조 이미지 촬영 (16장)
□ Neural Object Field 학습 (Model-Free 사용 시)
□ 추론 테스트 및 정확도 확인
```

### 9.3 Phase 3: 측정 시스템 통합 (2주)

```
□ YOLO + FoundationPose 파이프라인 구축
□ 속도/각도 측정 모듈 구현
□ Kalman Filter 통합
□ 실시간 스트리밍 테스트
```

### 9.4 Phase 4: 최적화 및 배포 (1주)

```
□ TensorRT 변환
□ Jetson 성능 테스트
□ 추적 모드 안정화
□ 에러 처리 및 복구 로직
```

### 9.5 Phase 5: 검증 및 튜닝 (1주)

```
□ 다양한 환경에서 정확도 테스트
□ Domain Shift 테스트
□ 처리 속도 최적화
□ 문서화 및 배포
```

---

## 10. 참고 자료

### 10.1 공식 리소스

- [FoundationPose GitHub](https://github.com/NVlabs/FoundationPose)
- [FoundationPose Project Page](https://nvlabs.github.io/FoundationPose/)
- [NVIDIA NGC - FoundationPose](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/foundationpose)
- [Isaac ROS FoundationPose](https://nvidia-isaac-ros.github.io/concepts/pose_estimation/foundationpose/index.html)

### 10.2 논문

- [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects (CVPR 2024)](https://arxiv.org/abs/2312.08344)
- [BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects (CVPR 2023)](https://arxiv.org/abs/2303.14158)

### 10.3 관련 프로젝트 문서

- [ai_based_velocity_angle_measurement_research.md](./251218/ai_based_velocity_angle_measurement_research.md)
- [yolo_domain_shift_analysis_report.md](./251229/yolo_domain_shift_analysis_report.md)
- [FoundationPose_Korean_Translation.md](./251224/FoundationPose_Korean_Translation.md)

### 10.4 튜토리얼

- [NVIDIA Isaac Tutorial Summary](https://tutorial.j3soon.com/robotics/nvidia-isaac-summary/)
- [Estimate and track object poses with NVIDIA TAO FoundationPose model](https://www.nvidia.com/en-us/on-demand/session/other2024-tao55fdn/)

---

## 부록: 코드 예시 모음

### A. 전체 파이프라인 예시

```python
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# 시스템 초기화
def initialize_system():
    # YOLO 탐지기
    yolo = YOLO("models/yolo11s_custom.pt")

    # FoundationPose 추정기 (Model-Based)
    pose_estimator = ModelBasedPoseEstimator(
        mesh_path="models/painting_object.obj",
        model_dir="models/foundationpose_weights"
    )

    # 통합 측정 시스템
    measurement_system = OptimizedMeasurementSystem(
        estimator=pose_estimator,
        use_kalman=True,
        fps=30
    )

    # RealSense 초기화
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    align = rs.align(rs.stream.color)

    # 카메라 파라미터
    intrinsics = {
        'fx': 383.883,
        'fy': 383.883,
        'cx': 320.499,
        'cy': 237.913
    }

    return yolo, measurement_system, pipeline, align, intrinsics


def main():
    yolo, measurement_system, pipeline, align, intrinsics = initialize_system()

    try:
        while True:
            # 프레임 획득
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            rgb = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0

            # YOLO 탐지
            results = yolo.predict(rgb, conf=0.5, verbose=False)

            for r in results:
                for box in r.boxes:
                    # 바운딩 박스에서 마스크 생성
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
                    mask[y1:y2, x1:x2] = 255

                    # FoundationPose로 측정
                    result = measurement_system.process_frame(
                        rgb, depth, mask, intrinsics
                    )

                    # 결과 출력
                    print(f"Position: {result['translation']}")
                    print(f"Roll: {result['euler']['roll']:.2f}°")
                    print(f"Pitch: {result['euler']['pitch']:.2f}°")
                    print(f"Yaw: {result['euler']['yaw']:.2f}°")
                    print(f"Speed: {result['speed']:.3f} m/s")
                    print(f"Confidence: {result['confidence']:.3f}")
                    print("-" * 40)

                    # 시각화
                    cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(rgb, f"Speed: {result['speed']:.3f} m/s",
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("FoundationPose Measurement", rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### B. 결과 로깅 및 저장

```python
import json
import csv
from datetime import datetime

class MeasurementLogger:
    """측정 결과 로깅"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"{output_dir}/measurements_{timestamp}.csv"
        self.json_path = f"{output_dir}/measurements_{timestamp}.json"

        # CSV 헤더 작성
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame_id',
                'x', 'y', 'z',
                'roll', 'pitch', 'yaw',
                'vx', 'vy', 'vz', 'speed',
                'confidence'
            ])

        self.measurements = []

    def log(self, frame_id, result):
        """측정 결과 로깅"""
        timestamp = datetime.now().isoformat()

        # CSV 저장
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, frame_id,
                result['translation'][0],
                result['translation'][1],
                result['translation'][2],
                result['euler']['roll'],
                result['euler']['pitch'],
                result['euler']['yaw'],
                result['velocity'][0],
                result['velocity'][1],
                result['velocity'][2],
                result['speed'],
                result['confidence']
            ])

        # JSON 저장용 누적
        self.measurements.append({
            'timestamp': timestamp,
            'frame_id': frame_id,
            **result
        })

    def save_json(self):
        """JSON 최종 저장"""
        with open(self.json_path, 'w') as f:
            json.dump(self.measurements, f, indent=2, default=str)
```

---

*작성: 2025-12-31*
*환경: Intel RealSense D455 + NVIDIA Jetson Orin Nano Super*
*목적: FoundationPose 기반 도장물 속도/각도 측정 시스템 구현 가이드*
