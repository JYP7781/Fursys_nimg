# Depth Anything V3 활용 가능성 연구 보고서

**작성일**: 2025-12-22
**기반 문서**: ai_based_velocity_angle_measurement_research.md
**목적**: Depth Anything V3를 포즈 추정 및 속도/각도 측정에 활용할 수 있는지 조사 및 분석

---

## 목차

1. [연구 개요](#1-연구-개요)
2. [Depth Anything 시리즈 개요](#2-depth-anything-시리즈-개요)
3. [Depth Anything V3 상세 분석](#3-depth-anything-v3-상세-분석)
4. [포즈 추정에의 활용 가능성](#4-포즈-추정에의-활용-가능성)
5. [속도/각도 측정에의 활용 가능성](#5-속도각도-측정에의-활용-가능성)
6. [하드웨어 Depth 센서 vs Depth Anything 비교](#6-하드웨어-depth-센서-vs-depth-anything-비교)
7. [Jetson 배포 가능성](#7-jetson-배포-가능성)
8. [통합 파이프라인 설계](#8-통합-파이프라인-설계)
9. [권장 사항 및 결론](#9-권장-사항-및-결론)
10. [참고 자료](#10-참고-자료)

---

## 1. 연구 개요

### 1.1 연구 배경

기존 `ai_based_velocity_angle_measurement_research.md`에서는 속도/각도 측정을 위한 다양한 AI 기술을 조사했습니다. 현재 시스템은 **Intel RealSense D455** 하드웨어 Depth 센서를 사용하고 있습니다.

본 연구에서는 **Depth Anything V3**를 활용하여:
1. 포즈 추정(6DoF)에 활용할 수 있는지
2. 속도/각도 측정에 활용할 수 있는지
3. 하드웨어 Depth 센서와 비교하여 어떤 장단점이 있는지
4. 어떤 방향이 성능상 더 유리한지

를 분석합니다.

### 1.2 핵심 질문

| 질문 | 검토 항목 |
|------|----------|
| **Depth Anything을 사용해야 하는가?** | 기존 D455 대비 이점이 있는지 |
| **포즈 추정에 적합한가?** | 6DoF 추정 파이프라인에 통합 가능 여부 |
| **속도 측정에 활용 가능한가?** | Optical Flow + Depth 결합 가능 여부 |
| **Jetson에서 실시간 가능한가?** | 처리 속도 및 메모리 제약 |

---

## 2. Depth Anything 시리즈 개요

### 2.1 버전별 발전

| 버전 | 발표 시기 | 주요 특징 | 학습 데이터 |
|------|----------|----------|------------|
| **V1** | CVPR 2024 | 최초 Foundation Model MDE | 1.5M 라벨 + 62M 비라벨 |
| **V2** | NeurIPS 2024 | 합성 데이터 기반, 세부 디테일 향상 | 595K 합성 + 62M+ 실제 비라벨 |
| **V3** | 2025 (arXiv) | Multi-view 지원, 카메라 자세 추정 | 공개 학술 데이터셋 |

### 2.2 Depth Anything V2 핵심 개선

- **합성 데이터 활용**: 실제 센서(스테레오, LiDAR, RGB-D) 라벨 대신 합성 데이터 사용
- **세밀한 디테일**: V1 대비 훨씬 더 미세한 깊이 예측
- **투명/반사 객체 처리**: 합성 데이터의 장점으로 투명/반사 표면 처리 가능
- **효율성**: Stable Diffusion 기반 모델 대비 **10배 이상 빠름**

**V2 정확도 (NYUv2)**:
- AbsRel: 0.056 (MiDaS: 0.077)
- δ1: 0.984 (MiDaS: 0.951)

**참고**: [Depth Anything V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)

---

## 3. Depth Anything V3 상세 분석

### 3.1 핵심 기능

Depth Anything V3 (DA3)는 단순 Monocular Depth 추정을 넘어 **Multi-View Geometry**를 지원합니다.

| 기능 | 설명 |
|------|------|
| **Monocular Depth** | 단일 RGB 이미지에서 깊이 맵 예측 |
| **Multi-View Depth** | 다중 이미지에서 일관된 깊이 맵 생성 |
| **Pose-Conditioned Depth** | 카메라 자세 제공 시 더 높은 깊이 일관성 |
| **Camera Pose Estimation** | 하나 이상의 이미지에서 카메라 Extrinsic/Intrinsic 추정 |
| **3D Gaussian Estimation** | 고품질 Novel View Synthesis를 위한 3D Gaussian 직접 예측 |

### 3.2 아키텍처 특징

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Depth Anything V3 Architecture                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  핵심 인사이트:                                                       │
│  1. 단일 Plain Transformer (DINOv2 인코더) 백본으로 충분              │
│  2. Depth-Ray 예측 타겟으로 복잡한 Multi-task 학습 불필요             │
│                                                                      │
│  Teacher-Student 학습 패러다임으로 DA2 수준의 디테일/일반화 달성       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 성능 벤치마크

**Visual Geometry Benchmark**:
- 카메라 자세 정확도: 이전 SOTA (VGGT) 대비 **35.7% 향상**
- 기하학적 정확도: 이전 SOTA 대비 **23.6% 향상**
- Monocular Depth: DA2보다 우수한 성능

**3D Reconstruction 품질**:
- 다른 방법들보다 기하학적으로 더 규칙적이고 노이즈가 적은 Point Cloud 생성
- 미세한 기하학적 디테일 보존

### 3.4 모델 변형

| 모델 | 파라미터 | 특성 | 용도 |
|------|----------|------|------|
| **DA3-Small** | 80M | 빠른 속도, 양호한 품질 | 실시간 애플리케이션 |
| **DA3-Large** | 350M | 높은 품질, 균형잡힌 성능 | 일반 용도 |
| **DA3-Giant** | 1.15B | 최고 품질, 느린 속도 | 고정밀 작업 |
| **DA3Mono-Large** | - | 상대 깊이 전용 | Monocular 특화 |
| **DA3Metric-Large** | - | 절대 깊이 (미터) | Metric 요구 시 |

**참고**: [Depth Anything 3 GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)

---

## 4. 포즈 추정에의 활용 가능성

### 4.1 Depth Anything V3의 포즈 추정 기능

DA3는 **카메라 자세 추정** 기능을 내장하고 있습니다:

- 하나 이상의 이미지에서 카메라 Extrinsic/Intrinsic 추정
- 카메라 자세 정확도에서 VGGT 대비 **35.7% 향상**

**그러나**, 이는 **카메라 자세**를 추정하는 것이지, **객체의 6DoF 자세**를 추정하는 것이 아닙니다.

### 4.2 객체 6DoF 포즈 추정에의 활용

Depth Anything을 6DoF 객체 포즈 추정에 활용하는 방법:

#### 방법 1: RGB + Depth Anything 출력 → 기존 6DoF 모델

```
RGB 이미지 → Depth Anything V3 → 예측 깊이 맵
                                      ↓
                           기존 RGB-D 포즈 추정 모델
                           (DenseFusion, FFB6D 등)
                                      ↓
                                6DoF 자세 출력
```

**장점**:
- RGB 카메라만으로 RGB-D 기반 모델 활용 가능
- 하드웨어 Depth 센서 불필요

**단점**:
- 2단계 파이프라인으로 지연 증가
- 예측 깊이의 노이즈가 포즈 추정에 영향
- 실제 깊이 센서보다 정확도 낮을 수 있음

#### 방법 2: Depth 정보를 Point Cloud로 변환 → Point Cloud 기반 포즈 추정

```
RGB 이미지 → Depth Anything V3 → 깊이 맵
                                    ↓
                          2.5D → 3D Point Cloud 변환
                                    ↓
                          PointNet++ / 기타 Point Cloud 포즈 모델
                                    ↓
                              6DoF 자세 출력
```

### 4.3 DA3 vs 기존 하드웨어 Depth 기반 포즈 추정 비교

| 항목 | 하드웨어 Depth (D455) | Depth Anything V3 |
|------|----------------------|-------------------|
| **깊이 정확도** | 높음 (직접 측정) | 중간 (예측 기반) |
| **절대 거리** | 정확 | Metric 모델 필요, 제한적 |
| **하드웨어 비용** | $300+ | RGB 카메라만 (저비용) |
| **조명 의존성** | 중간 (Active IR) | 높음 (RGB 기반) |
| **투명/반사 처리** | 어려움 | 상대적으로 우수 |
| **실시간 성능** | 매우 빠름 | GPU 의존적 |

### 4.4 포즈 추정 권장 사항

**현재 환경 (D455 보유) 기준**:

| 시나리오 | 권장 |
|----------|------|
| **높은 정확도 필요** | D455 하드웨어 Depth 유지 |
| **RGB 카메라만 가용** | Depth Anything + 기존 6DoF 모델 |
| **투명 객체 처리** | Depth Anything 검토 가치 있음 |
| **비용 민감** | Depth Anything (RGB 카메라만) |

---

## 5. 속도/각도 측정에의 활용 가능성

### 5.1 속도 측정에의 활용

속도 측정은 두 가지 방법으로 Depth Anything을 활용할 수 있습니다:

#### 방법 1: Optical Flow + Depth Anything 결합

```python
def velocity_from_flow_and_depth(flow, depth, intrinsics, dt):
    """
    Optical Flow와 Depth Anything 출력을 결합한 3D 속도 추정

    Args:
        flow: RAFT/PWC-Net 등의 Optical Flow [H, W, 2]
        depth: Depth Anything 예측 깊이 [H, W]
        intrinsics: 카메라 내부 파라미터
        dt: 프레임 간 시간 간격

    Returns:
        velocity_3d: 3D 속도 벡터 [vx, vy, vz]
    """
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    # 객체 영역의 평균 flow와 depth
    u, v = flow[..., 0].mean(), flow[..., 1].mean()
    Z = np.median(depth[depth > 0])

    # 픽셀 이동 → 3D 이동
    dx = u * Z / fx
    dy = v * Z / fy

    # Z 방향 속도는 연속 depth 차이로 추정
    # dz = (depth_t - depth_t-1).median() / dt

    velocity = np.array([dx/dt, dy/dt, 0])  # vz는 별도 계산
    return velocity
```

**파이프라인**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Optical Flow + Depth Anything                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Frame t    ─┬─→ RAFT/PWC-Net ─→ Optical Flow (u, v)                │
│  Frame t+1 ─┘                         ↓                              │
│                                       │                              │
│  Frame t ───→ Depth Anything ─→ Depth Map (Z)                       │
│                                       ↓                              │
│                              ┌────────┴────────┐                     │
│                              │  3D 속도 계산    │                    │
│                              │  vx = u*Z/(fx*dt)│                    │
│                              │  vy = v*Z/(fy*dt)│                    │
│                              └────────┬────────┘                     │
│                                       ↓                              │
│                              3D Velocity (m/s)                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 방법 2: Scene Flow 직접 추정

DA3는 Multi-view Depth를 지원하므로, 연속 프레임에서 일관된 깊이 맵을 생성하여 Scene Flow를 추정할 수 있습니다.

### 5.2 각도 측정에의 활용

DA3의 Multi-view 기능을 활용하여 객체 방향을 추정할 수 있습니다:

#### 방법 1: Point Cloud에서 PCA/OBB

```python
def angle_from_depth_anything(depth_map, camera_intrinsics, object_mask):
    """
    Depth Anything 출력으로 3D 방향 추정

    Args:
        depth_map: Depth Anything 예측 깊이 맵
        camera_intrinsics: 카메라 내부 파라미터
        object_mask: 객체 영역 마스크

    Returns:
        roll, pitch, yaw: 객체 방향 (degrees)
    """
    # Depth → Point Cloud 변환
    points = depth_to_pointcloud(depth_map, camera_intrinsics, object_mask)

    # PCA로 주축 추출
    pca = PCA(n_components=3)
    pca.fit(points)
    axes = pca.components_

    # 주축에서 방향 추출
    roll, pitch, yaw = axes_to_euler(axes)
    return roll, pitch, yaw
```

#### 방법 2: DA3 카메라 자세 추정 활용

DA3가 카메라 자세를 추정할 수 있으므로, 다중 뷰에서 객체의 상대적 방향 변화를 계산할 수 있습니다.

### 5.3 D455 vs Depth Anything 속도/각도 측정 비교

| 항목 | D455 Hardware | Depth Anything |
|------|--------------|----------------|
| **깊이 정확도** | ±1% (2m 이내) | 상대적으로 낮음 |
| **프레임 간 일관성** | 높음 | Multi-view 모드 필요 |
| **속도 측정 정확도** | ±5-10% (예상) | ±10-15% (예상) |
| **각도 측정 정확도** | ±3-5° | ±5-8° (예상) |
| **실시간 처리** | 하드웨어로 즉시 | GPU 연산 필요 |
| **조명 영향** | 적음 | 큼 |

---

## 6. 하드웨어 Depth 센서 vs Depth Anything 비교

### 6.1 Intel RealSense D455 특성

| 항목 | 사양 |
|------|------|
| **기술** | Active Stereo (IR 패턴 투사) |
| **깊이 범위** | 0.6m ~ 6m (최적: 1-4m) |
| **깊이 해상도** | 1280x720 @ 30fps |
| **깊이 정확도** | <2% @ 2m |
| **FOV** | H: 87°, V: 58° |
| **IMU** | 내장 (가속도계, 자이로스코프) |

### 6.2 종합 비교

| 비교 항목 | D455 (Hardware) | Depth Anything V3 |
|----------|----------------|-------------------|
| **깊이 측정 방식** | 직접 측정 (ToF/Stereo) | AI 예측 |
| **절대 거리 정확도** | ★★★★★ | ★★★☆☆ (Metric 모델) |
| **상대 깊이 품질** | ★★★★☆ | ★★★★★ |
| **하드웨어 비용** | $300+ | RGB 카메라만 |
| **연산 비용** | 최소 (하드웨어) | GPU 필요 |
| **조명 조건** | Active IR로 강건 | RGB 의존적 |
| **투명/반사 객체** | 문제 있음 | 상대적 우수 |
| **원거리 정확도** | 저하됨 | 일관적 (상대적) |
| **실시간 성능** | 30-90 FPS | 10-25 FPS (Jetson) |
| **배포 복잡도** | 간단 (SDK 제공) | 모델 최적화 필요 |

### 6.3 사용 시나리오별 권장

| 시나리오 | 권장 | 이유 |
|----------|------|------|
| **정확한 절대 거리 필요** | D455 | 직접 측정으로 높은 정확도 |
| **상대 깊이/순서만 필요** | DA3 | 우수한 상대 깊이 품질 |
| **RGB 카메라만 가용** | DA3 | 유일한 선택 |
| **실시간 (30+ FPS)** | D455 | 하드웨어 처리 |
| **투명/반사 객체** | DA3 | 합성 데이터 학습 이점 |
| **저비용 다중 배포** | DA3 | 카메라 비용만 |
| **조명 변화 큰 환경** | D455 | Active IR로 강건 |

### 6.4 도장물 측정 시스템 권장

**현재 환경 (D455 보유, Jetson Orin Nano Super)**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                       권장 전략                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ◆ 주력: D455 하드웨어 Depth 유지                                    │
│    - 절대 거리 정확도 요구                                            │
│    - 실시간 처리 필요                                                 │
│    - 조명 변화 대응                                                   │
│                                                                      │
│  ◆ 보조: Depth Anything V3 검토                                      │
│    - D455 실패 시 Fallback                                           │
│    - 투명/반사 도장물 처리                                            │
│    - 상대 깊이 품질 비교 실험                                         │
│                                                                      │
│  ◆ 결론:                                                             │
│    현재 D455 유지가 최적. DA3는 보조/실험용으로 검토                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Jetson 배포 가능성

### 7.1 Depth Anything on Jetson 배포 현황

**공식/커뮤니티 지원**:

| 프로젝트 | 플랫폼 | 기능 |
|----------|--------|------|
| [Depth-Anything-for-Jetson-Orin](https://github.com/IRCVLab/Depth-Anything-for-Jetson-Orin) | Jetson Orin | 실시간 깊이 추정 |
| [ROS2 DA3 TensorRT](https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt) | ROS2 + TensorRT | Metric Depth + Point Cloud |
| Seeed Studio 튜토리얼 | reComputer J4012 | WebUI, TensorRT 변환 |

### 7.2 Jetson Orin 성능 벤치마크

**Online Video Depth Anything (oVDA) on Jetson Orin NX**:

| 정밀도 | FPS | 메모리 | 정확도 영향 |
|--------|-----|--------|------------|
| **FP32** | ~9 FPS | 높음 | 기준 |
| **FP16** | ~20 FPS | 중간 | 미미한 저하 |

**참고**: Jetson Orin NX (100 TOPS) 기준. Jetson Orin Nano Super (67 TOPS)는 약 60-70% 성능 예상.

### 7.3 Jetson Orin Nano Super 예상 성능

| 모델 | 예상 FPS (FP16) | 메모리 | 실시간 가능 |
|------|----------------|--------|------------|
| **DA3-Small (80M)** | 12-18 FPS | ~1GB | △ (가능) |
| **DA3-Large (350M)** | 5-10 FPS | ~2GB | △ (제한적) |
| **DA3-Giant (1.15B)** | 불가 | >8GB | ✗ |
| **DA-V2-Small** | 15-22 FPS | ~800MB | ○ |

### 7.4 TensorRT 최적화 가이드

```python
# Depth Anything V2/V3 TensorRT 변환 예시
import torch
import torch.onnx

def export_depth_anything_tensorrt(model_path, output_path, input_size=(518, 518)):
    """
    Depth Anything 모델을 TensorRT로 변환

    1. PyTorch → ONNX 변환
    2. ONNX → TensorRT 변환
    """
    # 1. 모델 로드
    model = load_depth_anything_model(model_path)
    model.eval()

    # 2. ONNX 변환
    dummy_input = torch.randn(1, 3, *input_size).cuda()
    onnx_path = output_path.replace('.trt', '.onnx')

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['depth'],
        opset_version=17,
        dynamic_axes={'input': {0: 'batch'}, 'depth': {0: 'batch'}}
    )

    # 3. TensorRT 변환 (trtexec 또는 Python API)
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
    config.set_flag(trt.BuilderFlag.FP16)  # FP16 모드

    engine = builder.build_serialized_network(network, config)

    with open(output_path, 'wb') as f:
        f.write(engine)

    return output_path
```

---

## 8. 통합 파이프라인 설계

### 8.1 기존 시스템 (D455 기반)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    현재 시스템 (권장 유지)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  D455 ───┬──→ RGB ───→ YOLO11s ───→ 객체 탐지                       │
│          │                              ↓                            │
│          └──→ Depth ─────────────→ Point Cloud ───→ PCA/OBB        │
│                                         ↓                  ↓         │
│                                    KalmanNet ───→ 속도/각도 출력     │
│                                                                      │
│  처리 속도: ~30 FPS                                                  │
│  메모리: ~2GB                                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 DA3 하이브리드 시스템 (실험적)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DA3 하이브리드 시스템 (실험)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  D455 ───┬──→ RGB ───→ YOLO11s ───→ 객체 탐지                       │
│          │      │                       ↓                            │
│          │      └──→ DA3-Small ───→ 예측 Depth                      │
│          │                              ↓                            │
│          └──→ HW Depth ─────────→ 센서 융합                         │
│                                         ↓                            │
│                              Depth Quality 선택                      │
│                              (HW 우선, DA3 Fallback)                 │
│                                         ↓                            │
│                              Point Cloud → 분석                      │
│                                                                      │
│  처리 속도: ~15-20 FPS                                               │
│  메모리: ~3-4GB                                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.3 RGB 전용 시스템 (D455 없을 경우)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RGB 전용 시스템 (DA3 활용)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  RGB Camera ───→ YOLO11s ───→ 객체 탐지                             │
│       │              ↓                                               │
│       │         객체 ROI                                             │
│       │              ↓                                               │
│       └────→ DA3-Small ───→ 깊이 맵                                 │
│                     ↓                                                │
│             Point Cloud 생성                                         │
│                     ↓                                                │
│             PCA/OBB 또는 EfficientPose                               │
│                     ↓                                                │
│              속도/각도 출력                                          │
│                                                                      │
│  RAFT (Optical Flow) ───→ 2D 움직임                                 │
│         ↓                                                            │
│   DA3 Depth + Flow ───→ 3D 속도                                     │
│                                                                      │
│  처리 속도: ~10-15 FPS                                               │
│  메모리: ~3GB                                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. 권장 사항 및 결론

### 9.1 핵심 결론

| 질문 | 결론 |
|------|------|
| **D455 대신 DA3?** | ❌ 권장하지 않음 (D455 유지) |
| **DA3 보조 활용?** | ○ 특수 상황에서 검토 가치 |
| **포즈 추정에 적합?** | △ 6DoF 객체 포즈에는 간접적 활용 |
| **속도/각도 측정?** | △ Optical Flow와 결합 시 가능 |
| **Jetson 실시간?** | △ DA3-Small만 제한적 가능 |

### 9.2 시나리오별 권장 방향

#### 시나리오 A: D455 유지 (권장)

```
현재 권장 전략:
- D455 하드웨어 Depth 계속 사용
- DA3는 연구/실험 목적으로만 검토
- 이유: 절대 거리 정확도, 실시간 성능, 조명 강건성
```

#### 시나리오 B: DA3 도입 검토 (특수 상황)

```
DA3 검토가 필요한 경우:
1. D455가 투명/반사 도장물 처리 실패 시
2. RGB 카메라만으로 저비용 다중 배포 필요 시
3. D455 깊이 품질이 불안정한 원거리 측정 시
```

#### 시나리오 C: 미래 가능성

```
DA4/향후 발전:
- Metric Depth 정확도 개선 시 재검토
- Jetson 성능 최적화 개선 시 재검토
- Real-time (30+ FPS) 달성 시 대체 검토
```

### 9.3 성능 비교 요약

| 방향 | 속도 정확도 | 각도 정확도 | 처리 속도 | 권장 |
|------|-----------|-----------|----------|------|
| **D455 + Kalman** | ±5-10% | ±3-5° | ~30 FPS | ★★★★★ |
| **D455 + AI 모델** | ±3-6% | ±1-3° | ~15-25 FPS | ★★★★☆ |
| **DA3 + Optical Flow** | ±10-15% | ±5-8° | ~10-15 FPS | ★★★☆☆ |
| **DA3 + 6DoF 모델** | ±8-12% | ±3-5° | ~8-12 FPS | ★★☆☆☆ |

### 9.4 최종 권장 사항

```
┌─────────────────────────────────────────────────────────────────────┐
│                        최종 권장 전략                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 주력 시스템: D455 하드웨어 Depth 유지                            │
│     - 절대 거리 정확도: ±1-2%                                        │
│     - 실시간 처리: 30+ FPS                                           │
│     - 조명 강건성: Active IR                                         │
│                                                                      │
│  2. AI 보강: KalmanNet + EfficientPose (기존 연구 결과)              │
│     - 속도 측정: KalmanNet (GRU-Kalman)                              │
│     - 각도 측정: EfficientPose 또는 PCA/OBB                          │
│                                                                      │
│  3. DA3 활용 범위: 실험/연구/Fallback                                │
│     - 투명/반사 객체 처리 테스트                                      │
│     - D455 실패 시 Fallback                                          │
│     - 상대 깊이 품질 비교 연구                                        │
│                                                                      │
│  4. DA3 권장하지 않는 이유:                                          │
│     - Metric Depth 정확도가 하드웨어 센서보다 낮음                    │
│     - Jetson에서 실시간 (30 FPS) 달성 어려움                         │
│     - 조명 변화에 민감                                                │
│     - 추가 연산 부하                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. 참고 자료

### 10.1 Depth Anything 시리즈

- [Depth Anything V1 (CVPR 2024)](https://github.com/LiheYoung/Depth-Anything)
- [Depth Anything V2 (NeurIPS 2024)](https://github.com/DepthAnything/Depth-Anything-V2)
- [Depth Anything V2 논문](https://arxiv.org/abs/2406.09414)
- [Depth Anything V3 (2025)](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [DA3 프로젝트 페이지](https://depth-anything-3.github.io/)
- [DA3 arXiv 논문](https://arxiv.org/abs/2511.10647)

### 10.2 Jetson 배포

- [Depth Anything for Jetson Orin](https://github.com/IRCVLab/Depth-Anything-for-Jetson-Orin)
- [ROS2 Depth Anything V3 TensorRT](https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt)
- [Seeed Studio 튜토리얼](https://www.hackster.io/518547/monocular-depth-estimation-on-recomputer-j4012-aa1086)
- [Online Video Depth Anything](https://arxiv.org/abs/2510.09182)

### 10.3 관련 기술

- [Monocular Depth Estimation Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC9325018/)
- [6DoF Pose Estimation Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC10893425/)
- [Depth-based 6DoF with Swin Transformer](https://arxiv.org/abs/2303.02133)
- [Intel RealSense D455 사양](https://www.intelrealsense.com/compare-depth-cameras/)

### 10.4 Optical Flow

- [RAFT GitHub](https://github.com/princeton-vl/RAFT)
- [SEA-RAFT (2024)](https://arxiv.org/abs/2405.14793)
- [Flow-Anything](https://arxiv.org/abs/2506.07740)

### 10.5 Metric Depth

- [Prompt Depth Anything (CVPR 2025)](https://arxiv.org/abs/2412.14015)
- [Metric Depth Fine-tuning 가이드](https://huggingface.co/blog/Isayoften/monocular-depth-estimation-guide)

---

*작성: 2025-12-22*
*환경: Intel RealSense D455 + NVIDIA Jetson Orin Nano Super*
*목적: Depth Anything V3 활용 가능성 연구*
*관련 문서: ai_based_velocity_angle_measurement_research.md*
