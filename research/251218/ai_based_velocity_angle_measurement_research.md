# AI 기반 도장물 속도/각도 측정 기술 연구 보고서

**작성일**: 2025-12-18
**기반 문서**: comprehensive_improvement_research_2025.md, implementation_design_guide.md
**목적**: 딥러닝(AI)을 활용한 도장물의 속도 및 각도 측정 방법 조사 및 분석

---

## 목차

1. [연구 개요](#1-연구-개요)
2. [기존 시스템 분석](#2-기존-시스템-분석)
3. [속도 측정을 위한 AI 기술](#3-속도-측정을-위한-ai-기술)
4. [각도 측정을 위한 AI 기술](#4-각도-측정을-위한-ai-기술)
5. [통합 추적 및 상태 추정 AI](#5-통합-추적-및-상태-추정-ai)
6. [Jetson 환경 최적화](#6-jetson-환경-최적화)
7. [기술 비교 및 권장 사항](#7-기술-비교-및-권장-사항)
8. [구현 전략](#8-구현-전략)
9. [결론](#9-결론)
10. [참고 자료](#10-참고-자료)

---

## 1. 연구 개요

### 1.1 연구 배경

기존 `implementation_design_guide.md`와 `comprehensive_improvement_research_2025.md`에서는 속도 측정에 **Kalman Filter (CA 모델)**를, 각도 측정에 **PCA/OBB** 방식을 권장했습니다. 이러한 전통적 방식은 연산량이 적고 구현이 간단한 장점이 있지만, 본 연구에서는 **딥러닝(AI) 기반** 접근법을 탐색하여 더 높은 정확도와 강건성을 확보할 수 있는지 조사합니다.

### 1.2 연구 목표

1. **속도 측정**: Optical Flow, Scene Flow 등 딥러닝 기반 모션 추정 기술 조사
2. **각도 측정**: 6DoF Pose Estimation 딥러닝 모델 조사
3. **통합 추적**: 딥러닝 기반 객체 추적 및 상태 추정 기술 조사
4. **실용성 평가**: Jetson Orin Nano Super 환경에서의 적용 가능성 분석

### 1.3 연구 범위

| 영역 | 조사 내용 |
|------|----------|
| **속도 측정** | Optical Flow (FlowNet, RAFT, PWC-Net), Scene Flow (FlowNet3D, RMS-FlowNet++) |
| **각도 측정** | 6DoF Pose Estimation (DenseFusion, FFB6D, PoseCNN, EfficientPose) |
| **추적/상태추정** | 딥러닝 기반 Kalman Filter (KalmanNet, KalmanFormer), Transformer 추적기 |
| **Edge 최적화** | Jetson TensorRT 최적화, 경량 모델 |

---

## 2. 기존 시스템 분석

### 2.1 현재 권장 방식 (기존 문서 기반)

| 측정 항목 | 현재 권장 방식 | 정확도 | 처리 시간 |
|----------|---------------|--------|----------|
| **속도** | CA Kalman Filter (FilterPy) | ±5-10% | <1ms |
| **Pitch** | PCA/OBB (Open3D) | ±3-5° | ~5ms |
| **Yaw** | PCA/OBB (Open3D) | ±2-4° | ~5ms |
| **Roll** | PCA/OBB (Open3D) | ±3-5° | ~5ms |

### 2.2 기존 방식의 한계

1. **Kalman Filter 한계**:
   - 수동으로 모션 모델 설계 필요 (CV, CA, CTRV 등)
   - 노이즈 분포 가정 필요 (가우시안)
   - 비선형/비정상 움직임 처리 어려움
   - 급격한 방향 전환 시 추적 실패 가능

2. **PCA/OBB 한계**:
   - Point Cloud 품질에 민감
   - 대칭 객체에서 모호성 발생
   - 가려짐(occlusion)에 취약
   - 객체 특성을 학습하지 않음

---

## 3. 속도 측정을 위한 AI 기술

### 3.1 2D Optical Flow 기반 방법

#### 3.1.1 FlowNet / FlowNet2

**개요**: CNN을 사용하여 두 연속 이미지 간의 픽셀 단위 움직임(Optical Flow)을 학습

| 항목 | FlowNet | FlowNet2 |
|------|---------|----------|
| **모델 크기** | ~38M params | ~162M params |
| **정확도** | 중간 | 높음 (FlowNet 대비 50% 오차 감소) |
| **속도** | 빠름 | 중간 (GPU 필요) |
| **Jetson 적합성** | △ | ✗ (메모리 과다) |

**참고**: [FlowNet2.0 CVPR 2017](https://arxiv.org/abs/1612.01925)

#### 3.1.2 RAFT (Recurrent All-Pairs Field Transforms)

**개요**: ECCV 2020 Best Paper. 반복적 업데이트를 통해 고정밀 Optical Flow 추정

| 항목 | 내용 |
|------|------|
| **구조** | Feature Encoder + Correlation Volume + GRU 반복 업데이트 |
| **모델 크기** | RAFT-Large: 4.8M params, RAFT-Small: 1M params |
| **정확도** | KITTI F1-all 5.10% (당시 SOTA), Sintel 2.855 EPE |
| **속도** | ~10 FPS (GPU), RAFT-Small은 더 빠름 |
| **Jetson 적합성** | ★★★☆☆ (RAFT-Small 사용 시) |

**2024 업데이트 (Ef-RAFT)**:
- Sintel 10%, KITTI 5% 성능 향상
- 속도는 33% 감소하나 메모리는 13%만 증가

**참고**: [RAFT GitHub](https://github.com/princeton-vl/RAFT), [Ef-RAFT 2024](https://github.com/n3slami/Ef-RAFT)

#### 3.1.3 PWC-Net (Pyramid, Warping, Cost Volume)

**개요**: 피라미드 구조, 워핑, 비용 볼륨을 활용한 경량 고정밀 Optical Flow 모델

| 항목 | 내용 |
|------|------|
| **모델 크기** | FlowNet2 대비 **17배 작음** |
| **정확도** | MPI Sintel, KITTI 2015에서 우수한 성능 |
| **속도** | ~35 FPS (1024×436 해상도) |
| **Jetson 적합성** | ★★★★☆ |

**참고**: [PWC-Net GitHub](https://github.com/NVlabs/PWC-Net)

#### 3.1.4 Optical Flow → 속도 변환

```python
def optical_flow_to_velocity(flow, depth, intrinsics, dt):
    """
    Optical Flow와 Depth를 결합하여 3D 속도 계산

    Args:
        flow: Optical flow [H, W, 2] (u, v 방향 픽셀 이동)
        depth: Depth 이미지 [H, W] (미터 단위)
        intrinsics: 카메라 내부 파라미터 {fx, fy, cx, cy}
        dt: 프레임 간 시간 간격 (초)

    Returns:
        velocity_3d: 3D 속도 벡터 [vx, vy, vz]
    """
    fx, fy = intrinsics['fx'], intrinsics['fy']

    # 중앙 영역의 평균 flow와 depth
    h, w = flow.shape[:2]
    roi = slice(h//4, 3*h//4), slice(w//4, 3*w//4)

    mean_flow = np.mean(flow[roi], axis=(0,1))  # [u, v]
    mean_depth = np.median(depth[roi])

    # 픽셀 이동 → 3D 이동
    dx = mean_flow[0] * mean_depth / fx
    dy = mean_flow[1] * mean_depth / fy

    # Z 방향 속도는 연속 depth 차이로 추정
    # dz = (depth_t - depth_t-1) / dt

    # 속도 = 이동 / 시간
    vx = dx / dt
    vy = dy / dt

    return np.array([vx, vy, 0])  # vz는 별도 계산 필요
```

### 3.2 3D Scene Flow 기반 방법

Scene Flow는 3D 공간에서의 움직임을 직접 추정하므로, 속도 측정에 더 적합합니다.

#### 3.2.1 FlowNet3D

**개요**: 3D Point Cloud에서 직접 Scene Flow를 학습하는 최초의 딥러닝 모델

| 항목 | 내용 |
|------|------|
| **입력** | 연속 두 프레임의 Point Cloud |
| **출력** | 각 포인트의 3D 움직임 벡터 (scene flow) |
| **백본** | PointNet++ 기반 |
| **모델 크기** | ~15MB |
| **정확도** | FlyingThings3D, KITTI에서 검증 |
| **Jetson 적합성** | ★★★☆☆ |

**핵심 모듈**:
1. **Point Feature Learning**: PointNet++ 기반 특징 추출
2. **Flow Embedding Layer**: 두 프레임 간 대응 관계 학습
3. **Flow Refinement**: 추정된 flow 정제

**참고**: [FlowNet3D GitHub](https://github.com/xingyul/flownet3d), [FlowNet3D Paper](https://arxiv.org/abs/1806.01411)

#### 3.2.2 RMS-FlowNet++ (2024 최신)

**개요**: 대규모 Point Cloud에서 효율적이고 강건한 Scene Flow 추정

| 항목 | 내용 |
|------|------|
| **특징** | Random Sampling 사용 (FPS 대체), 250K 포인트 처리 가능 |
| **성능** | FlyingThings3D, KITTI에서 SOTA |
| **효율성** | 기존 방법 대비 빠른 예측, 낮은 메모리 사용 |
| **Jetson 적합성** | ★★★☆☆ |

**장점**:
- 기존 FPS(Farthest Point Sampling) 대신 Random Sampling으로 효율성 향상
- Flow Embedding Block으로 더 작은 대응 집합 사용
- Dense Point Cloud (250K+) 처리 가능

**참고**: [RMS-FlowNet++ IJCV 2024](https://link.springer.com/article/10.1007/s11263-024-02093-9)

#### 3.2.3 HALFlow (Hierarchical Attention Learning)

**개요**: 계층적 어텐션 메커니즘을 활용한 Scene Flow 추정

| 항목 | 내용 |
|------|------|
| **구조** | 2단계 어텐션 메커니즘 |
| **장점** | 올바른 대응 영역에 더 많은 attention 할당 |
| **백본** | PointNet++ 기반 |

**참고**: [HALFlow GitHub](https://github.com/IRMVLab/HALFlow)

#### 3.2.4 Scene Flow → 속도 변환

```python
def scene_flow_to_velocity(scene_flow, dt, roi_mask=None):
    """
    Scene Flow에서 객체 속도 추출

    Args:
        scene_flow: 각 포인트의 3D 이동 벡터 [N, 3]
        dt: 프레임 간 시간 간격 (초)
        roi_mask: 관심 영역 마스크 (객체 영역만 선택)

    Returns:
        velocity: 객체의 평균 3D 속도 [vx, vy, vz]
        speed: 속력 (m/s)
    """
    if roi_mask is not None:
        flow = scene_flow[roi_mask]
    else:
        flow = scene_flow

    # 이상치 제거 (IQR 방식)
    q1 = np.percentile(np.linalg.norm(flow, axis=1), 25)
    q3 = np.percentile(np.linalg.norm(flow, axis=1), 75)
    iqr = q3 - q1
    mask = (np.linalg.norm(flow, axis=1) >= q1 - 1.5*iqr) & \
           (np.linalg.norm(flow, axis=1) <= q3 + 1.5*iqr)

    # 평균 이동 계산
    mean_displacement = np.mean(flow[mask], axis=0)

    # 속도 = 이동 / 시간
    velocity = mean_displacement / dt
    speed = np.linalg.norm(velocity)

    return velocity, speed
```

### 3.3 딥러닝 기반 Optical Flow 속도 측정 정확도

| 방법 | EPE (Endpoint Error) | 속도 추정 상대 오차 | 방향 오차 |
|------|---------------------|-------------------|----------|
| **PIV (전통)** | - | 19-42% | 14-44° |
| **DLOF (딥러닝)** | - | 23-29% | 17-29° |
| **RAFT** | 2.855 (Sintel) | ~10-15% (추정) | ~5-10° |
| **FlowNet3D** | 0.0553m (EPE3D) | ~8-12% | - |

**참고**: [Deep-learning Optical Flow 연구](https://pubs.rsc.org/en/content/articlehtml/2024/sm/d4sm00483c)

---

## 4. 각도 측정을 위한 AI 기술

### 4.1 6DoF Pose Estimation 개요

6DoF(6 Degrees of Freedom) Pose Estimation은 객체의 3D 위치(x, y, z)와 방향(roll, pitch, yaw)을 동시에 추정합니다.

### 4.2 RGB-D 기반 6DoF Pose Estimation 모델

#### 4.2.1 PoseCNN

**개요**: RGB 이미지에서 6DoF 자세를 추정하는 CNN 기반 모델

| 항목 | 내용 |
|------|------|
| **입력** | RGB 이미지 |
| **출력** | 6DoF 자세 (위치 + 방향) |
| **방식** | Segmentation → 중심점 회귀 → 깊이 추정 → 자세 |
| **ICP 정제** | 필요 (정확도 향상을 위해) |
| **Jetson 적합성** | ★★☆☆☆ (ICP 필요 시 느림) |

**참고**: [PoseCNN GitHub](https://github.com/yuxng/PoseCNN)

#### 4.2.2 DenseFusion

**개요**: RGB와 Depth 정보를 픽셀 단위로 밀집 융합하여 6DoF 추정

| 항목 | 내용 |
|------|------|
| **입력** | RGB-D 이미지 |
| **구조** | RGB Encoder + Point Cloud Encoder + Dense Fusion |
| **특징** | 픽셀 단위 특징 융합, Occlusion에 강건 |
| **속도** | PoseCNN+ICP 대비 **200배 빠름** |
| **정확도** | YCB-Video에서 PoseCNN+ICP 대비 3.5% 향상 |
| **Jetson 적합성** | ★★★☆☆ |

**아키텍처**:
```
RGB Image → CNN Encoder → RGB Embedding
                                       ↘
                                         → Dense Fusion → Pose Prediction
                                       ↗
Depth → Point Cloud → PointNet → Geometry Embedding
```

**참고**: [DenseFusion GitHub](https://github.com/j96w/DenseFusion), [DenseFusion Paper](https://arxiv.org/abs/1901.04780)

#### 4.2.3 FFB6D (Full Flow Bidirectional Fusion)

**개요**: 양방향 전체 흐름 융합을 통한 6DoF 추정

| 항목 | 내용 |
|------|------|
| **입력** | RGB-D 이미지 |
| **특징** | 인코딩/디코딩 모든 레이어에서 RGB-Depth 융합 |
| **정확도** | **98.09%** (2024 테스트 기준) |
| **단점** | 데이터 품질 의존성 높음, 후처리 시간 소요 |
| **Jetson 적합성** | ★★★☆☆ |

**장점**:
- DenseFusion보다 더 깊은 융합
- 지역/전역 정보 최대 활용
- Keypoint Localization 단순화

**참고**: [FFB6D Paper](https://arxiv.org/abs/2103.02242)

#### 4.2.4 EfficientPose

**개요**: 효율적인 End-to-End 6DoF 추정, 다중 객체 최적화

| 항목 | 내용 |
|------|------|
| **특징** | Translation/Rotation 별도 서브네트워크 |
| **정확도** | **97.05%** (2024 테스트 기준) |
| **속도** | 후처리 단계 제거로 빠름 |
| **Jetson 적합성** | ★★★★☆ |

**장점**:
- 후처리 제거로 연산 비용 절감
- 데이터 증강 기법으로 소규모 데이터셋에도 일반화
- 다중 객체 처리에 효율적

**참고**: [EfficientPose 연구](https://www.sciencedirect.com/science/article/pii/S2590123024017110)

#### 4.2.5 RDPN6D (2024 CVPR)

**개요**: Residual 기반 Dense Point-wise Network

| 항목 | 내용 |
|------|------|
| **발표** | CVPR 2024 |
| **특징** | RGB-D 기반 잔차 학습 |
| **정확도** | YCB-Video SOTA |

**참고**: [RDPN6D Paper](https://arxiv.org/html/2405.08483v1)

### 4.3 Point Cloud 기반 방향 추정

#### 4.3.1 PointNet/PointNet++ 기반 방향 추정

**개요**: Point Cloud를 직접 입력으로 받아 방향 추정

| 항목 | PointNet | PointNet++ |
|------|----------|------------|
| **특징** | 순서 불변 처리 | 계층적 지역 특징 학습 |
| **방향 추정** | 전역 특징에서 방향 회귀 | 다중 스케일 특징에서 방향 회귀 |
| **Jetson 적합성** | ★★★★☆ | ★★★★☆ |

**응용**:
- **3D Orientation Estimation**: Partial Point Cloud에서 방향 및 클래스 추정
- **PCPNET**: 노이즈 Point Cloud에서 Normal/Curvature 추정

**참고**: [PointNet GitHub](https://github.com/charlesq34/pointnet), [PointNet++ GitHub](https://github.com/charlesq34/pointnet2)

### 4.4 6DoF → 각도 추출

```python
from scipy.spatial.transform import Rotation

def pose_to_angles(pose_matrix):
    """
    6DoF 변환 행렬에서 Roll, Pitch, Yaw 추출

    Args:
        pose_matrix: 4x4 변환 행렬 또는 3x3 회전 행렬

    Returns:
        roll, pitch, yaw: 오일러 각도 (degrees)
    """
    if pose_matrix.shape == (4, 4):
        rotation_matrix = pose_matrix[:3, :3]
    else:
        rotation_matrix = pose_matrix

    # scipy의 Rotation 클래스 사용
    r = Rotation.from_matrix(rotation_matrix)

    # 'xyz' 순서로 오일러 각도 추출 (roll, pitch, yaw)
    euler = r.as_euler('xyz', degrees=True)

    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]

    return roll, pitch, yaw
```

### 4.5 6DoF Pose Estimation 정확도 비교

| 모델 | YCB-Video ADD-S | LineMOD ADD | 속도 | Jetson 적합성 |
|------|----------------|-------------|------|--------------|
| **PoseCNN** | 61.3% | 55.9% | 느림 (ICP) | ★★☆☆☆ |
| **PoseCNN + ICP** | 78.0% | - | 매우 느림 | ★☆☆☆☆ |
| **DenseFusion** | 82.9% | 94.3% | 빠름 | ★★★☆☆ |
| **FFB6D** | **92.7%** | **99.4%** | 중간 | ★★★☆☆ |
| **EfficientPose** | 88.5% | - | 빠름 | ★★★★☆ |

---

## 5. 통합 추적 및 상태 추정 AI

### 5.1 딥러닝 강화 Kalman Filter

전통적인 Kalman Filter의 한계를 딥러닝으로 극복하는 하이브리드 접근법입니다.

#### 5.1.1 KalmanNet

**개요**: GRU(Gated Recurrent Unit)를 사용하여 Kalman Gain을 학습

| 항목 | 내용 |
|------|------|
| **구조** | Kalman Filter + GRU 네트워크 |
| **특징** | 모델 불일치 및 비선형 시스템에서 우수한 성능 |
| **장점** | 전통적 KF의 수학적 구조 유지 + 학습 능력 |
| **Jetson 적합성** | ★★★★☆ |

**KalmanNet 장점**:
- 노이즈 특성을 자동으로 학습
- 비선형 움직임 모델링 가능
- 클래스별 모델 불필요
- 기존 KF 대비 정확도 향상

**참고**: [KalmanNet Paper](https://arxiv.org/pdf/2107.10043)

#### 5.1.2 KalmanFormer (2024)

**개요**: Transformer를 사용하여 Kalman Gain을 모델링

| 항목 | 내용 |
|------|------|
| **구조** | Kalman Filter + Transformer |
| **특징** | 노이즈 파라미터 사전 지식 불필요 |
| **응용** | 비선형 조건, 부분 정보 시나리오 |
| **Jetson 적합성** | ★★★☆☆ |

**참고**: [KalmanFormer Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11747084/)

#### 5.1.3 GRU-Kalman Filter 구현 예시

```python
import torch
import torch.nn as nn

class GRUKalmanFilter(nn.Module):
    """GRU 강화 Kalman Filter"""

    def __init__(self, state_dim=9, meas_dim=3, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        # GRU for learning Kalman Gain
        self.gru = nn.GRU(
            input_size=state_dim + meas_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Output layer for Kalman Gain
        self.fc_gain = nn.Linear(hidden_dim, state_dim * meas_dim)

        # State transition model (learnable)
        self.fc_F = nn.Linear(state_dim, state_dim)

    def forward(self, state, measurement, hidden=None):
        """
        Args:
            state: 현재 상태 [batch, state_dim]
            measurement: 측정값 [batch, meas_dim]
            hidden: GRU hidden state

        Returns:
            next_state: 업데이트된 상태
            hidden: 새 hidden state
        """
        # 입력 결합
        x = torch.cat([state, measurement], dim=-1).unsqueeze(1)

        # GRU를 통해 Kalman Gain 학습
        gru_out, hidden = self.gru(x, hidden)

        # Kalman Gain 계산
        K = self.fc_gain(gru_out.squeeze(1))
        K = K.view(-1, self.state_dim, self.meas_dim)

        # 상태 예측
        state_pred = self.fc_F(state)

        # 측정 업데이트
        innovation = measurement - state_pred[:, :self.meas_dim]
        state_update = torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)

        next_state = state_pred + state_update

        return next_state, hidden
```

### 5.2 딥러닝 기반 객체 추적

#### 5.2.1 ByteTrack

**개요**: 모든 Detection Box를 활용한 Multi-Object Tracking

| 항목 | 내용 |
|------|------|
| **특징** | 저신뢰도 detection도 활용 |
| **MOTA** | 80.3% (MOT17) |
| **ID Switches** | 159회 (매우 낮음) |
| **속도** | 171 FPS |
| **Jetson 적합성** | ★★★★★ |

**속도 추정 방식**:
- Kalman Filter로 위치/속도 예측
- IoU 기반 매칭

**참고**: [ByteTrack GitHub](https://github.com/FoundationVision/ByteTrack)

#### 5.2.2 DeepSORT

**개요**: Deep Association Metric을 활용한 추적

| 항목 | 내용 |
|------|------|
| **특징** | 외관 특징 + 모션 정보 결합 |
| **장점** | ID 유지 능력 우수, Occlusion에 강건 |
| **Kalman Filter** | 위치, 속도, 가속도 상태 추정 |
| **Jetson 적합성** | ★★★★☆ |

**참고**: [DeepSORT 가이드](https://www.ikomia.ai/blog/deep-sort-object-tracking-guide)

#### 5.2.3 Transformer 기반 추적기

**개요**: Attention 메커니즘을 활용한 3D 객체 추적

| 모델 | 특징 | Jetson 적합성 |
|------|------|--------------|
| **LTTR** | LiDAR Point Cloud + Transformer | ★★★☆☆ |
| **TrackFormer** | End-to-end Transformer MOT | ★★☆☆☆ |
| **TLtrack** | Transformer + Linear 하이브리드 | ★★★☆☆ |

**참고**: [3D Object Tracking with Transformer](https://arxiv.org/abs/2110.14921)

### 5.3 추적기에서 속도/각도 추출

```python
class TrackerWithVelocityAngle:
    """추적기 기반 속도/각도 추정"""

    def __init__(self, dt=1/30.0):
        self.dt = dt
        self.tracks = {}  # track_id → track_state

    def update(self, detections):
        """
        Args:
            detections: List of {
                'id': track_id,
                'bbox_3d': [x, y, z, w, h, d],  # 3D 바운딩 박스
                'orientation': [roll, pitch, yaw]
            }

        Returns:
            results: track_id별 속도/각도 변화
        """
        results = {}

        for det in detections:
            track_id = det['id']
            pos = np.array(det['bbox_3d'][:3])
            orientation = np.array(det['orientation'])

            if track_id in self.tracks:
                prev = self.tracks[track_id]

                # 속도 계산
                velocity = (pos - prev['position']) / self.dt
                speed = np.linalg.norm(velocity)

                # 각도 변화 계산
                angle_change = orientation - prev['orientation']
                # -180 ~ 180 범위로 정규화
                angle_change = np.where(angle_change > 180, angle_change - 360, angle_change)
                angle_change = np.where(angle_change < -180, angle_change + 360, angle_change)

                results[track_id] = {
                    'velocity': velocity,
                    'speed': speed,
                    'roll_change': angle_change[0],
                    'pitch_change': angle_change[1],
                    'yaw_change': angle_change[2]
                }

            # 상태 업데이트
            self.tracks[track_id] = {
                'position': pos,
                'orientation': orientation
            }

        return results
```

---

## 6. Jetson 환경 최적화

### 6.1 Jetson Orin Nano Super 사양 검토

| 항목 | 사양 | 영향 |
|------|------|------|
| **AI 성능** | 67 TOPS (INT8) | 중형 모델까지 실시간 가능 |
| **GPU** | 1024 CUDA cores (Ampere) | TensorRT 최적화 필수 |
| **메모리** | 8GB LPDDR5 | 대형 모델 불가 |
| **TDP** | 25W (MAXN) | 열 관리 고려 |

### 6.2 모델별 Jetson 적합성 분석

#### 6.2.1 속도 측정 모델

| 모델 | 모델 크기 | 예상 FPS | 메모리 | 적합성 |
|------|----------|----------|--------|--------|
| **RAFT-Small** | ~1M params | 15-25 | ~500MB | ★★★★☆ |
| **PWC-Net** | ~4M params | 20-35 | ~600MB | ★★★★☆ |
| **FlowNet3D** | ~15MB | 10-15 | ~1GB | ★★★☆☆ |
| **RMS-FlowNet++** | 중형 | 8-12 | ~1.5GB | ★★★☆☆ |

#### 6.2.2 각도 측정 모델

| 모델 | 모델 크기 | 예상 FPS | 메모리 | 적합성 |
|------|----------|----------|--------|--------|
| **EfficientPose** | 경량 | 15-25 | ~600MB | ★★★★☆ |
| **DenseFusion** | 중형 | 10-15 | ~1GB | ★★★☆☆ |
| **FFB6D** | 대형 | 5-10 | ~2GB | ★★☆☆☆ |
| **PointNet++** | 경량 | 20-30 | ~500MB | ★★★★☆ |

### 6.3 TensorRT 최적화

```python
from ultralytics import YOLO
import tensorrt as trt

def convert_to_tensorrt(onnx_path, engine_path, fp16=True):
    """
    ONNX 모델을 TensorRT 엔진으로 변환

    Args:
        onnx_path: ONNX 모델 경로
        engine_path: 출력 TensorRT 엔진 경로
        fp16: FP16 모드 활성화 (Jetson 권장)
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # ONNX 모델 파싱
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 빌더 설정
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 엔진 빌드
    engine = builder.build_serialized_network(network, config)

    with open(engine_path, 'wb') as f:
        f.write(engine)

    return engine_path
```

### 6.4 추천 최적화 파이프라인

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Jetson 최적화 파이프라인                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [RGB + Depth 입력]                                                  │
│        │                                                             │
│        ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │           YOLO11s (TensorRT FP16)                            │    │
│  │           객체 탐지 - 55-60 FPS                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│        │                                                             │
│        ├─────────────────┬────────────────────┐                     │
│        ▼                 ▼                    ▼                      │
│  ┌───────────┐    ┌───────────────┐   ┌───────────────────┐        │
│  │ 옵션 A:   │    │  옵션 B:       │   │  옵션 C:           │        │
│  │ 전통 방식 │    │  하이브리드    │   │  Full AI          │        │
│  │           │    │               │   │                   │        │
│  │ PCA/OBB   │    │ RAFT-Small    │   │ EfficientPose     │        │
│  │ + CA KF   │    │ + KalmanNet   │   │ + FlowNet3D       │        │
│  │ (5ms)     │    │ (30ms)        │   │ (80ms)            │        │
│  └───────────┘    └───────────────┘   └───────────────────┘        │
│        │                 │                    │                      │
│        ▼                 ▼                    ▼                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    결과 출력                                  │    │
│  │           속도: m/s, 각도: degrees                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. 기술 비교 및 권장 사항

### 7.1 속도 측정 기술 비교

| 기술 | 정확도 | 속도 (Jetson) | 메모리 | 구현 난이도 | 권장 점수 |
|------|--------|--------------|--------|------------|----------|
| **CA Kalman Filter** | ±5-10% | <1ms | 최소 | 쉬움 | ★★★★☆ |
| **RAFT-Small** | ±3-7% | 15-25 FPS | ~500MB | 중간 | ★★★★☆ |
| **PWC-Net** | ±3-7% | 20-35 FPS | ~600MB | 중간 | ★★★★☆ |
| **FlowNet3D** | ±5-8% | 10-15 FPS | ~1GB | 높음 | ★★★☆☆ |
| **KalmanNet** | ±3-6% | 빠름 | ~300MB | 중간 | ★★★★★ |

### 7.2 각도 측정 기술 비교

| 기술 | 정확도 | 속도 (Jetson) | 메모리 | 구현 난이도 | 권장 점수 |
|------|--------|--------------|--------|------------|----------|
| **PCA/OBB** | ±3-5° | ~5ms | 최소 | 쉬움 | ★★★★☆ |
| **EfficientPose** | ±1-3° | 15-25 FPS | ~600MB | 중간 | ★★★★★ |
| **DenseFusion** | ±1-2° | 10-15 FPS | ~1GB | 높음 | ★★★★☆ |
| **FFB6D** | <±1° | 5-10 FPS | ~2GB | 높음 | ★★★☆☆ |
| **PointNet++** | ±2-4° | 20-30 FPS | ~500MB | 중간 | ★★★★☆ |

### 7.3 시나리오별 권장 조합

#### 시나리오 A: 실시간 우선 (30+ FPS)

```
속도 측정: CA Kalman Filter (기존 방식)
각도 측정: PCA/OBB (기존 방식)
추적: ByteTrack

장점: 가장 빠름, 구현 간단
단점: AI 기반 대비 정확도 낮음
```

#### 시나리오 B: 균형 (15-25 FPS)

```
속도 측정: RAFT-Small + KalmanNet
각도 측정: EfficientPose 또는 PointNet++
추적: ByteTrack

장점: AI 활용, 적절한 속도/정확도 균형
단점: 구현 복잡도 증가
```

#### 시나리오 C: 정확도 우선 (10-15 FPS)

```
속도 측정: FlowNet3D + KalmanNet
각도 측정: DenseFusion
추적: DeepSORT

장점: 최고 정확도
단점: 느린 처리 속도, 높은 메모리 사용
```

### 7.4 최종 권장 사항

**현재 환경 (Jetson Orin Nano Super + D455)에서의 권장 조합:**

| 항목 | 1순위 권장 | 2순위 대안 |
|------|----------|----------|
| **속도 측정** | KalmanNet (GRU-Kalman) | RAFT-Small + CA KF |
| **각도 측정** | EfficientPose | PointNet++ 커스텀 |
| **객체 추적** | ByteTrack | DeepSORT |
| **객체 탐지** | YOLO11s (TensorRT) | YOLOv8s (TensorRT) |

**권장 이유:**

1. **KalmanNet**:
   - 기존 Kalman Filter 구조 유지하면서 딥러닝 장점 활용
   - 비선형 움직임에 강건
   - 경량으로 Jetson에 적합

2. **EfficientPose**:
   - End-to-end 구조로 후처리 불필요
   - 97% 이상 정확도
   - 다중 객체 처리 효율적

3. **ByteTrack**:
   - 171 FPS 처리 속도
   - 낮은 ID Switch
   - Jetson에서 검증된 성능

---

## 8. 구현 전략

### 8.1 단계별 구현 계획

#### Phase 1: 기존 시스템 유지 + AI 검증 (1-2주)

1. 기존 Kalman Filter + PCA/OBB 시스템 유지
2. RAFT-Small, EfficientPose 모델 다운로드 및 테스트
3. 오프라인에서 정확도 비교 평가

#### Phase 2: 하이브리드 통합 (2-3주)

1. KalmanNet 구현 및 학습
2. 기존 파이프라인에 선택적 AI 모듈 추가
3. 성능 벤치마크 (속도, 정확도, 메모리)

#### Phase 3: 최적화 및 배포 (1-2주)

1. TensorRT 변환 및 최적화
2. 전체 파이프라인 통합 테스트
3. 성능 튜닝 및 안정화

### 8.2 데이터셋 요구사항

AI 모델 학습/미세조정에 필요한 데이터:

| 데이터 유형 | 수량 | 용도 |
|------------|------|------|
| **RGB-D 이미지 쌍** | 10,000+ | Optical Flow, Pose Estimation |
| **Ground Truth Flow** | 5,000+ | Scene Flow 학습 |
| **6DoF 자세 레이블** | 5,000+ | Pose Estimation 학습 |
| **추적 시퀀스** | 100+ | KalmanNet 학습 |

### 8.3 평가 메트릭

| 메트릭 | 속도 측정 | 각도 측정 |
|--------|----------|----------|
| **주 메트릭** | MAE (m/s), RMSE | MAE (°), RMSE |
| **상대 오차** | % 오차 | - |
| **처리 속도** | FPS, ms/frame | FPS, ms/frame |
| **메모리** | MB/GB | MB/GB |

---

## 9. 결론

### 9.1 핵심 요약

본 연구에서는 도장물의 속도/각도 측정을 위한 다양한 AI/딥러닝 기술을 조사했습니다.

**속도 측정:**
- **Optical Flow 기반**: RAFT, PWC-Net 등이 높은 정확도 제공
- **Scene Flow 기반**: FlowNet3D, RMS-FlowNet++가 3D 직접 추정 가능
- **하이브리드**: KalmanNet이 전통적 KF + 딥러닝 장점 결합

**각도 측정:**
- **6DoF Pose Estimation**: EfficientPose, DenseFusion, FFB6D가 고정밀 방향 추정
- **Point Cloud 기반**: PointNet++로 방향 직접 회귀 가능

**추적:**
- ByteTrack, DeepSORT가 Jetson에서 효율적인 추적 제공
- Transformer 기반 추적기는 높은 정확도나 연산량 증가

### 9.2 권장 접근 방식

**현재 권장 (즉시 적용 가능):**
- 기존 CA Kalman Filter + PCA/OBB 유지 (빠른 처리)
- ByteTrack으로 추적 강화

**향후 업그레이드 (검증 후 적용):**
- KalmanNet으로 Kalman Filter 대체 (속도 측정 개선)
- EfficientPose로 PCA/OBB 대체 (각도 측정 개선)

### 9.3 예상 성능 향상

| 항목 | 기존 방식 | AI 적용 후 (예상) |
|------|----------|-----------------|
| **속도 정확도** | ±5-10% | ±3-6% |
| **각도 정확도** | ±3-5° | ±1-3° |
| **처리 속도** | ~30 FPS | ~15-25 FPS |
| **강건성** | 중간 | 높음 |

---

## 10. 참고 자료

### 10.1 Optical Flow / Scene Flow

- [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039)
- [RAFT GitHub](https://github.com/princeton-vl/RAFT)
- [Ef-RAFT: Rethinking RAFT for Efficient Optical Flow (2024)](https://github.com/n3slami/Ef-RAFT)
- [PWC-Net: CNNs for Optical Flow](https://arxiv.org/abs/1709.02371)
- [PWC-Net GitHub](https://github.com/NVlabs/PWC-Net)
- [FlowNet3D: Learning Scene Flow in 3D Point Clouds](https://arxiv.org/abs/1806.01411)
- [FlowNet3D GitHub](https://github.com/xingyul/flownet3d)
- [RMS-FlowNet++ (IJCV 2024)](https://link.springer.com/article/10.1007/s11263-024-02093-9)
- [HALFlow GitHub](https://github.com/IRMVLab/HALFlow)
- [Deep-learning Optical Flow for Velocity Fields](https://pubs.rsc.org/en/content/articlehtml/2024/sm/d4sm00483c)
- [Optical Flow Review (viso.ai)](https://viso.ai/deep-learning/optical-flow/)

### 10.2 6DoF Pose Estimation

- [A Survey of 6DoF Object Pose Estimation Methods (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10893425/)
- [DenseFusion: 6D Object Pose Estimation](https://arxiv.org/abs/1901.04780)
- [DenseFusion GitHub](https://github.com/j96w/DenseFusion)
- [FFB6D: Full Flow Bidirectional Fusion](https://arxiv.org/abs/2103.02242)
- [PoseCNN GitHub](https://github.com/yuxng/PoseCNN)
- [EfficientPose/FFB6D 정확도 연구 (2024)](https://www.sciencedirect.com/science/article/pii/S2590123024017110)
- [RDPN6D (CVPR 2024)](https://arxiv.org/html/2405.08483v1)
- [Deep Learning-Based Object Pose Estimation Survey](https://arxiv.org/html/2405.07801v1)

### 10.3 Point Cloud Processing

- [PointNet GitHub](https://github.com/charlesq34/pointnet)
- [PointNet++ GitHub](https://github.com/charlesq34/pointnet2)
- [PointNet++ Stanford](https://stanford.edu/~rqi/pointnet2/)

### 10.4 딥러닝 강화 Kalman Filter

- [KalmanNet: Neural Network Aided Kalman Filtering](https://arxiv.org/pdf/2107.10043)
- [KalmanFormer (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11747084/)
- [Deep-Learning-Based-State-Estimation GitHub](https://github.com/zshicode/Deep-Learning-Based-State-Estimation)
- [GRU-Kalman Filter 연구](https://pmc.ncbi.nlm.nih.gov/articles/PMC10610770/)

### 10.5 Object Tracking

- [ByteTrack GitHub](https://github.com/FoundationVision/ByteTrack)
- [ByteTrack vs FairMOT](https://www.tredence.com/blog/advancements-in-object-tracking-bytetrack-vs-fairmot)
- [DeepSORT Guide](https://www.ikomia.ai/blog/deep-sort-object-tracking-guide)
- [3D Object Tracking with Transformer](https://arxiv.org/abs/2110.14921)
- [Transformer Tracking Repository](https://github.com/Little-Podi/Transformer_Tracking)

### 10.6 Edge/Jetson Deployment

- [Jetson-Inference GitHub](https://github.com/dusty-nv/jetson-inference)
- [NVIDIA Jetson Benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks)
- [3D Object Detection on Jetson Benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC10144830/)
- [Jetson Computer Vision (viso.ai)](https://viso.ai/edge-ai/nvidia-jetson/)

---

*작성: 2025-12-18*
*환경: Intel RealSense D455 + NVIDIA Jetson Orin Nano Super*
*목적: AI 기반 도장물 속도/각도 측정 기술 연구*
