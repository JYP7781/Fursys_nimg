# nimg_v3 FoundationPose 기반 종합 설계 및 구현 가이드

**작성일**: 2025-12-31
**버전**: v3.0
**목적**: nimg 시스템 진화 히스토리 분석 및 FoundationPose 기반 nimg_v3 구현 가이드
**환경**: Intel RealSense D455 + NVIDIA Jetson Orin Nano Super

---

## 목차

1. [프로젝트 히스토리 및 진화 과정](#1-프로젝트-히스토리-및-진화-과정)
2. [기존 버전 분석 요약](#2-기존-버전-분석-요약)
3. [nimg_v3 설계 목표 및 방향](#3-nimg_v3-설계-목표-및-방향)
4. [FoundationPose 핵심 기술 분석](#4-foundationpose-핵심-기술-분석)
5. [시스템 아키텍처 설계](#5-시스템-아키텍처-설계)
6. [모듈별 상세 설계](#6-모듈별-상세-설계)
7. [구현 로드맵](#7-구현-로드맵)
8. [기술적 고려사항](#8-기술적-고려사항)
9. [테스트 및 검증 계획](#9-테스트-및-검증-계획)
10. [참고 자료](#10-참고-자료)

---

## 1. 프로젝트 히스토리 및 진화 과정

### 1.1 버전 진화 타임라인

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        nimg 시스템 버전 진화 히스토리                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  [nimg (초기 버전)]                                                                   │
│  └── 기간: ~2025-12                                                                  │
│  └── 특징: ROS2 기반 YOLOv5 탐지, Hough 변환 각도 측정                                │
│  └── 문제점: 속도 미구현, 각도 오차 ±10-20°                                          │
│                                                                                      │
│         │                                                                            │
│         ▼                                                                            │
│                                                                                      │
│  [nimg_v2]                                                                           │
│  └── 기간: 2025-12-05 ~                                                              │
│  └── 기반 문서: implementation_design_guide.md                                       │
│  └── 특징:                                                                           │
│      • 모듈화된 아키텍처 (data/detection/tracking/estimation/analysis/output)        │
│      • CA Kalman Filter 기반 속도 측정                                               │
│      • PCA/OBB 기반 3D 방향 추정                                                     │
│      • 데이터 동기화 (RGB + Depth + IMU)                                             │
│  └── 한계점:                                                                         │
│      • PCA/OBB 정확도 한계 (±3-5°)                                                  │
│      • Domain Shift 취약성                                                           │
│      • 다중 모듈 관리 복잡성                                                         │
│                                                                                      │
│         │                                                                            │
│         ├── ai_based_velocity_angle_measurement_research.md (2025-12-18)             │
│         │   └── AI 기반 접근법 조사 (RAFT, KalmanNet, EfficientPose 등)              │
│         │   └── 권장: KalmanNet + EfficientPose + ByteTrack                         │
│         │                                                                            │
│         ├── yolo_domain_shift_analysis_report.md (2025-12-29)                        │
│         │   └── YOLO Domain Shift 문제 분석                                          │
│         │   └── 해결방안: 데이터 증강, Fine-tuning, Domain Adaptation                │
│         │                                                                            │
│         ▼                                                                            │
│                                                                                      │
│  [nimg_v3] ← 현재 구현 목표                                                          │
│  └── 기간: 2025-12-31 ~                                                              │
│  └── 기반 문서: foundationpose_implementation_guide.md                               │
│  └── 핵심 변경:                                                                      │
│      • FoundationPose 기반 통합 6DoF 자세 추정                                       │
│      • Model-Free 설정 지원 (CAD 없이 참조 이미지로 동작)                            │
│      • Zero-Shot 일반화 (Domain Shift 해결)                                          │
│      • 파이프라인 단순화 (통합 프레임워크)                                           │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 핵심 문서 관계도

```
research/
├── 251008/
│   └── depth_3d_velocity_angle_measurement.md       # 초기 연구
│
├── 251204/
│   ├── nimg_code_analysis_and_ai_recommendations.md  # 코드 분석
│   ├── ai_model_selection_guide.md                   # AI 모델 선택 가이드
│   └── advanced_improvements_d455_jetson_orin_nano_super.md
│
├── 251205/
│   ├── comprehensive_improvement_research_2025.md    # 종합 개선 연구
│   ├── implementation_design_guide.md  ─────────────┐# nimg_v2 기반 문서
│   └── orb_application_analysis_2025.md              │
│                                                     │
├── 251218/                                           │
│   ├── ai_based_velocity_angle_measurement_research.md ← nimg_v2 개선 연구
│   └── depth_anything_v3_analysis_research.md        │
│                                                     │
├── 251224/                                           │
│   └── FoundationPose_Korean_Translation.md ────────┐# 핵심 논문 번역
│                                                    │
├── 251229/                                          │
│   └── yolo_domain_shift_analysis_report.md         │# Domain Shift 분석
│                                                    │
└── 251231/                                          │
    ├── foundationpose_implementation_guide.md ──────┴# nimg_v3 기반 문서
    ├── yolo_roboflow_training_complete_guide.md
    ├── training_data_preparation_guide.md
    └── nimg_v3_foundationpose_comprehensive_design_guide.md  ← 현재 문서
```

### 1.3 접근 방식 변화 요약

| 버전 | 속도 측정 | 각도 측정 | 추적 | Domain 적응 |
|------|----------|----------|------|------------|
| **nimg** | 미구현 (speed=0) | Hough 변환 (Yaw만) | 없음 | 없음 |
| **nimg_v2** | CA Kalman Filter | PCA/OBB | 없음 | 없음 |
| **nimg_v2 (AI 개선 계획)** | KalmanNet + RAFT | EfficientPose | ByteTrack | 없음 |
| **nimg_v3 (현재)** | FoundationPose 추적 + KF | FoundationPose 6DoF | 내장 추적 | Zero-Shot |

---

## 2. 기존 버전 분석 요약

### 2.1 nimg (초기 버전) 분석

**디렉토리**: `src/nimg/`

#### 주요 파일 구조
```
src/nimg/
├── nimg/
│   ├── nimg.py                    # 메인 ROS2 노드 (nimg_x86)
│   ├── submodules/
│   │   ├── nodeL515.py            # dSensor - L515 카메라 처리
│   │   ├── detectProcessor.py     # 객체 탐지 및 각도 측정
│   │   ├── detect.py              # YOLOv5 Detector
│   │   ├── lineDetector.py        # Hough 변환 라인 검출
│   │   ├── orbProcessor.py        # ORB 특징점
│   │   ├── ItemList.py            # Item, ItemList 클래스
│   │   └── ...
│   ├── models/                    # YOLOv5 모델 정의
│   └── utils/                     # 유틸리티 함수
└── models/
    └── class187_image85286_v12x_250epochs.pt
```

#### 문제점 요약
| 문제 영역 | 현재 상태 | 영향 |
|----------|----------|------|
| **속도 측정** | 미구현 (`speed = 0`) | 속도 정보 없음 |
| **Pitch 각도** | 단순 depth 차이 | ±15-20° 오차 |
| **Yaw 각도** | 2D Hough 변환 | ±10-15° 오차 |
| **Roll 각도** | 미구현 | Roll 정보 없음 |
| **Point Cloud** | 파일 저장용만 | 실시간 분석 불가 |

### 2.2 nimg_v2 분석

**디렉토리**: `src/nimg_v2/`

#### 주요 파일 구조
```
src/nimg_v2/
├── nimg_v2/
│   ├── __init__.py
│   ├── main.py                        # 메인 파이프라인
│   ├── config/
│   │   ├── camera_intrinsics.yaml     # D455 카메라 파라미터
│   │   └── processing_config.yaml     # 처리 설정
│   ├── data/
│   │   ├── data_loader.py             # RGB/Depth/IMU 로더
│   │   └── frame_synchronizer.py      # 프레임-IMU 동기화
│   ├── detection/
│   │   ├── yolo_detector.py           # YOLO 기반 탐지
│   │   └── item.py                    # Item/ItemList 확장
│   ├── tracking/
│   │   ├── kalman_filter_3d.py        # 3D CA Kalman Filter
│   │   └── object_tracker.py          # 객체 추적기
│   ├── estimation/
│   │   ├── position_estimator.py      # 3D 위치 추정
│   │   ├── orientation_estimator.py   # PCA/OBB 방향 추정
│   │   └── point_cloud_processor.py   # Point Cloud 처리
│   ├── analysis/
│   │   ├── change_calculator.py       # 변화량 계산
│   │   └── reference_manager.py       # 기준 프레임 관리
│   └── output/
│       ├── result_exporter.py         # 결과 내보내기
│       └── visualizer.py              # 시각화
└── tests/
    └── ...
```

#### 개선점 vs 한계
| 영역 | nimg → nimg_v2 개선 | 남은 한계 |
|------|---------------------|----------|
| **아키텍처** | 모듈화된 구조 | 다중 모듈 복잡성 |
| **속도 측정** | CA Kalman Filter 구현 | ±5-10% 오차 |
| **각도 측정** | PCA/OBB 3D 방향 | ±3-5° 오차 |
| **데이터** | RGB+Depth+IMU 동기화 | - |
| **Domain** | 없음 | Domain Shift 취약 |

### 2.3 AI 기반 개선 연구 결과 (ai_based_velocity_angle_measurement_research.md)

#### 조사된 기술
| 측정 영역 | 기술 | 예상 성능 |
|----------|------|----------|
| **속도 측정** | KalmanNet, RAFT-Small, FlowNet3D | ±3-6% |
| **각도 측정** | EfficientPose, DenseFusion, FFB6D | ±1-3° |
| **추적** | ByteTrack, DeepSORT | ID Switch 감소 |

#### 권장 조합
```
속도: KalmanNet (GRU-Kalman)
각도: EfficientPose
추적: ByteTrack
탐지: YOLO11s (TensorRT)
```

#### 한계점
- **다중 모델 관리**: 4개 이상의 별도 모델 필요
- **Domain Shift**: 여전히 해결 안됨
- **파이프라인 복잡성**: 오류 전파 가능성

---

## 3. nimg_v3 설계 목표 및 방향

### 3.1 핵심 설계 목표

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       nimg_v3 핵심 설계 목표                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. 통합 프레임워크                                                       │
│     • 자세 추정 + 추적 + 각도 측정을 단일 모델로                         │
│     • FoundationPose 기반 End-to-End 6DoF 추정                          │
│                                                                          │
│  2. Zero-Shot 일반화                                                      │
│     • 새로운 환경 (조명, 각도 변화)에 미세조정 없이 적용                  │
│     • Domain Shift 문제 근본적 해결                                      │
│                                                                          │
│  3. Model-Free 지원                                                       │
│     • CAD 모델 없이 ~16개 참조 이미지로 동작                             │
│     • 새로운 객체에 빠른 적용 가능                                        │
│                                                                          │
│  4. 파이프라인 단순화                                                     │
│     • 기존: YOLO → ByteTrack → EfficientPose → KalmanNet                │
│     • 신규: YOLO → FoundationPose → Kalman (후처리)                     │
│                                                                          │
│  5. 실시간 성능                                                           │
│     • 추적 모드: 30+ FPS                                                 │
│     • Jetson Orin Nano Super 최적화                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 nimg_v2 vs nimg_v3 비교

| 항목 | nimg_v2 | nimg_v3 | 개선 이유 |
|------|---------|---------|----------|
| **자세 추정** | PCA/OBB | FoundationPose | 정확도 향상, 통합 |
| **속도 측정** | CA Kalman 단독 | FP 추적 + Kalman | 더 정밀한 위치 |
| **추적** | 없음 | FoundationPose 내장 | 프레임간 일관성 |
| **Domain 적응** | 없음 | Zero-Shot | 새 환경 즉시 적용 |
| **객체 모델링** | 없음 | Neural Object Field | CAD 없이 동작 |
| **모듈 수** | 6개 | 3개 | 단순화, 유지보수 |
| **예상 각도 오차** | ±3-5° | ±1-2° | 2배 이상 개선 |
| **예상 속도 오차** | ±5-10% | ±2-5% | 개선 |

### 3.3 왜 FoundationPose인가?

#### FoundationPose 핵심 장점

1. **통합 프레임워크**
   ```
   기존: [탐지] → [추적] → [자세추정] → [속도계산]
   FP:   [탐지] → [FoundationPose] → [후처리]
   ```

2. **Zero-Shot 일반화**
   - YCB-Video ADD-S: 97.4% (기존 SOTA 88.4%)
   - LINEMOD ADD-0.1d: 99.9% (기존 91.5%)
   - 새로운 객체/환경에 미세조정 없이 적용

3. **Model-Free 지원**
   - CAD 모델 없이 ~16개 참조 이미지로 동작
   - Neural Object Field로 효율적 객체 표현
   - 빠른 프로토타입 및 배포

4. **실시간 추적**
   - 자세 추정: ~1 FPS (초기화)
   - 추적 모드: ~32 Hz
   - 인스턴스 수준 방법과 비교할 만한 성능

---

## 4. FoundationPose 핵심 기술 분석

### 4.1 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FoundationPose 아키텍처                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   [입력]                                                                         │
│   ┌────────────────┐    ┌────────────────┐                                      │
│   │   RGB 이미지    │    │  Depth 이미지   │  ← RGBD 입력                        │
│   └───────┬────────┘    └───────┬────────┘                                      │
│           │                     │                                                │
│           └──────────┬──────────┘                                                │
│                      ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │              객체 정보 (Model-Based OR Model-Free)                       │   │
│   │  ┌─────────────────────┐     ┌─────────────────────────────┐            │   │
│   │  │ Model-Based:        │ OR  │ Model-Free:                 │            │   │
│   │  │ CAD 모델 (텍스처)   │     │ ~16개 참조 이미지           │            │   │
│   │  │                     │     │ + Neural Object Field       │            │   │
│   │  └─────────────────────┘     └─────────────────────────────┘            │   │
│   └────────────────────────────────────┬────────────────────────────────────┘   │
│                                        │                                         │
│                                        ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     1. 자세 가설 생성 (Pose Hypothesis)                  │   │
│   │  • 정이십면체 샘플링 → N_s개 시점                                       │   │
│   │  • 평면 내 회전 증강 → N_i개                                            │   │
│   │  • 총 N_s × N_i 개의 초기 자세 가설                                     │   │
│   └────────────────────────────────────┬────────────────────────────────────┘   │
│                                        │                                         │
│                                        ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     2. 자세 정제 네트워크 (Pose Refiner)                 │   │
│   │  • Transformer 기반 아키텍처                                            │   │
│   │  • 렌더링 vs 관측 비교                                                  │   │
│   │  • 반복적 업데이트: Δt (이동), ΔR (회전)                               │   │
│   │                                                                          │   │
│   │  입력: [렌더링 RGBD | 관측 RGBD] → CNN Encoder → Transformer            │   │
│   │  출력: Δt ∈ ℝ³, ΔR ∈ SO(3)                                             │   │
│   └────────────────────────────────────┬────────────────────────────────────┘   │
│                                        │                                         │
│                                        ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     3. 자세 선택 (Pose Selector)                         │   │
│   │  • 계층적 비교 전략 (2단계)                                             │   │
│   │  • 대조 학습 기반 점수 계산                                             │   │
│   │  • 최고 점수 자세 선택                                                   │   │
│   └────────────────────────────────────┬────────────────────────────────────┘   │
│                                        │                                         │
│                                        ▼                                         │
│   [출력]                                                                         │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │  6DoF 자세: [R | t] ∈ SE(3)                                             │    │
│   │  • 이동 (Translation): t = [tx, ty, tz]                                 │    │
│   │  • 회전 (Rotation): R (3×3 행렬)                                        │    │
│   │  → Roll, Pitch, Yaw 직접 추출 가능                                      │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Model-Based vs Model-Free 설정

| 항목 | Model-Based | Model-Free |
|------|-------------|------------|
| **입력** | RGBD + CAD 모델 | RGBD + ~16개 참조 이미지 |
| **객체 표현** | 텍스처 메시 | Neural Object Field |
| **정확도** | 더 높음 | 약간 낮음 |
| **렌더링 속도** | 빠름 | 빠름 (메시 추출 후) |
| **적용 난이도** | CAD 필요 | 사진 촬영만 |
| **적합한 경우** | 제조업, 표준 객체 | 다양한 객체, 프로토타입 |

### 4.3 Neural Object Field (Model-Free 핵심)

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

학습:
- ~16개 참조 이미지로 몇 초 내에 학습
- 한 번 학습 후 재사용 가능

장점:
- NeRF 대비 더 정확한 깊이 렌더링
- 효율적인 새로운 뷰 합성
"""
```

### 4.4 성능 벤치마크

| 데이터셋 | 메트릭 | 기존 SOTA | FoundationPose | 개선율 |
|----------|--------|----------|----------------|--------|
| **YCB-Video** | ADD-S AUC | 88.4% (FS6D) | **97.4%** | +9.0% |
| **LINEMOD** | ADD-0.1d | 91.5% (FS6D+ICP) | **99.9%** | +8.4% |
| **YCBInEOAT** | ADD AUC | 92.66% (TrackNet) | **93.09%** | +0.43% |
| **BOP (LM-O)** | AR | 58.3% (MegaPose) | **78.8%** | +20.5% |
| **BOP (T-LESS)** | AR | 54.3% (MegaPose) | **83.0%** | +28.7% |
| **BOP (YCB-V)** | AR | 63.3% (MegaPose) | **88.0%** | +24.7% |

---

## 5. 시스템 아키텍처 설계

### 5.1 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         nimg_v3 시스템 아키텍처                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                           입력 계층 (Input Layer)                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │ RealSense   │  │   RGB       │  │   Depth     │  │ Camera          │   │  │
│  │  │ D455        │──│   Image     │──│   Image     │──│ Intrinsics      │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                          │
│                                        ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                         탐지 계층 (Detection Layer)                        │  │
│  │  ┌───────────────────────────────────────────────────────────────────────┐│  │
│  │  │                    YOLO11s / YOLOv12x (TensorRT)                      ││  │
│  │  │  • 객체 탐지 및 분류                                                  ││  │
│  │  │  • 바운딩 박스 → 마스크 생성                                          ││  │
│  │  │  • 예상 FPS: 55-60 (Jetson)                                          ││  │
│  │  └───────────────────────────────────────────────────────────────────────┘│  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                          │
│                                        ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                       자세 추정 계층 (Pose Estimation Layer)               │  │
│  │  ┌───────────────────────────────────────────────────────────────────────┐│  │
│  │  │                         FoundationPose                                ││  │
│  │  │  ┌─────────────────────────────────────────────────────────────────┐ ││  │
│  │  │  │ 모드 선택:                                                       │ ││  │
│  │  │  │  • Model-Based: CAD 모델 + 렌더링                               │ ││  │
│  │  │  │  • Model-Free: Neural Object Field + 참조 이미지                │ ││  │
│  │  │  └─────────────────────────────────────────────────────────────────┘ ││  │
│  │  │                                                                       ││  │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  ││  │
│  │  │  │ 자세 초기화     │→│ 자세 정제       │→│ 자세 선택           │  ││  │
│  │  │  │ (Pose Init)    │  │ (Pose Refine)   │  │ (Pose Selection)    │  ││  │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  ││  │
│  │  │                                                                       ││  │
│  │  │  출력: 6DoF 자세 [R | t], Confidence                                  ││  │
│  │  └───────────────────────────────────────────────────────────────────────┘│  │
│  │                                                                            │  │
│  │  ┌───────────────────────────────────────────────────────────────────────┐│  │
│  │  │                         추적 모드 (Tracking)                          ││  │
│  │  │  • 이전 자세 기반 정제만 수행 (초기화 생략)                           ││  │
│  │  │  • 예상 FPS: 30-40 (Jetson)                                          ││  │
│  │  └───────────────────────────────────────────────────────────────────────┘│  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                          │
│                                        ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                       후처리 계층 (Post-Processing Layer)                  │  │
│  │  ┌───────────────────────────────────────────────────────────────────────┐│  │
│  │  │                      6DoF → 속도/각도 변환                             ││  │
│  │  │                                                                       ││  │
│  │  │  ┌─────────────────────┐  ┌─────────────────────────────────────────┐ ││  │
│  │  │  │ 각도 추출           │  │ 속도 추정                               │ ││  │
│  │  │  │ pose_to_euler()     │  │ PoseKalmanFilter (12-state)             │ ││  │
│  │  │  │ → Roll, Pitch, Yaw │  │ → vx, vy, vz, ωx, ωy, ωz               │ ││  │
│  │  │  └─────────────────────┘  └─────────────────────────────────────────┘ ││  │
│  │  │                                                                       ││  │
│  │  │  ┌─────────────────────────────────────────────────────────────────┐ ││  │
│  │  │  │ 기준 프레임 대비 변화량 계산                                     │ ││  │
│  │  │  │ ChangeCalculator (기존 nimg_v2 모듈 재사용)                     │ ││  │
│  │  │  └─────────────────────────────────────────────────────────────────┘ ││  │
│  │  └───────────────────────────────────────────────────────────────────────┘│  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                          │
│                                        ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                         출력 계층 (Output Layer)                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │ CSV/JSON    │  │ Visualizer  │  │ ROS2 Topic  │  │ Database        │   │  │
│  │  │ Export      │  │             │  │ Publisher   │  │ (Optional)      │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 새 디렉토리 구조

```
src/nimg_v3/
├── nimg_v3/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── camera_intrinsics.yaml       # D455 카메라 파라미터
│   │   ├── foundationpose_config.yaml   # FoundationPose 설정
│   │   └── system_config.yaml           # 전체 시스템 설정
│   │
│   ├── input/
│   │   ├── __init__.py
│   │   ├── realsense_camera.py          # RealSense D455 인터페이스
│   │   ├── data_loader.py               # 오프라인 데이터 로더 (nimg_v2 계승)
│   │   └── frame_buffer.py              # 프레임 버퍼 관리
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py             # YOLO 탐지기 (nimg_v2 계승)
│   │   └── mask_generator.py            # 바운딩박스 → 마스크 변환
│   │
│   ├── pose/
│   │   ├── __init__.py
│   │   ├── foundationpose_estimator.py  # FoundationPose 래퍼 (핵심)
│   │   ├── model_based_setup.py         # Model-Based 설정
│   │   ├── model_free_setup.py          # Model-Free 설정
│   │   ├── neural_object_field.py       # Neural Object Field 학습
│   │   └── reference_capture.py         # 참조 이미지 촬영 도구
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── pose_tracker.py              # 자세 추적 관리
│   │   └── tracking_recovery.py         # 추적 손실 복구
│   │
│   ├── measurement/
│   │   ├── __init__.py
│   │   ├── pose_to_euler.py             # 6DoF → 오일러 각도 변환
│   │   ├── pose_kalman_filter.py        # 12-state Kalman Filter
│   │   ├── velocity_estimator.py        # 속도 추정
│   │   └── change_calculator.py         # 기준 대비 변화량 (nimg_v2 계승)
│   │
│   ├── output/
│   │   ├── __init__.py
│   │   ├── result_exporter.py           # CSV/JSON 내보내기 (nimg_v2 계승)
│   │   ├── visualizer.py                # 시각화 (nimg_v2 계승)
│   │   ├── ros2_publisher.py            # ROS2 토픽 발행
│   │   └── measurement_logger.py        # 측정 결과 로깅
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── rotation_utils.py            # 회전 변환 유틸리티
│   │   ├── point_cloud_utils.py         # Point Cloud 유틸리티
│   │   └── performance_monitor.py       # 성능 모니터링
│   │
│   └── main.py                          # 메인 파이프라인
│
├── models/
│   ├── foundationpose/                  # FoundationPose 가중치
│   │   ├── refine_model/
│   │   └── scorer_model/
│   ├── neural_fields/                   # 학습된 Neural Object Field
│   │   └── painting_object/
│   ├── yolo/                            # YOLO 모델
│   │   └── class187_image85286_v12x_250epochs.pt
│   └── cad_models/                      # CAD 모델 (Model-Based용)
│       └── painting_object.obj
│
├── reference_images/                    # 참조 이미지 (Model-Free용)
│   └── painting_object/
│       ├── rgb/
│       ├── depth/
│       └── mask/
│
├── tests/
│   ├── __init__.py
│   ├── test_foundationpose.py
│   ├── test_pose_kalman.py
│   ├── test_pipeline.py
│   └── test_measurement.py
│
├── scripts/
│   ├── capture_reference_images.py      # 참조 이미지 촬영
│   ├── train_neural_field.py            # Neural Object Field 학습
│   ├── convert_to_tensorrt.py           # TensorRT 변환
│   └── run_benchmark.py                 # 성능 벤치마크
│
├── docker/
│   ├── Dockerfile.foundationpose
│   └── docker-compose.yml
│
├── requirements.txt
├── setup.py
└── README.md
```

### 5.3 nimg_v2에서 재사용할 모듈

| 모듈 | 파일 | 재사용 방식 |
|------|------|------------|
| **데이터 로더** | `data/data_loader.py` | 그대로 사용 (오프라인 테스트용) |
| **YOLO 탐지기** | `detection/yolo_detector.py` | 그대로 사용 |
| **변화량 계산** | `analysis/change_calculator.py` | 인터페이스 수정 후 사용 |
| **결과 내보내기** | `output/result_exporter.py` | 확장하여 사용 |
| **시각화** | `output/visualizer.py` | 확장하여 사용 |

---

## 6. 모듈별 상세 설계

### 6.1 FoundationPose 통합 모듈 (pose/foundationpose_estimator.py)

```python
"""
FoundationPose 통합 래퍼 클래스
- Model-Based와 Model-Free 모드 통합 지원
- 추정 및 추적 모드 자동 전환
- Jetson 최적화 옵션
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

class PoseMode(Enum):
    MODEL_BASED = "model_based"
    MODEL_FREE = "model_free"

class TrackingState(Enum):
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    LOST = "lost"

@dataclass
class PoseResult:
    """자세 추정 결과"""
    pose_matrix: np.ndarray      # 4x4 변환 행렬
    translation: np.ndarray      # [tx, ty, tz]
    rotation_matrix: np.ndarray  # 3x3 회전 행렬
    confidence: float            # 신뢰도 점수
    tracking_state: TrackingState
    processing_time_ms: float

class FoundationPoseEstimator:
    """FoundationPose 기반 6DoF 자세 추정기"""

    def __init__(
        self,
        model_dir: str,
        mode: PoseMode = PoseMode.MODEL_FREE,
        mesh_path: Optional[str] = None,
        neural_field_dir: Optional[str] = None,
        device: str = 'cuda:0',
        use_tensorrt: bool = True,
        tracking_recovery_threshold: float = 0.3,
        max_lost_frames: int = 5
    ):
        """
        Args:
            model_dir: FoundationPose 모델 디렉토리
            mode: Model-Based 또는 Model-Free
            mesh_path: CAD 모델 경로 (Model-Based용)
            neural_field_dir: Neural Object Field 경로 (Model-Free용)
            device: 추론 디바이스
            use_tensorrt: TensorRT 최적화 사용
            tracking_recovery_threshold: 추적 복구 신뢰도 임계값
            max_lost_frames: 최대 추적 손실 프레임
        """
        self.mode = mode
        self.device = device
        self.tracking_recovery_threshold = tracking_recovery_threshold
        self.max_lost_frames = max_lost_frames

        # FoundationPose 초기화
        self._init_foundationpose(model_dir, use_tensorrt)

        # 모드별 객체 모델 로드
        if mode == PoseMode.MODEL_BASED:
            if mesh_path is None:
                raise ValueError("mesh_path required for MODEL_BASED mode")
            self._load_mesh(mesh_path)
        else:
            if neural_field_dir is None:
                raise ValueError("neural_field_dir required for MODEL_FREE mode")
            self._load_neural_field(neural_field_dir)

        # 추적 상태
        self._tracking_state = TrackingState.INITIALIZING
        self._prev_pose = None
        self._lost_frame_count = 0

    def _init_foundationpose(self, model_dir: str, use_tensorrt: bool):
        """FoundationPose 모델 초기화"""
        # TODO: 실제 FoundationPose import 및 초기화
        pass

    def _load_mesh(self, mesh_path: str):
        """CAD 모델 로드 (Model-Based)"""
        # TODO: 메시 로드
        pass

    def _load_neural_field(self, neural_field_dir: str):
        """Neural Object Field 로드 (Model-Free)"""
        # TODO: Neural Field에서 메시 추출
        pass

    def estimate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: Dict[str, float]
    ) -> PoseResult:
        """
        자세 추정 (초기화 또는 재초기화)

        Args:
            rgb: RGB 이미지 [H, W, 3]
            depth: Depth 이미지 [H, W] (미터 단위)
            mask: 객체 마스크 [H, W] (binary)
            intrinsics: 카메라 파라미터 {'fx', 'fy', 'cx', 'cy'}

        Returns:
            PoseResult: 자세 추정 결과
        """
        import time
        start_time = time.time()

        # TODO: FoundationPose 추정 호출
        # pose, confidence = self._estimator.estimate(...)

        # 임시 더미 결과
        pose_matrix = np.eye(4)
        confidence = 0.9

        processing_time = (time.time() - start_time) * 1000

        # 추적 상태 업데이트
        self._tracking_state = TrackingState.TRACKING
        self._prev_pose = pose_matrix.copy()
        self._lost_frame_count = 0

        return PoseResult(
            pose_matrix=pose_matrix,
            translation=pose_matrix[:3, 3],
            rotation_matrix=pose_matrix[:3, :3],
            confidence=confidence,
            tracking_state=self._tracking_state,
            processing_time_ms=processing_time
        )

    def track(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: Dict[str, float]
    ) -> PoseResult:
        """
        자세 추적 (이전 자세 기반 빠른 추정)

        Args:
            rgb: RGB 이미지
            depth: Depth 이미지
            intrinsics: 카메라 파라미터

        Returns:
            PoseResult: 자세 추정 결과
        """
        if self._prev_pose is None:
            raise RuntimeError("Cannot track without previous pose. Call estimate() first.")

        import time
        start_time = time.time()

        # TODO: FoundationPose 추적 호출
        # pose, confidence = self._estimator.track(...)

        # 임시 더미 결과
        pose_matrix = self._prev_pose.copy()
        confidence = 0.85

        processing_time = (time.time() - start_time) * 1000

        # 신뢰도 체크 및 추적 상태 업데이트
        if confidence < self.tracking_recovery_threshold:
            self._lost_frame_count += 1
            if self._lost_frame_count >= self.max_lost_frames:
                self._tracking_state = TrackingState.LOST
        else:
            self._tracking_state = TrackingState.TRACKING
            self._prev_pose = pose_matrix.copy()
            self._lost_frame_count = 0

        return PoseResult(
            pose_matrix=pose_matrix,
            translation=pose_matrix[:3, 3],
            rotation_matrix=pose_matrix[:3, :3],
            confidence=confidence,
            tracking_state=self._tracking_state,
            processing_time_ms=processing_time
        )

    def process(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: Optional[np.ndarray],
        intrinsics: Dict[str, float],
        force_estimate: bool = False
    ) -> PoseResult:
        """
        자동 모드 선택 처리

        Args:
            rgb, depth, mask, intrinsics: 입력 데이터
            force_estimate: 강제로 전체 추정 수행

        Returns:
            PoseResult: 자세 추정 결과
        """
        # 추적 손실 또는 강제 추정 시
        if (self._tracking_state == TrackingState.LOST or
            self._tracking_state == TrackingState.INITIALIZING or
            force_estimate):

            if mask is None:
                raise ValueError("mask required for pose estimation")
            return self.estimate(rgb, depth, mask, intrinsics)

        # 추적 모드
        return self.track(rgb, depth, intrinsics)

    def reset(self):
        """추적 상태 리셋"""
        self._tracking_state = TrackingState.INITIALIZING
        self._prev_pose = None
        self._lost_frame_count = 0

    @property
    def is_tracking(self) -> bool:
        return self._tracking_state == TrackingState.TRACKING

    @property
    def tracking_state(self) -> TrackingState:
        return self._tracking_state
```

### 6.2 6DoF → 오일러 변환 모듈 (measurement/pose_to_euler.py)

```python
"""
6DoF 자세 행렬에서 오일러 각도 추출
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple
from dataclasses import dataclass

@dataclass
class EulerAngles:
    """오일러 각도 (degrees)"""
    roll: float   # X축 회전
    pitch: float  # Y축 회전
    yaw: float    # Z축 회전

def pose_to_euler(pose_matrix: np.ndarray) -> Tuple[np.ndarray, EulerAngles]:
    """
    4x4 자세 행렬에서 이동 벡터와 오일러 각도 추출

    Args:
        pose_matrix: 4x4 변환 행렬 [R|t]

    Returns:
        translation: [tx, ty, tz] 미터 단위
        euler_angles: EulerAngles (roll, pitch, yaw) 도 단위
    """
    # 이동 추출
    translation = pose_matrix[:3, 3].copy()

    # 회전 추출
    rotation_matrix = pose_matrix[:3, :3]

    # scipy로 오일러 각도 변환 (XYZ 순서)
    r = Rotation.from_matrix(rotation_matrix)
    euler = r.as_euler('xyz', degrees=True)

    euler_angles = EulerAngles(
        roll=euler[0],
        pitch=euler[1],
        yaw=euler[2]
    )

    return translation, euler_angles

def compute_angle_change(
    prev_pose: np.ndarray,
    curr_pose: np.ndarray
) -> np.ndarray:
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

    delta_angles = np.array([
        curr_euler.roll - prev_euler.roll,
        curr_euler.pitch - prev_euler.pitch,
        curr_euler.yaw - prev_euler.yaw
    ])

    # -180 ~ 180 범위로 정규화
    delta_angles = np.where(delta_angles > 180, delta_angles - 360, delta_angles)
    delta_angles = np.where(delta_angles < -180, delta_angles + 360, delta_angles)

    return delta_angles

def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """회전 행렬을 쿼터니언으로 변환"""
    r = Rotation.from_matrix(rotation_matrix)
    return r.as_quat()  # [x, y, z, w]

def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """쿼터니언을 회전 행렬로 변환"""
    r = Rotation.from_quat(quaternion)  # [x, y, z, w]
    return r.as_matrix()
```

### 6.3 12-상태 Kalman Filter (measurement/pose_kalman_filter.py)

```python
"""
6DoF 자세를 위한 확장 Kalman Filter
상태: [x, y, z, vx, vy, vz, roll, pitch, yaw, ωx, ωy, ωz]
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class PoseKalmanState:
    """Kalman Filter 상태"""
    position: np.ndarray       # [x, y, z] 미터
    velocity: np.ndarray       # [vx, vy, vz] m/s
    orientation: np.ndarray    # [roll, pitch, yaw] 도
    angular_velocity: np.ndarray  # [ωx, ωy, ωz] °/s
    speed: float               # 선속도 크기 m/s

class PoseKalmanFilter:
    """
    6DoF 자세 추정을 위한 12-상태 Kalman Filter

    상태 벡터: [x, y, z, vx, vy, vz, roll, pitch, yaw, ωx, ωy, ωz]
    측정 벡터: [x, y, z, roll, pitch, yaw]
    """

    def __init__(
        self,
        dt: float = 1/30.0,
        process_noise_pos: float = 0.01,
        process_noise_vel: float = 0.1,
        process_noise_angle: float = 0.1,
        process_noise_angular_vel: float = 1.0,
        measurement_noise_pos: float = 0.005,
        measurement_noise_angle: float = 0.5
    ):
        """
        Args:
            dt: 프레임 간 시간 간격 (초)
            process_noise_*: 프로세스 노이즈 파라미터
            measurement_noise_*: 측정 노이즈 파라미터
        """
        self.dt = dt

        # 12차원 상태, 6차원 측정
        self.kf = KalmanFilter(dim_x=12, dim_z=6)

        # 상태 전이 행렬 F (등속 모델)
        self.kf.F = np.eye(12)
        # 위치 += 속도 * dt
        self.kf.F[0, 3] = dt  # x += vx * dt
        self.kf.F[1, 4] = dt  # y += vy * dt
        self.kf.F[2, 5] = dt  # z += vz * dt
        # 각도 += 각속도 * dt
        self.kf.F[6, 9] = dt   # roll += ωx * dt
        self.kf.F[7, 10] = dt  # pitch += ωy * dt
        self.kf.F[8, 11] = dt  # yaw += ωz * dt

        # 측정 행렬 H
        self.kf.H = np.zeros((6, 12))
        self.kf.H[0, 0] = 1  # x
        self.kf.H[1, 1] = 1  # y
        self.kf.H[2, 2] = 1  # z
        self.kf.H[3, 6] = 1  # roll
        self.kf.H[4, 7] = 1  # pitch
        self.kf.H[5, 8] = 1  # yaw

        # 프로세스 노이즈 Q
        self.kf.Q = np.diag([
            process_noise_pos, process_noise_pos, process_noise_pos,       # 위치
            process_noise_vel, process_noise_vel, process_noise_vel,       # 속도
            process_noise_angle, process_noise_angle, process_noise_angle, # 각도
            process_noise_angular_vel, process_noise_angular_vel, process_noise_angular_vel  # 각속도
        ])

        # 측정 노이즈 R
        self.kf.R = np.diag([
            measurement_noise_pos, measurement_noise_pos, measurement_noise_pos,  # 위치
            measurement_noise_angle, measurement_noise_angle, measurement_noise_angle  # 각도
        ])

        # 초기 공분산
        self.kf.P *= 1.0

        self._initialized = False

    def initialize(self, position: np.ndarray, euler_angles: np.ndarray):
        """
        필터 초기화

        Args:
            position: [x, y, z] 초기 위치
            euler_angles: [roll, pitch, yaw] 초기 각도 (도)
        """
        self.kf.x = np.array([
            position[0], position[1], position[2],  # 위치
            0.0, 0.0, 0.0,                           # 속도
            euler_angles[0], euler_angles[1], euler_angles[2],  # 각도
            0.0, 0.0, 0.0                            # 각속도
        ])
        self._initialized = True

    def predict(self) -> PoseKalmanState:
        """예측 단계"""
        if not self._initialized:
            raise RuntimeError("Filter not initialized")

        self.kf.predict()
        return self.get_state()

    def update(
        self,
        position: np.ndarray,
        euler_angles: np.ndarray
    ) -> PoseKalmanState:
        """
        업데이트 단계

        Args:
            position: 측정된 위치 [x, y, z]
            euler_angles: 측정된 각도 [roll, pitch, yaw] (도)

        Returns:
            PoseKalmanState: 업데이트된 상태
        """
        if not self._initialized:
            self.initialize(position, euler_angles)
            return self.get_state()

        measurement = np.concatenate([position, euler_angles])
        self.kf.update(measurement)
        return self.get_state()

    def predict_and_update(
        self,
        position: np.ndarray,
        euler_angles: np.ndarray
    ) -> PoseKalmanState:
        """예측 + 업데이트 한번에"""
        self.predict()
        return self.update(position, euler_angles)

    def get_state(self) -> PoseKalmanState:
        """현재 상태 반환"""
        state = self.kf.x.flatten()
        return PoseKalmanState(
            position=state[0:3],
            velocity=state[3:6],
            orientation=state[6:9],
            angular_velocity=state[9:12],
            speed=np.linalg.norm(state[3:6])
        )

    def update_adaptive_noise(self, depth_distance: float):
        """
        거리 기반 적응형 노이즈 업데이트

        Args:
            depth_distance: 평균 깊이 거리 (미터)
        """
        # D455 depth 오차 모델: 거리²에 비례
        base_error = 0.005  # 1m에서 5mm
        pos_error = base_error * (depth_distance ** 2)
        pos_error = np.clip(pos_error, 0.002, 0.1)

        # 측정 노이즈 업데이트 (위치만)
        self.kf.R[0, 0] = pos_error ** 2
        self.kf.R[1, 1] = pos_error ** 2
        self.kf.R[2, 2] = pos_error ** 2

    def reset(self):
        """필터 리셋"""
        self.kf.x = np.zeros(12)
        self.kf.P = np.eye(12) * 1.0
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized
```

### 6.4 통합 측정 시스템 (main.py의 핵심 클래스)

```python
"""
nimg_v3 통합 측정 시스템
YOLO + FoundationPose + Kalman Filter 파이프라인
"""

import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import time

from .detection.yolo_detector import YOLODetector
from .pose.foundationpose_estimator import (
    FoundationPoseEstimator, PoseMode, TrackingState, PoseResult
)
from .measurement.pose_to_euler import pose_to_euler, EulerAngles
from .measurement.pose_kalman_filter import PoseKalmanFilter, PoseKalmanState

@dataclass
class MeasurementResult:
    """측정 결과"""
    # 탐지 정보
    detection_bbox: tuple          # (x1, y1, x2, y2)
    detection_confidence: float
    class_id: int
    class_name: str

    # 자세 정보
    pose_matrix: np.ndarray        # 4x4
    translation: np.ndarray        # [x, y, z] 미터
    euler_angles: EulerAngles      # roll, pitch, yaw (도)
    pose_confidence: float
    tracking_state: str

    # Kalman Filter 상태
    filtered_position: np.ndarray  # [x, y, z]
    velocity: np.ndarray           # [vx, vy, vz] m/s
    speed: float                   # m/s
    angular_velocity: np.ndarray   # [ωx, ωy, ωz] °/s

    # 기준 대비 변화량
    position_change: Optional[np.ndarray]  # [Δx, Δy, Δz]
    angle_change: Optional[np.ndarray]     # [Δroll, Δpitch, Δyaw]

    # 메타데이터
    frame_idx: int
    timestamp: float
    processing_time_ms: float

class IntegratedMeasurementSystem:
    """
    nimg_v3 통합 측정 시스템

    파이프라인:
    1. YOLO로 객체 탐지 및 마스크 생성
    2. FoundationPose로 6DoF 자세 추정/추적
    3. Kalman Filter로 속도 추정
    4. 기준 프레임 대비 변화량 계산
    """

    def __init__(
        self,
        yolo_model_path: str,
        foundationpose_model_dir: str,
        pose_mode: PoseMode = PoseMode.MODEL_FREE,
        mesh_path: Optional[str] = None,
        neural_field_dir: Optional[str] = None,
        intrinsics: Dict[str, float] = None,
        reference_frame_idx: int = 0,
        fps: float = 30.0,
        device: str = 'cuda:0'
    ):
        """
        Args:
            yolo_model_path: YOLO 모델 경로
            foundationpose_model_dir: FoundationPose 모델 디렉토리
            pose_mode: Model-Based 또는 Model-Free
            mesh_path: CAD 모델 경로 (Model-Based)
            neural_field_dir: Neural Field 경로 (Model-Free)
            intrinsics: 카메라 내부 파라미터
            reference_frame_idx: 기준 프레임 인덱스
            fps: 프레임 레이트
            device: 추론 디바이스
        """
        # YOLO 탐지기
        self.detector = YOLODetector(yolo_model_path, conf_threshold=0.5)

        # FoundationPose 추정기
        self.pose_estimator = FoundationPoseEstimator(
            model_dir=foundationpose_model_dir,
            mode=pose_mode,
            mesh_path=mesh_path,
            neural_field_dir=neural_field_dir,
            device=device
        )

        # Kalman Filter
        self.kalman_filter = PoseKalmanFilter(dt=1.0/fps)

        # 카메라 파라미터
        self.intrinsics = intrinsics or {
            'fx': 383.883, 'fy': 383.883,
            'cx': 320.499, 'cy': 237.913
        }

        # 기준 프레임 관리
        self.reference_frame_idx = reference_frame_idx
        self._reference_pose = None
        self._reference_euler = None
        self._reference_set = False

        # 프레임 카운터
        self._frame_count = 0

    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Optional[MeasurementResult]:
        """
        단일 프레임 처리

        Args:
            rgb: RGB 이미지 [H, W, 3]
            depth: Depth 이미지 [H, W] (미터 단위)
            timestamp: 프레임 타임스탬프

        Returns:
            MeasurementResult: 측정 결과 (탐지 실패 시 None)
        """
        start_time = time.time()
        frame_idx = self._frame_count
        self._frame_count += 1

        if timestamp is None:
            timestamp = frame_idx / 30.0

        # 1. YOLO 탐지
        detections = self.detector.detect(rgb)
        if len(detections) == 0:
            return None

        # 가장 신뢰도 높은 탐지 선택
        best_det = max(detections, key=lambda d: d.confidence)
        bbox = (best_det.x, best_det.y, best_det.x2, best_det.y2)

        # 2. 마스크 생성 (바운딩박스 기반)
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[best_det.y:best_det.y2, best_det.x:best_det.x2] = 255

        # 3. FoundationPose 자세 추정/추적
        force_estimate = (frame_idx == self.reference_frame_idx) or not self._reference_set

        pose_result = self.pose_estimator.process(
            rgb=rgb,
            depth=depth,
            mask=mask if force_estimate else None,
            intrinsics=self.intrinsics,
            force_estimate=force_estimate
        )

        # 4. 오일러 각도 추출
        translation, euler_angles = pose_to_euler(pose_result.pose_matrix)

        # 5. Kalman Filter 업데이트
        kalman_state = self.kalman_filter.predict_and_update(
            position=translation,
            euler_angles=np.array([euler_angles.roll, euler_angles.pitch, euler_angles.yaw])
        )

        # 6. 기준 프레임 설정 또는 변화량 계산
        position_change = None
        angle_change = None

        if frame_idx == self.reference_frame_idx:
            self._set_reference(pose_result.pose_matrix, euler_angles)
        elif self._reference_set:
            position_change = translation - self._reference_pose[:3, 3]
            angle_change = np.array([
                euler_angles.roll - self._reference_euler.roll,
                euler_angles.pitch - self._reference_euler.pitch,
                euler_angles.yaw - self._reference_euler.yaw
            ])
            # 각도 정규화
            angle_change = self._normalize_angles(angle_change)

        processing_time = (time.time() - start_time) * 1000

        return MeasurementResult(
            # 탐지 정보
            detection_bbox=bbox,
            detection_confidence=best_det.confidence,
            class_id=best_det.class_id,
            class_name=best_det.class_name,

            # 자세 정보
            pose_matrix=pose_result.pose_matrix,
            translation=translation,
            euler_angles=euler_angles,
            pose_confidence=pose_result.confidence,
            tracking_state=pose_result.tracking_state.value,

            # Kalman 상태
            filtered_position=kalman_state.position,
            velocity=kalman_state.velocity,
            speed=kalman_state.speed,
            angular_velocity=kalman_state.angular_velocity,

            # 변화량
            position_change=position_change,
            angle_change=angle_change,

            # 메타
            frame_idx=frame_idx,
            timestamp=timestamp,
            processing_time_ms=processing_time
        )

    def _set_reference(self, pose_matrix: np.ndarray, euler_angles: EulerAngles):
        """기준 프레임 설정"""
        self._reference_pose = pose_matrix.copy()
        self._reference_euler = euler_angles
        self._reference_set = True
        print(f"Reference frame set:")
        print(f"  Position: {pose_matrix[:3, 3]}")
        print(f"  Orientation: R={euler_angles.roll:.2f}°, P={euler_angles.pitch:.2f}°, Y={euler_angles.yaw:.2f}°")

    def _normalize_angles(self, angles: np.ndarray) -> np.ndarray:
        """각도를 -180 ~ 180 범위로 정규화"""
        angles = angles.copy()
        angles = np.where(angles > 180, angles - 360, angles)
        angles = np.where(angles < -180, angles + 360, angles)
        return angles

    def reset(self):
        """시스템 리셋"""
        self.pose_estimator.reset()
        self.kalman_filter.reset()
        self._reference_pose = None
        self._reference_euler = None
        self._reference_set = False
        self._frame_count = 0

    @property
    def is_reference_set(self) -> bool:
        return self._reference_set

    @property
    def is_tracking(self) -> bool:
        return self.pose_estimator.is_tracking
```

---

## 7. 구현 로드맵

### 7.1 Phase 1: 환경 구축 (1주)

```
□ 개발 환경 설정
  ├── □ FoundationPose 환경 설정 (Conda 또는 Docker)
  ├── □ 필수 패키지 설치 (PyTorch, Open3D, TensorRT 등)
  ├── □ RealSense D455 SDK 및 연동 테스트
  └── □ nimg_v3 디렉토리 구조 생성

□ FoundationPose 기본 테스트
  ├── □ 공개 데이터셋으로 추론 테스트
  ├── □ Model-Based 모드 테스트 (제공된 메시)
  └── □ Model-Free 모드 테스트 (제공된 참조 이미지)

□ Isaac ROS 설정 (Jetson용)
  ├── □ Isaac ROS 설치
  └── □ isaac_ros_foundationpose 설정
```

### 7.2 Phase 2: 객체 모델링 (1주)

```
□ Model-Free 설정 (권장)
  ├── □ 참조 이미지 촬영 도구 구현 (reference_capture.py)
  ├── □ 16개 참조 이미지 촬영 (다양한 시점)
  ├── □ Neural Object Field 학습
  └── □ 학습된 모델 테스트

□ Model-Based 설정 (대안)
  ├── □ 도장물 CAD 모델 확보 또는 생성
  ├── □ 메시 전처리 (스케일, 중심 정렬)
  └── □ Model-Based 추론 테스트
```

### 7.3 Phase 3: 핵심 모듈 구현 (2주)

```
□ 자세 추정 모듈
  ├── □ FoundationPoseEstimator 클래스 구현
  ├── □ Model-Based/Model-Free 모드 통합
  ├── □ 추적 모드 구현
  └── □ 추적 손실 복구 로직

□ 측정 모듈
  ├── □ pose_to_euler 변환 함수
  ├── □ PoseKalmanFilter 구현 (12-상태)
  ├── □ 속도/각속도 추정 테스트
  └── □ 변화량 계산기 (nimg_v2에서 계승)

□ 통합 파이프라인
  ├── □ IntegratedMeasurementSystem 구현
  ├── □ YOLO + FoundationPose 연동
  └── □ 실시간 스트리밍 테스트
```

### 7.4 Phase 4: 최적화 및 배포 (1주)

```
□ Jetson 최적화
  ├── □ TensorRT 변환 (가능한 경우)
  ├── □ 전력 모드 최적화
  └── □ 성능 벤치마크

□ 출력 모듈
  ├── □ CSV/JSON 내보내기 (nimg_v2 계승)
  ├── □ 시각화 모듈 (nimg_v2 계승)
  └── □ ROS2 토픽 발행 (선택)

□ 에러 처리
  ├── □ 탐지 실패 처리
  ├── □ 추적 손실 복구
  └── □ 신뢰도 기반 필터링
```

### 7.5 Phase 5: 테스트 및 검증 (1주)

```
□ 단위 테스트
  ├── □ test_foundationpose.py
  ├── □ test_pose_kalman.py
  └── □ test_measurement.py

□ 통합 테스트
  ├── □ 오프라인 데이터셋 테스트
  ├── □ 실시간 카메라 테스트
  └── □ 다양한 환경 테스트 (조명, 각도)

□ 성능 검증
  ├── □ 각도 정확도 측정 (Ground Truth 비교)
  ├── □ 속도 정확도 측정
  └── □ 처리 속도 (FPS) 측정

□ 문서화
  ├── □ API 문서
  ├── □ 사용자 가이드
  └── □ 트러블슈팅 가이드
```

---

## 8. 기술적 고려사항

### 8.1 Jetson Orin Nano Super 성능 특성

| 항목 | 사양 | 영향 |
|------|------|------|
| **AI 성능** | 67 TOPS (INT8) | 중형 모델 실시간 가능 |
| **GPU** | 1024 CUDA cores (Ampere) | TensorRT 최적화 필수 |
| **메모리** | 8GB LPDDR5 | 대형 모델 제한 |
| **TDP** | 25W (MAXN) | 열 관리 필요 |

### 8.2 예상 성능

| 모드 | 예상 FPS | GPU 사용률 | 메모리 사용 |
|------|----------|-----------|-----------|
| **자세 추정 (초기화)** | 0.7-1.0 | 100% | ~4GB |
| **추적 모드** | 30-40 | 60-80% | ~2GB |
| **YOLO + FP 추적** | 15-25 | 70-90% | ~3GB |

### 8.3 메모리 최적화 전략

```python
# 추적 모드 우선 사용
class OptimizedPipeline:
    def process(self, ...):
        # 추적 모드 우선 (더 빠름)
        if self.is_tracking:
            return self.track(...)

        # 초기화 또는 추적 손실 시에만 전체 추정
        return self.estimate(...)

# 배치 처리 지양
# 단일 프레임 처리 권장

# 메모리 해제
import torch
torch.cuda.empty_cache()
```

### 8.4 Domain Shift 대응

FoundationPose의 Zero-Shot 일반화로 기존 Domain Shift 문제 해결:

| 상황 | nimg_v2 | nimg_v3 |
|------|---------|---------|
| **새 조명** | 재학습 필요 | 그대로 사용 |
| **새 각도** | 재학습 필요 | 그대로 사용 |
| **새 객체** | 전체 재학습 | 참조 이미지만 추가 |

---

## 9. 테스트 및 검증 계획

### 9.1 정확도 검증 방법

#### 각도 정확도 (Ground Truth 비교)

```python
def validate_angle_accuracy():
    """
    알려진 자세의 객체로 각도 정확도 검증
    """
    # 정밀 회전 스테이지 사용 또는
    # 알려진 각도로 배치된 객체 촬영

    test_cases = [
        {'true_roll': 0, 'true_pitch': 0, 'true_yaw': 0},
        {'true_roll': 15, 'true_pitch': 0, 'true_yaw': 0},
        {'true_roll': 0, 'true_pitch': 30, 'true_yaw': 0},
        {'true_roll': 0, 'true_pitch': 0, 'true_yaw': 45},
    ]

    errors = []
    for case in test_cases:
        # 측정 수행
        result = system.process_frame(rgb, depth)

        # 오차 계산
        roll_error = abs(result.euler_angles.roll - case['true_roll'])
        pitch_error = abs(result.euler_angles.pitch - case['true_pitch'])
        yaw_error = abs(result.euler_angles.yaw - case['true_yaw'])

        errors.append({
            'roll_error': roll_error,
            'pitch_error': pitch_error,
            'yaw_error': yaw_error
        })

    # 결과 분석
    print(f"Roll RMSE: {np.sqrt(np.mean([e['roll_error']**2 for e in errors])):.2f}°")
    print(f"Pitch RMSE: {np.sqrt(np.mean([e['pitch_error']**2 for e in errors])):.2f}°")
    print(f"Yaw RMSE: {np.sqrt(np.mean([e['yaw_error']**2 for e in errors])):.2f}°")
```

#### 속도 정확도 (알려진 움직임)

```python
def validate_velocity_accuracy():
    """
    일정 속도로 이동하는 객체로 속도 정확도 검증
    """
    # 컨베이어 벨트 또는 선형 스테이지 사용
    true_velocity = 0.1  # m/s

    velocities = []
    for frame in video_frames:
        result = system.process_frame(frame.rgb, frame.depth)
        if result:
            velocities.append(result.speed)

    # 안정 구간의 평균
    stable_velocity = np.mean(velocities[10:])  # 초기 과도 상태 제외

    error_percent = abs(stable_velocity - true_velocity) / true_velocity * 100
    print(f"Velocity Error: {error_percent:.1f}%")
```

### 9.2 예상 결과

| 메트릭 | nimg_v2 (예상) | nimg_v3 (목표) |
|--------|---------------|---------------|
| **Roll 오차** | ±3-5° | ±1-2° |
| **Pitch 오차** | ±3-5° | ±1-2° |
| **Yaw 오차** | ±2-4° | ±0.5-1° |
| **속도 오차** | ±5-10% | ±2-5% |
| **처리 FPS (추적)** | ~30 | 15-25 |
| **Domain 적응** | 불가 | Zero-Shot |

---

## 10. 참고 자료

### 10.1 공식 리소스

- [FoundationPose GitHub](https://github.com/NVlabs/FoundationPose)
- [FoundationPose Project Page](https://nvlabs.github.io/FoundationPose/)
- [NVIDIA NGC - FoundationPose](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/foundationpose)
- [Isaac ROS FoundationPose](https://nvidia-isaac-ros.github.io/concepts/pose_estimation/foundationpose/index.html)

### 10.2 논문

- [FoundationPose: Unified 6D Pose Estimation and Tracking (CVPR 2024)](https://arxiv.org/abs/2312.08344)
- [BundleSDF: Neural 6-DoF Tracking (CVPR 2023)](https://arxiv.org/abs/2303.14158)

### 10.3 관련 프로젝트 문서

| 문서 | 경로 | 역할 |
|------|------|------|
| **nimg_v2 설계** | `research/251205/implementation_design_guide.md` | 기존 구현 참조 |
| **AI 연구** | `research/251218/ai_based_velocity_angle_measurement_research.md` | AI 접근법 비교 |
| **Domain Shift** | `research/251229/yolo_domain_shift_analysis_report.md` | 문제 분석 |
| **FP 구현 가이드** | `research/251231/foundationpose_implementation_guide.md` | 상세 구현 참조 |
| **FP 논문 번역** | `research/251224/FoundationPose_Korean_Translation.md` | 핵심 개념 이해 |

### 10.4 환경 설정 참조

```bash
# Conda 환경
conda create -n nimg_v3 python=3.9 -y
conda activate nimg_v3
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ultralytics filterpy open3d scipy
pip install git+https://github.com/NVlabs/nvdiffrast

# Docker (대안)
docker pull wenbowen123/foundationpose
```

---

## 부록: 빠른 시작 가이드

### A. Model-Free 빠른 시작

```python
from nimg_v3 import IntegratedMeasurementSystem, PoseMode

# 1. 시스템 초기화
system = IntegratedMeasurementSystem(
    yolo_model_path='models/yolo/class187_v12x.pt',
    foundationpose_model_dir='models/foundationpose',
    pose_mode=PoseMode.MODEL_FREE,
    neural_field_dir='models/neural_fields/painting_object',
    intrinsics={
        'fx': 383.883, 'fy': 383.883,
        'cx': 320.499, 'cy': 237.913
    }
)

# 2. 프레임 처리
result = system.process_frame(rgb, depth)

if result:
    print(f"Position: {result.translation}")
    print(f"Roll: {result.euler_angles.roll:.2f}°")
    print(f"Pitch: {result.euler_angles.pitch:.2f}°")
    print(f"Yaw: {result.euler_angles.yaw:.2f}°")
    print(f"Speed: {result.speed:.3f} m/s")
```

### B. 참조 이미지 촬영

```bash
# 참조 이미지 촬영 스크립트 실행
python scripts/capture_reference_images.py \
    --output_dir reference_images/my_object \
    --num_images 16

# Neural Object Field 학습
python scripts/train_neural_field.py \
    --ref_dir reference_images/my_object \
    --output_dir models/neural_fields/my_object
```

---

*작성: 2025-12-31*
*버전: v3.0*
*환경: Intel RealSense D455 + NVIDIA Jetson Orin Nano Super*
*목적: FoundationPose 기반 nimg_v3 종합 설계 및 구현 가이드*
