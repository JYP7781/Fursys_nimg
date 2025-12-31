# YOLO 모델 Domain Shift 분석 연구 보고서

**작성일**: 2025-12-29
**연구 목적**: 조명/각도 변화 환경에서 YOLO 객체인식 실패 원인 분석 및 해결방안 연구
**관련 프로젝트**: nimg_v2 (comprehensive_improvement_research_2025.md 기반 구현)

---

## 목차

1. [문제 정의](#1-문제-정의)
2. [Domain Shift의 개념](#2-domain-shift의-개념)
3. [조명/각도 변화가 객체인식에 미치는 영향](#3-조명각도-변화가-객체인식에-미치는-영향)
4. [YOLO 모델 일반화 실패 원인 분석](#4-yolo-모델-일반화-실패-원인-분석)
5. [해결 방안 및 개선 전략](#5-해결-방안-및-개선-전략)
6. [권장 구현 계획](#6-권장-구현-계획)
7. [참고 자료](#7-참고-자료)

---

## 1. 문제 정의

### 1.1 현재 상황

- **학습 환경**: 특정 조명 조건 및 카메라 각도에서 촬영된 이미지로 YOLO 모델 학습
- **테스트 환경**: 다른 조명 조건, 다른 카메라 각도에서 촬영된 새로운 영상
- **문제 현상**: 학습된 모델이 새로운 환경에서 객체를 전혀 인식하지 못함

### 1.2 핵심 질문

> "왜 기존 이미지에서 학습시킨 YOLO 모델이 조명이나 각도 등 다른 환경에서 촬영한 영상은 객체인식이 안되는가?"

---

## 2. Domain Shift의 개념

### 2.1 Domain Shift란?

**Domain Shift** (도메인 전이)는 학습 데이터(Source Domain)와 테스트 데이터(Target Domain) 간의 통계적 분포 차이로 인해 모델 성능이 저하되는 현상입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Domain Shift 개념도                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Source Domain (학습)          Target Domain (테스트)        │
│   ┌─────────────────┐          ┌─────────────────┐          │
│   │ • 특정 조명      │   ≠     │ • 다른 조명      │          │
│   │ • 특정 카메라    │          │ • 다른 카메라    │          │
│   │ • 특정 배경      │          │ • 다른 배경      │          │
│   │ • 특정 각도      │          │ • 다른 각도      │          │
│   └─────────────────┘          └─────────────────┘          │
│            │                            │                    │
│            ▼                            ▼                    │
│      높은 정확도 (99%)            낮은 정확도 (~50%)         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Domain Gap의 유형

| 유형 | 설명 | 예시 |
|------|------|------|
| **Covariate Shift** | 입력 데이터 분포 변화 | 조명, 각도, 해상도 변화 |
| **Concept Drift** | 입력-출력 관계 변화 | 객체의 외관 변화 |
| **Geometric Shift** | 기하학적 특성 변화 | 카메라 위치, 시점 변화 |
| **Style Shift** | 이미지 스타일 변화 | 색상, 질감, 노이즈 변화 |

---

## 3. 조명/각도 변화가 객체인식에 미치는 영향

### 3.1 조명 변화의 영향

#### 3.1.1 물리적 현상

조명 변화는 다음과 같은 이미지 특성 변화를 유발합니다:

| 조명 요소 | 영향 | 결과 |
|----------|------|------|
| **밝기 (Brightness)** | 픽셀 강도 전체 변화 | 과노출/저노출로 디테일 손실 |
| **대비 (Contrast)** | 밝은/어두운 영역 차이 | 객체 경계 모호화 |
| **그림자 (Shadow)** | 객체 형태 왜곡 | 잘못된 객체 분할 |
| **반사 (Reflection)** | 하이라이트 추가 | 특징 추출 오류 |
| **색상 온도** | 전체 색조 변화 | HSV 공간에서의 불일치 |

#### 3.1.2 CNN 특징 추출에 미치는 영향

```python
# 조명 변화가 CNN 특징에 미치는 영향 예시
"""
학습 환경: 표준 조명 (평균 픽셀값 ~128)
테스트 환경: 저조도 조명 (평균 픽셀값 ~40)

결과:
- 초기 Conv 레이어: 완전히 다른 활성화 패턴
- 특징 맵: 학습된 패턴과 불일치
- 분류 결과: 낮은 confidence 또는 오탐지
"""
```

### 3.2 카메라 각도 변화의 영향

#### 3.2.1 기하학적 변형

| 변형 유형 | 설명 | CNN에 미치는 영향 |
|----------|------|------------------|
| **회전 (Rotation)** | 이미지 내 객체 회전 | 표준 CNN은 회전에 민감함 |
| **축소/확대 (Scale)** | 거리에 따른 크기 변화 | 앵커 박스 불일치 |
| **기울기 (Perspective)** | 시점에 따른 왜곡 | 바운딩 박스 형태 변화 |
| **가림 (Occlusion)** | 새로운 시점에서의 부분 가림 | 특징 불완전성 |

#### 3.2.2 CNN의 불변성 한계

최신 연구에 따르면, **CNN은 이론적으로 이동(Translation) 불변성**을 가지지만:

- **회전 불변성(Rotation Invariance)**: 부재 - 회전된 객체를 다른 객체로 인식
- **스케일 불변성(Scale Invariance)**: 제한적 - 앵커 박스에 의존
- **시점 불변성(Viewpoint Invariance)**: 부재 - 다른 각도는 완전히 새로운 패턴

### 3.3 복합 효과

조명과 각도 변화가 동시에 발생하면 영향이 기하급수적으로 증가합니다:

```
Domain Gap = f(조명 변화) × f(각도 변화) × f(기타 환경 요인)

예시:
- 조명만 변화: 정확도 70% 유지 가능
- 각도만 변화: 정확도 65% 유지 가능
- 조명 + 각도 변화: 정확도 40% 이하로 급락
```

---

## 4. YOLO 모델 일반화 실패 원인 분석

### 4.1 학습 데이터 문제

#### 4.1.1 데이터 다양성 부족

| 문제 | 설명 | 영향 |
|------|------|------|
| **단일 환경 학습** | 한 가지 조명/각도만 포함 | 해당 조건에만 동작 |
| **낮은 데이터 다양성** | 유사한 이미지 반복 | 패턴 과적합 |
| **클래스 불균형** | 특정 객체 비율 편향 | 일부 객체 인식 저하 |

#### 4.1.2 데이터 증강 부족

권장 학습 데이터 가이드라인 (Ultralytics 권장):

| 항목 | 권장 최소값 | 현재 상태 (추정) |
|------|------------|-----------------|
| 클래스당 이미지 수 | ≥ 1,500장 | ? |
| 클래스당 객체 인스턴스 | ≥ 10,000개 | ? |
| 환경 다양성 | 다양한 조명/날씨/시간대 | 단일 환경 |
| 카메라 다양성 | 다양한 각도/거리 | 단일 설정 |

### 4.2 모델 과적합 (Overfitting)

#### 4.2.1 과적합 현상

```
┌─────────────────────────────────────────────────────────────┐
│                     과적합 진단                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   학습 손실 (Training Loss)                                  │
│   ────────────────────────────                               │
│   높음                                                       │
│    │ ╲                                                       │
│    │  ╲                                                      │
│    │   ╲───────────────  (지속 감소)                         │
│   낮음                                                       │
│       └─────────────────────▶ Epoch                          │
│                                                              │
│   검증 손실 (Validation Loss)                                │
│   ────────────────────────────                               │
│   높음       ╱─────  (증가 시작 = 과적합!)                   │
│    │ ╲      ╱                                                │
│    │  ╲    ╱                                                 │
│    │   ╲__╱  (최저점)                                        │
│   낮음                                                       │
│       └─────────────────────▶ Epoch                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 4.2.2 과적합 원인

1. **모델 복잡성 과다**: 데이터 양 대비 파라미터 수가 너무 많음
2. **데이터 양 부족**: 충분한 패턴 학습 불가
3. **학습 에폭 과다**: 조기 종료(Early Stopping) 미적용
4. **정규화 부족**: Dropout, Weight Decay 미사용

### 4.3 Feature 학습의 근본적 한계

#### 4.3.1 조명 불변 특징 추출의 어려움

연구에 따르면:

> "저조도 환경의 영향은 단순한 '조명 불변성(Illumination Invariance)'으로 해결될 수 없을 만큼 특징 추출 깊숙이 영향을 미친다."

```python
# 조명 변화에 따른 CNN 활성화 변화
"""
표준 조명에서 학습된 필터:
- 특정 밝기/대비 패턴에 최적화
- 에지 검출이 특정 그래디언트 강도에 튜닝됨

저조도에서의 문제:
- 노이즈 대 신호 비율 증가
- 에지 그래디언트 약화
- 색상 채널 정보 손실
"""
```

#### 4.3.2 표준 CNN의 불변성 부재

| 불변성 유형 | 이론적 기대 | 실제 성능 |
|------------|------------|----------|
| Translation | 있음 (Convolution 특성) | 대체로 유지 |
| Rotation | 없음 | 학습된 회전만 인식 |
| Scale | 제한적 (피라미드 구조) | 앵커 박스 범위 내만 |
| Illumination | 없음 | 학습된 조명만 인식 |
| Viewpoint | 없음 | 학습된 시점만 인식 |

### 4.4 YOLO 특화 문제

#### 4.4.1 앵커 박스 불일치

```python
"""
학습 시 앵커 박스:
- 학습 데이터의 객체 크기/비율에 맞춤
- 특정 거리/각도에서의 바운딩 박스 통계

테스트 시 문제:
- 다른 각도 → 다른 종횡비
- 다른 거리 → 다른 스케일
- 앵커 박스와 불일치 → IoU 저하 → 탐지 실패
"""
```

#### 4.4.2 Confidence Threshold 문제

```python
"""
학습 환경: confidence 0.7 ~ 0.95
새 환경: confidence 0.2 ~ 0.4

결과: threshold (예: 0.5) 이하로 필터링되어 탐지 결과 없음
"""
```

---

## 5. 해결 방안 및 개선 전략

### 5.1 즉시 적용 가능한 해결책

#### 5.1.1 Confidence Threshold 조정

```python
# 기존 설정
detector = YOLODetector(model_path, conf_threshold=0.5)

# 권장 설정 (새 환경에서)
detector = YOLODetector(model_path, conf_threshold=0.25)  # 더 낮은 임계값

# 또는 동적 조정
def adaptive_threshold(environment_type):
    if environment_type == 'new_lighting':
        return 0.15  # 매우 낮게
    elif environment_type == 'new_angle':
        return 0.25
    else:
        return 0.5
```

#### 5.1.2 NMS 임계값 조정

```python
# NMS IoU 임계값 조정
iou_threshold=0.45  # 기본값
iou_threshold=0.3   # 더 공격적 중복 제거
```

### 5.2 데이터 증강 전략 (Training)

#### 5.2.1 조명 관련 증강

```python
# Ultralytics YOLO 기본 증강 파라미터
augmentation_params = {
    'hsv_h': 0.015,    # Hue 조정 (기본값)
    'hsv_s': 0.7,      # Saturation 조정 (권장: 높임)
    'hsv_v': 0.4,      # Value(밝기) 조정 (권장: 높임)
}

# 권장 강화 설정 (조명 변화에 강건하게)
enhanced_augmentation = {
    'hsv_h': 0.05,     # Hue 범위 확대
    'hsv_s': 0.9,      # Saturation 범위 확대
    'hsv_v': 0.6,      # Value 범위 크게 확대
}
```

#### 5.2.2 기하학적 증강

```python
# 각도 변화에 대응하는 증강
geometric_augmentation = {
    'degrees': 15,      # 회전 범위 (기본: 0)
    'translate': 0.1,   # 이동 범위
    'scale': 0.5,       # 스케일 변화 범위
    'shear': 10,        # 전단 변형
    'perspective': 0.001,  # 원근 변환
    'flipud': 0.0,      # 상하 반전
    'fliplr': 0.5,      # 좌우 반전
}
```

#### 5.2.3 Albumentations 활용

```python
import albumentations as A

transform = A.Compose([
    # 조명 관련
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.5
    ),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),

    # 각도 관련
    A.Rotate(limit=30, p=0.5),
    A.Affine(
        scale=(0.8, 1.2),
        translate_percent=(-0.1, 0.1),
        rotate=(-20, 20),
        shear=(-10, 10),
        p=0.5
    ),

    # 노이즈 및 블러
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.MotionBlur(blur_limit=7, p=0.2),
])
```

### 5.3 도메인 적응 기법 (Domain Adaptation)

#### 5.3.1 Fine-Tuning (미세 조정)

가장 실용적인 접근법:

```python
from ultralytics import YOLO

# 기존 모델 로드
model = YOLO('path/to/pretrained_model.pt')

# 새 환경 데이터로 미세 조정
model.train(
    data='new_environment.yaml',
    epochs=50,                    # 적은 에폭
    lr0=0.001,                    # 낮은 학습률
    freeze=10,                    # 초기 레이어 동결
    imgsz=640,
)
```

#### 5.3.2 SSDA-YOLO (Semi-Supervised Domain Adaptive)

연구 기반 해결책:

```python
"""
SSDA-YOLO 특징:
1. Mean Teacher 모델로 타겟 도메인의 인스턴스 수준 특징 학습
2. Scene Style Transfer로 도메인 간 이미지 수준 차이 보정
3. Consistency Loss로 교차 도메인 예측 정렬

GitHub: https://github.com/hnuzhy/SSDA-YOLO
"""
```

#### 5.3.3 SF-YOLO (Source-Free Domain Adaptation)

소스 데이터 없이 적응:

```python
"""
SF-YOLO 특징:
- Teacher-Student 프레임워크 사용
- 타겟 도메인 특화 증강 학습
- 레이블 없는 타겟 데이터만으로 학습

GitHub: https://github.com/vs-cv/sf-yolo
"""
```

### 5.4 저조도 특화 YOLO 변형

#### 5.4.1 YOLO-D (Domain Adaptive)

```python
"""
YOLO-D 아키텍처:
1. LLEM (Low Light Enhancement Module): 전처리 단계에서 노출/대비 개선
2. MS-DAN (Multiscale Domain Adaptive Network): 도메인 불변 특징 학습
3. YOLOv8 기반

성능: 저조도에서 기존 대비 mAP 15-20% 향상
"""
```

#### 5.4.2 Dark-YOLO

```python
"""
Dark-YOLO 구성요소:
1. SCI 모듈: Self-Calibrated Illumination
2. PSM 모듈: 핵심 특징 추출
3. CSPPF 모듈: 저조도 이미지 품질 향상
4. D-RAMiT: 공간/채널 특징 융합
"""
```

### 5.5 하드웨어/센서 수준 해결책

#### 5.5.1 적응형 카메라 센서 제어

```python
"""
'Lens' 접근법:
- ISO, 셔터 속도, 조리개를 동적으로 제어
- 각 장면에 맞게 환경광 최적화
- 데이터 수집 단계에서 도메인 갭 감소
"""
```

### 5.6 재학습 전략

#### 5.6.1 혼합 데이터셋 학습

```python
# 다양한 환경의 이미지를 포함한 데이터셋 구성
mixed_dataset = {
    'original_environment': 5000,  # 기존 학습 환경
    'new_lighting_bright': 1000,   # 밝은 조명
    'new_lighting_dark': 1000,     # 어두운 조명
    'new_angle_45deg': 1000,       # 45도 각도
    'new_angle_overhead': 1000,    # 상단 뷰
}

# 총: 9,000장 (다양성 확보)
```

#### 5.6.2 점진적 도메인 확장

```python
"""
Phase 1: 원본 데이터로 기본 학습
Phase 2: 합성 증강 데이터 추가 학습
Phase 3: 새 환경의 소량 레이블 데이터로 미세 조정
Phase 4: 새 환경 데이터 지속 추가 및 재학습
"""
```

---

## 6. 권장 구현 계획

### 6.1 즉시 조치 (1-2일)

| 작업 | 설명 | 예상 효과 |
|------|------|----------|
| Confidence 임계값 낮춤 | 0.5 → 0.15~0.25 | 부분적 탐지 복구 |
| 테스트 이미지 전처리 | 밝기/대비 정규화 | 약간의 개선 |
| 결과 시각화 | 탐지 결과 디버깅 | 문제 진단 |

### 6.2 단기 조치 (1-2주)

| 작업 | 설명 | 예상 효과 |
|------|------|----------|
| 새 환경 데이터 수집 | 다양한 조명/각도 이미지 | 근본적 개선 기반 |
| 데이터 증강 강화 | HSV, 기하학적 변환 확대 | 일반화 능력 향상 |
| Fine-tuning 수행 | 혼합 데이터셋으로 재학습 | 상당한 성능 향상 |

### 6.3 중장기 조치 (2-4주)

| 작업 | 설명 | 예상 효과 |
|------|------|----------|
| Domain Adaptation 기법 적용 | SSDA-YOLO 또는 SF-YOLO | 도메인 갭 근본 해결 |
| 저조도 특화 모델 검토 | YOLO-D, Dark-YOLO | 조명 변화에 강건 |
| 테스트 파이프라인 개선 | 실시간 적응형 전처리 | 운영 안정성 |

### 6.4 구현 코드 예시

```python
# improved_detector.py
import cv2
import numpy as np
from ultralytics import YOLO

class RobustYOLODetector:
    """조명/각도 변화에 강건한 YOLO 탐지기"""

    def __init__(self, model_path, conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def preprocess_for_lighting(self, image):
        """조명 정규화 전처리"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced

    def detect_with_multi_conf(self, image, conf_levels=[0.5, 0.3, 0.15]):
        """다중 confidence 레벨로 탐지 시도"""
        for conf in conf_levels:
            results = self.model(image, conf=conf, verbose=False)
            if len(results[0].boxes) > 0:
                return results, conf
        return results, conf_levels[-1]

    def detect(self, image, enhance=True):
        """향상된 탐지"""
        if enhance:
            image = self.preprocess_for_lighting(image)

        results, used_conf = self.detect_with_multi_conf(image)

        detections = []
        for box in results[0].boxes:
            detections.append({
                'bbox': box.xyxy[0].cpu().numpy(),
                'confidence': float(box.conf[0]),
                'class_id': int(box.cls[0]),
                'used_threshold': used_conf
            })

        return detections
```

---

## 7. 참고 자료

### 7.1 Domain Adaptation 관련

- [SSDA-YOLO: Semi-supervised Domain Adaptive YOLO](https://www.sciencedirect.com/science/article/abs/pii/S1077314223000292) - ScienceDirect
- [Source-Free Domain Adaptation for YOLO Object Detection](https://arxiv.org/html/2409.16538v1) - arXiv
- [Attention-Based Domain Adaptive YOLO](https://link.springer.com/chapter/10.1007/978-981-96-6594-5_22) - Springer

### 7.2 저조도 객체 탐지

- [YOLO-D: A Domain Adaptive approach towards low light object detection](https://www.sciencedirect.com/science/article/pii/S1877050925016667) - ScienceDirect
- [Dark-YOLO: Low-Light Object Detection Algorithm](https://www.mdpi.com/2076-3417/15/9/5170) - MDPI
- [Enhancing low-light object detection with En-YOLO](https://link.springer.com/article/10.1007/s00530-025-01820-7) - Springer

### 7.3 데이터 증강 및 학습 최적화

- [Ultralytics YOLO Data Augmentation](https://docs.ultralytics.com/guides/yolo-data-augmentation/) - 공식 문서
- [YOLOv8 Best Practices for Training](https://medium.com/internet-of-technology/yolov8-best-practices-for-training-cdb6eacf7e4f) - Medium
- [Tips for Best YOLOv5 Training Results](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/) - Ultralytics

### 7.4 과적합 및 일반화

- [Overfitting in Machine Learning](https://www.ultralytics.com/glossary/overfitting) - Ultralytics
- [Identifying Bias in Deep Neural Networks Using Image Transforms](https://arxiv.org/html/2412.13079v1) - arXiv
- [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486) - arXiv

### 7.5 산업용 객체 탐지

- [Advancing Industrial Object Detection Through Domain Adaptation](https://www.mdpi.com/2076-0825/13/12/513) - MDPI
- [Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2024/html/Du_Boosting_Object_Detection_with_Zero-Shot_Day-Night_Domain_Adaptation_CVPR_2024_paper.html) - CVPR 2024
- [Object Detectors in the Open Environment: Challenges, Solutions, and Outlook](https://arxiv.org/html/2403.16271v1) - arXiv

### 7.6 GitHub 저장소

- [SSDA-YOLO](https://github.com/hnuzhy/SSDA-YOLO)
- [SF-YOLO (Source-Free YOLO)](https://github.com/vs-cv/sf-yolo)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

---

## 결론

### 핵심 원인 요약

1. **Domain Shift**: 학습 환경과 테스트 환경의 통계적 분포 차이
2. **데이터 다양성 부족**: 단일 조명/각도 환경에서만 학습
3. **CNN의 불변성 한계**: 회전, 조명 변화에 대한 내재적 취약성
4. **과적합**: 특정 환경 패턴에 과도하게 최적화

### 권장 해결 순서

1. **즉시**: Confidence 임계값 조정 및 전처리 적용
2. **단기**: 다양한 환경 데이터 수집 및 강화된 증강으로 재학습
3. **중장기**: Domain Adaptation 기법 적용 및 저조도 특화 모델 검토

### 예상 성능 개선

| 단계 | 적용 후 예상 정확도 |
|------|-------------------|
| 현재 | ~0% (탐지 안됨) |
| 즉시 조치 후 | 30-50% |
| 단기 조치 후 | 70-85% |
| 중장기 조치 후 | 90%+ |

---

*작성: 2025-12-29*
*연구 목적: YOLO 모델의 Domain Shift 문제 분석 및 해결방안 연구*
*관련 프로젝트: nimg_v2 (src/nimg_v2)*
