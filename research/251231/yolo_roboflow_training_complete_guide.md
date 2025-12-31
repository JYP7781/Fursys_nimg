# YOLO 커스텀 모델 학습 완전 가이드
## Roboflow 라벨링부터 학습/배포까지

**작성일**: 2025-12-31
**목적**: 도장물 객체 인식을 위한 YOLO 커스텀 모델 학습 전체 프로세스 가이드
**관련 문서**: yolo_domain_shift_analysis_report.md, ai_based_velocity_angle_measurement_research.md

---

## 목차

1. [개요 및 배경](#1-개요-및-배경)
2. [환경 분석 및 데이터 요구사항](#2-환경-분석-및-데이터-요구사항)
3. [Roboflow 프로젝트 설정](#3-roboflow-프로젝트-설정)
4. [데이터 라벨링 상세 가이드](#4-데이터-라벨링-상세-가이드)
5. [데이터 증강 전략](#5-데이터-증강-전략)
6. [데이터셋 버전 생성 및 내보내기](#6-데이터셋-버전-생성-및-내보내기)
7. [YOLO 모델 학습](#7-yolo-모델-학습)
8. [모델 평가 및 검증](#8-모델-평가-및-검증)
9. [Domain Shift 대응 전략](#9-domain-shift-대응-전략)
10. [배포 및 운영](#10-배포-및-운영)
11. [참고 자료](#11-참고-자료)

---

## 1. 개요 및 배경

### 1.1 현재 상황

기존 YOLO 모델이 특정 환경(조명, 각도)에서만 학습되어 새로운 환경에서 객체 인식 실패가 발생하고 있습니다. 이 가이드는 **Domain Shift 문제를 근본적으로 해결**하기 위한 체계적인 데이터 수집, 라벨링, 학습 프로세스를 제공합니다.

### 1.2 목표

| 목표 | 설명 |
|------|------|
| **다양한 환경 대응** | 조명 변화, 각도 변화에 강건한 모델 구축 |
| **높은 정확도** | mAP@0.5 90% 이상 달성 |
| **실시간 추론** | Jetson Orin Nano Super에서 30+ FPS |
| **Domain Shift 해결** | 새로운 환경에서도 안정적인 인식 |

### 1.3 YOLO 버전 선택 가이드

| 모델 | 파라미터 | mAP@0.5 | 추론 속도 | Jetson 적합성 | 권장 용도 |
|------|----------|---------|----------|--------------|----------|
| **YOLOv11n** | 2.6M | 39.5% | 매우 빠름 | ★★★★★ | 엣지 디바이스, 빠른 프로토타입 |
| **YOLOv11s** | 9.4M | 47.0% | 빠름 | ★★★★★ | **권장: 속도/정확도 균형** |
| **YOLOv11m** | 20.1M | 51.5% | 중간 | ★★★★☆ | 정확도 중시 |
| **YOLOv11l** | 25.3M | 53.4% | 느림 | ★★★☆☆ | 서버/클라우드 |
| **YOLOv11x** | 56.9M | 54.7% | 매우 느림 | ★★☆☆☆ | 최고 정확도 필요 시 |

**권장**: 본 프로젝트에서는 **YOLOv11s**를 기본으로 사용하되, 성능 테스트 후 조정

---

## 2. 환경 분석 및 데이터 요구사항

### 2.1 현재 이미지 데이터 분석

현재 프로젝트의 이미지 데이터 구조:

```
src/nimg/img_data/
├── 20240525/
│   ├── p1/          # 14개 이미지
│   ├── t4/          # 40개 이미지
│   └── t5/          # 28개 이미지
└── 20240529/        # 28개 이미지
```

**문제점**:
- 총 이미지 수: ~110장 (매우 부족)
- 단일 환경 조건
- 다양성 부족

### 2.2 데이터 요구사항 (Domain Shift 해결 기준)

| 항목 | 최소 권장 | 이상적 | 현재 상태 |
|------|----------|--------|----------|
| **클래스당 이미지 수** | 1,500장 | 3,000장+ | ~110장 |
| **클래스당 객체 인스턴스** | 10,000개 | 20,000개+ | ? |
| **조명 조건 종류** | 5가지+ | 10가지+ | 1가지 |
| **카메라 각도 종류** | 5가지+ | 10가지+ | 1-2가지 |
| **배경 다양성** | 다양 | 매우 다양 | 단일 |

### 2.3 데이터 수집 체크리스트

#### 조명 조건 (최소 5가지)

```
□ 표준 조명 (현재 환경)
□ 밝은 조명 (형광등 직접 조명)
□ 어두운 조명 (저조도)
□ 측면 조명 (그림자 발생)
□ 역광 조명
□ 혼합 조명 (자연광 + 인공광)
□ 깜빡이는 조명 (산업 환경)
```

#### 카메라 각도 (최소 5가지)

```
□ 정면 뷰 (0°)
□ 상단 뷰 (45°)
□ 측면 뷰 좌 (30°)
□ 측면 뷰 우 (30°)
□ 비스듬한 각도 (15°)
□ 완전 상단 뷰 (90°)
```

#### 거리 변화

```
□ 가까운 거리 (0.5m)
□ 중간 거리 (1.0m) - 기준
□ 먼 거리 (1.5m)
□ 매우 먼 거리 (2.0m+)
```

---

## 3. Roboflow 프로젝트 설정

### 3.1 계정 생성 및 프로젝트 시작

1. [Roboflow](https://roboflow.com) 접속 후 무료 계정 생성
2. 새 프로젝트 생성:
   - **Project Name**: `fursys-painting-object-detection`
   - **Project Type**: `Object Detection`
   - **Annotation Group**: `Object Detection (Bounding Box)`

### 3.2 프로젝트 설정 권장사항

```yaml
# 프로젝트 설정
project_config:
  name: "fursys-painting-object-detection"
  type: "object-detection"
  license: "Private"

  # 클래스 정의
  classes:
    - name: "painting_object"
      description: "도장물 객체"
      color: "#FF6B6B"
    # 필요시 추가 클래스 정의
    # - name: "defect"
    #   description: "불량품"
    #   color: "#FFE66D"
```

### 3.3 이미지 업로드

#### 방법 1: 웹 인터페이스

1. 프로젝트 대시보드에서 "Upload" 클릭
2. 이미지 파일 드래그 앤 드롭
3. 업로드 완료 확인

#### 방법 2: Roboflow Python SDK

```python
from roboflow import Roboflow

# API 키 설정
rf = Roboflow(api_key="YOUR_API_KEY")

# 프로젝트 접근
project = rf.workspace("your-workspace").project("fursys-painting-object-detection")

# 이미지 업로드
import os
image_dir = "/path/to/images"

for image_file in os.listdir(image_dir):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, image_file)
        project.upload(image_path)
        print(f"Uploaded: {image_file}")
```

#### 방법 3: CLI 사용

```bash
# Roboflow CLI 설치
pip install roboflow

# 로그인
roboflow login

# 이미지 업로드
roboflow upload ./images --project fursys-painting-object-detection
```

---

## 4. 데이터 라벨링 상세 가이드

### 4.1 라벨링 기본 원칙

#### 바운딩 박스 규칙

```
┌─────────────────────────────────────────────────────────────┐
│                    바운딩 박스 가이드                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   올바른 라벨링:                                              │
│   ┌─────────────┐                                            │
│   │ ┌─────────┐ │  ← 객체에 딱 맞게 (tight)                 │
│   │ │ 객  체  │ │                                            │
│   │ └─────────┘ │                                            │
│   └─────────────┘                                            │
│                                                              │
│   잘못된 라벨링 (너무 큼):                                    │
│   ┌─────────────────┐                                        │
│   │                 │  ← 불필요한 여백 포함                  │
│   │   ┌───────┐    │                                        │
│   │   │ 객체  │    │                                        │
│   │   └───────┘    │                                        │
│   │                 │                                        │
│   └─────────────────┘                                        │
│                                                              │
│   잘못된 라벨링 (잘림):                                       │
│   ┌─────────┐                                                │
│   │ 객  체 ──┼──  ← 객체 일부가 잘림                         │
│   └─────────┘                                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 핵심 규칙

| 규칙 | 설명 | 중요도 |
|------|------|--------|
| **Tight Bounding Box** | 객체 경계에 딱 맞게 박스 그리기 | ★★★★★ |
| **완전한 객체 포함** | 객체의 모든 부분이 박스 안에 포함 | ★★★★★ |
| **일관성 유지** | 모든 이미지에서 동일한 기준 적용 | ★★★★★ |
| **가려진 객체 처리** | 부분적으로 가려진 객체도 전체 크기로 라벨링 | ★★★★☆ |

### 4.2 Roboflow Annotate 사용법

#### 기본 인터페이스

1. **이미지 선택**: Annotate 탭에서 라벨링할 이미지 선택
2. **바운딩 박스 도구**:
   - 키보드 `B` 또는 사이드바에서 박스 도구 선택
   - 드래그하여 박스 그리기
3. **클래스 할당**: 그린 박스에 클래스 이름 할당
4. **저장**: `Enter` 또는 저장 버튼 클릭

#### 단축키 목록

| 단축키 | 기능 |
|--------|------|
| `B` | 바운딩 박스 도구 |
| `V` | 선택 도구 |
| `Delete` | 선택한 어노테이션 삭제 |
| `Enter` | 저장 및 다음 이미지 |
| `←` / `→` | 이전/다음 이미지 |
| `Ctrl+Z` | 실행 취소 |
| `Ctrl+S` | 저장 |

### 4.3 AI 보조 라벨링 활용

#### Label Assist 사용

```
1. Roboflow Annotate에서 마법봉 아이콘 클릭
2. 사용할 모델 선택:
   - COCO 데이터셋 모델 (80개 클래스)
   - Roboflow Universe의 공개 모델
   - 이전에 학습한 자체 모델
3. AI가 제안한 박스 검토 및 수정
```

#### Auto Label (대량 라벨링)

```python
# Auto Label 설정 예시
auto_label_config = {
    "classes": [
        {
            "name": "painting_object",
            "description": "industrial painting object on conveyor belt",
            "prompts": [
                "painted metal object",
                "industrial component",
                "manufacturing part"
            ]
        }
    ],
    "confidence_threshold": 0.5,
    "model": "grounding-dino"  # 또는 "segment-anything"
}
```

#### SAM (Segment Anything) 활용

1. 사이드바에서 SAM 도구 선택
2. 객체 영역 클릭
3. 자동 생성된 마스크를 바운딩 박스로 변환
4. 필요시 수동 조정

### 4.4 라벨링 품질 체크리스트

#### 개별 이미지 검증

```
□ 모든 객체가 라벨링 되었는가?
□ 바운딩 박스가 객체에 딱 맞는가?
□ 올바른 클래스가 할당되었는가?
□ 중복 라벨링이 없는가?
□ 가려진 객체도 처리되었는가?
```

#### 데이터셋 전체 검증

```
□ 클래스별 이미지 수가 균형 잡혀 있는가?
□ 다양한 환경 조건이 포함되었는가?
□ 라벨링 일관성이 유지되는가?
□ null/empty 어노테이션이 없는가?
```

### 4.5 산업용 객체 라벨링 팁

#### 도장물 특화 가이드라인

```yaml
labeling_guidelines:
  # 도장물 특성 고려
  object_boundaries:
    - 도장면 경계를 기준으로 라벨링
    - 그림자는 포함하지 않음
    - 반사광으로 인한 하이라이트는 객체에 포함

  # 가려짐(Occlusion) 처리
  occlusion:
    - 50% 이상 보이는 객체: 전체 크기로 라벨링
    - 50% 미만 보이는 객체: 라벨링 제외 또는 별도 클래스

  # 경계 케이스
  edge_cases:
    - 이미지 경계에 걸친 객체: 보이는 부분만 라벨링
    - 매우 작은 객체 (< 20px): 라벨링 제외
    - 블러된 객체: confidence가 낮더라도 라벨링
```

---

## 5. 데이터 증강 전략

### 5.1 Roboflow 내장 증강

#### 전처리 (Preprocessing)

| 옵션 | 설명 | 권장 설정 |
|------|------|----------|
| **Auto-Orient** | EXIF 기반 자동 회전 | ✅ 활성화 |
| **Resize** | 이미지 크기 조정 | 640x640 (YOLOv11 기준) |
| **Grayscale** | 흑백 변환 | ❌ 비활성화 |
| **Static Crop** | 특정 영역 크롭 | 필요시 설정 |

#### 증강 (Augmentation)

**조명 변화 대응 (필수)**:

```yaml
augmentation_lighting:
  brightness:
    enabled: true
    range: [-25%, +25%]  # 권장: 넓은 범위

  exposure:
    enabled: true
    range: [-20%, +20%]

  saturation:
    enabled: true
    range: [-30%, +30%]

  hue:
    enabled: true
    range: [-15°, +15°]
```

**기하학적 변환 (필수)**:

```yaml
augmentation_geometric:
  rotation:
    enabled: true
    range: [-15°, +15°]  # 도장물 특성상 큰 회전 필요 없음

  flip:
    horizontal: true
    vertical: false  # 수직 뒤집기는 비현실적

  shear:
    enabled: true
    range: [-10°, +10°]

  crop:
    enabled: true
    min_zoom: 0%
    max_zoom: 20%
```

**노이즈 및 블러**:

```yaml
augmentation_noise:
  blur:
    enabled: true
    max_pixels: 2.5

  noise:
    enabled: true
    max_percent: 3%

  cutout:
    enabled: true
    count: 3
    size: 5%  # 작은 영역 제거로 폐색 학습
```

### 5.2 고급 증강 (Albumentations 활용)

Roboflow 외부에서 추가 증강이 필요한 경우:

```python
import albumentations as A
import cv2

# 산업용 객체 특화 증강 파이프라인
transform = A.Compose([
    # 조명 변화 (강력)
    A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=1.0
        ),
        A.CLAHE(clip_limit=4.0, p=1.0),
        A.RandomGamma(gamma_limit=(60, 140), p=1.0),
    ], p=0.7),

    # 색상 변화
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=40,
        val_shift_limit=30,
        p=0.5
    ),

    # 기하학적 변환
    A.Affine(
        scale=(0.85, 1.15),
        translate_percent=(-0.1, 0.1),
        rotate=(-15, 15),
        shear=(-10, 10),
        p=0.5
    ),

    # 노이즈
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),

    # 저조도 시뮬레이션
    A.RandomShadow(
        shadow_roi=(0, 0.5, 1, 1),
        num_shadows_lower=1,
        num_shadows_upper=2,
        shadow_dimension=5,
        p=0.3
    ),
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels']
))

def augment_image(image_path, annotation_path, output_dir, num_augmentations=5):
    """이미지와 어노테이션을 증강"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # YOLO 형식 어노테이션 로드
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    bboxes = []
    class_labels = []
    for ann in annotations:
        parts = ann.strip().split()
        class_labels.append(int(parts[0]))
        bboxes.append([float(x) for x in parts[1:5]])

    for i in range(num_augmentations):
        transformed = transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )

        # 결과 저장
        aug_image = transformed['image']
        aug_bboxes = transformed['bboxes']
        aug_labels = transformed['class_labels']

        # 이미지 저장
        output_image_path = f"{output_dir}/aug_{i}_{os.path.basename(image_path)}"
        cv2.imwrite(output_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

        # 어노테이션 저장
        output_ann_path = output_image_path.replace('.png', '.txt').replace('.jpg', '.txt')
        with open(output_ann_path, 'w') as f:
            for label, bbox in zip(aug_labels, aug_bboxes):
                f.write(f"{label} {' '.join(map(str, bbox))}\n")
```

### 5.3 Domain Shift 대응 증강 전략

#### Phase 1: 기본 다양성 확보

```python
# 최소 증강 설정
basic_augmentation = {
    'hsv_h': 0.015,  # Hue
    'hsv_s': 0.7,    # Saturation
    'hsv_v': 0.4,    # Value (밝기)
    'degrees': 0.0,  # 회전
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
}
```

#### Phase 2: 강력한 일반화 (Domain Shift 방지)

```python
# 강화된 증강 설정
robust_augmentation = {
    'hsv_h': 0.05,   # Hue 범위 확대
    'hsv_s': 0.9,    # Saturation 범위 확대
    'hsv_v': 0.6,    # Value 범위 크게 확대
    'degrees': 15.0, # 회전 추가
    'translate': 0.15,
    'scale': 0.7,
    'shear': 5.0,
    'perspective': 0.001,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.15,
    'copy_paste': 0.1,
}
```

---

## 6. 데이터셋 버전 생성 및 내보내기

### 6.1 데이터셋 분할

#### 권장 비율

| 분할 | 비율 | 목적 |
|------|------|------|
| **Train** | 70% | 모델 학습 |
| **Validation** | 20% | 하이퍼파라미터 튜닝 |
| **Test** | 10% | 최종 성능 평가 |

#### Roboflow에서 분할 설정

```
1. Generate > Create New Version
2. Train/Valid/Test Split 설정
3. 권장: 70-20-10 또는 80-10-10
```

### 6.2 버전 생성 워크플로우

```
Version 1.0 (Baseline)
├── 전처리: Auto-Orient, Resize 640x640
├── 증강: 없음
└── 목적: 기준 성능 측정

Version 2.0 (Light Augmentation)
├── 전처리: Auto-Orient, Resize 640x640
├── 증강: 기본 조명/기하학적 변환
└── 목적: 일반화 테스트

Version 3.0 (Heavy Augmentation)
├── 전처리: Auto-Orient, Resize 640x640
├── 증강: 강력한 증강 (Domain Shift 대응)
└── 목적: 최종 배포용
```

### 6.3 내보내기 형식

#### YOLOv11 호환 형식

```yaml
# data.yaml 예시
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1  # 클래스 수
names: ['painting_object']  # 클래스 이름
```

#### 내보내기 방법

**방법 1: Roboflow 웹 인터페이스**
```
1. Generate Version 완료 후
2. Export > YOLOv8 PyTorch TXT 선택
3. Download ZIP 또는 Show Download Code
```

**방법 2: Python SDK**
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace").project("project-name")
version = project.version(1)

# YOLOv11 호환 형식으로 다운로드
dataset = version.download("yolov8")  # YOLOv11도 동일 형식 사용
print(f"Dataset downloaded to: {dataset.location}")
```

---

## 7. YOLO 모델 학습

### 7.1 환경 설정

#### 필수 패키지 설치

```bash
# Python 가상 환경 생성 (권장)
python -m venv yolo_env
source yolo_env/bin/activate  # Linux/Mac
# yolo_env\Scripts\activate  # Windows

# Ultralytics YOLO 설치
pip install ultralytics

# Roboflow SDK 설치
pip install roboflow

# 추가 의존성
pip install albumentations opencv-python-headless
```

#### GPU 확인

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### 7.2 학습 스크립트

#### 기본 학습

```python
from ultralytics import YOLO
from roboflow import Roboflow

# 데이터셋 다운로드
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace").project("fursys-painting-object-detection")
version = project.version(1)
dataset = version.download("yolov8")

# 모델 초기화
model = YOLO("yolo11s.pt")  # 사전 학습된 YOLOv11s

# 학습 실행
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU 사용
    workers=8,
    patience=20,  # Early Stopping
    save=True,
    plots=True,
    verbose=True,
)
```

#### 고급 학습 설정 (Domain Shift 대응)

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

# Domain Shift 대응 강화 학습
results = model.train(
    # 데이터 설정
    data="path/to/data.yaml",
    epochs=200,
    imgsz=640,
    batch=16,

    # 최적화 설정
    optimizer="AdamW",
    lr0=0.001,      # 초기 학습률
    lrf=0.01,       # 최종 학습률 비율
    momentum=0.937,
    weight_decay=0.0005,

    # 증강 설정 (강화)
    hsv_h=0.05,     # Hue 범위 확대
    hsv_s=0.9,      # Saturation 범위 확대
    hsv_v=0.6,      # Value 범위 확대
    degrees=15.0,   # 회전
    translate=0.15,
    scale=0.7,
    shear=5.0,
    perspective=0.001,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,

    # 정규화
    dropout=0.1,    # Dropout 추가

    # 기타 설정
    device=0,
    workers=8,
    patience=30,
    save=True,
    save_period=10,  # 10 에폭마다 저장
    plots=True,
    verbose=True,

    # 검증 설정
    val=True,
    split="val",
)
```

### 7.3 학습 모니터링

#### TensorBoard 사용

```bash
# TensorBoard 실행
tensorboard --logdir runs/detect/train

# 브라우저에서 http://localhost:6006 접속
```

#### 학습 메트릭

| 메트릭 | 설명 | 목표값 |
|--------|------|--------|
| **box_loss** | 바운딩 박스 손실 | < 0.05 |
| **cls_loss** | 분류 손실 | < 0.02 |
| **dfl_loss** | 분포 초점 손실 | < 1.0 |
| **mAP50** | IoU 0.5에서의 mAP | > 0.9 |
| **mAP50-95** | IoU 0.5-0.95 평균 mAP | > 0.7 |

### 7.4 학습 팁

#### 과적합 방지

```python
# 과적합 징후
# - train_loss가 계속 감소하지만 val_loss가 증가
# - mAP가 정체되거나 감소

# 해결책
training_config = {
    'patience': 20,        # Early Stopping
    'dropout': 0.1,        # Dropout
    'weight_decay': 0.001, # L2 정규화
    'mosaic': 1.0,         # Mosaic 증강
    'mixup': 0.2,          # MixUp 증강
}
```

#### 학습률 스케줄링

```python
# Cosine Annealing 사용 (Ultralytics 기본값)
# lr = lr0 * (1 + cos(pi * epoch / epochs)) / 2

# 또는 커스텀 스케줄러
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=100,
    steps_per_epoch=len(train_loader)
)
```

---

## 8. 모델 평가 및 검증

### 8.1 평가 메트릭

```python
from ultralytics import YOLO

# 학습된 모델 로드
model = YOLO("runs/detect/train/weights/best.pt")

# 검증 실행
metrics = model.val(
    data="path/to/data.yaml",
    split="test",
    batch=16,
    imgsz=640,
    conf=0.25,
    iou=0.6,
    save_json=True,
    plots=True,
)

# 결과 출력
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

### 8.2 Confusion Matrix 분석

```python
# Confusion Matrix 시각화
# runs/detect/train/confusion_matrix.png 확인

# 주요 분석 포인트
# 1. True Positive Rate (민감도)
# 2. False Positive Rate (오탐률)
# 3. 클래스별 혼동 패턴
```

### 8.3 추론 테스트

```python
from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

# 단일 이미지 추론
results = model.predict(
    source="test_image.jpg",
    conf=0.25,
    iou=0.45,
    save=True,
    show=True,
)

# 결과 분석
for result in results:
    boxes = result.boxes
    for box in boxes:
        print(f"Class: {result.names[int(box.cls)]}")
        print(f"Confidence: {float(box.conf):.4f}")
        print(f"Bbox: {box.xyxy[0].tolist()}")
```

### 8.4 다양한 환경에서 테스트

```python
import os
import cv2
from pathlib import Path

def evaluate_across_conditions(model, test_dirs, output_dir):
    """다양한 환경 조건에서 모델 평가"""

    results_summary = {}

    for condition, test_dir in test_dirs.items():
        print(f"\n{'='*50}")
        print(f"Testing on: {condition}")
        print(f"{'='*50}")

        # 해당 디렉토리의 모든 이미지에 대해 추론
        image_files = list(Path(test_dir).glob("*.png")) + list(Path(test_dir).glob("*.jpg"))

        total_detections = 0
        total_confidence = 0

        for img_path in image_files:
            results = model.predict(
                source=str(img_path),
                conf=0.25,
                verbose=False
            )

            for r in results:
                num_detections = len(r.boxes)
                total_detections += num_detections
                if num_detections > 0:
                    total_confidence += sum([float(b.conf) for b in r.boxes])

        avg_confidence = total_confidence / total_detections if total_detections > 0 else 0

        results_summary[condition] = {
            'total_images': len(image_files),
            'total_detections': total_detections,
            'avg_confidence': avg_confidence
        }

        print(f"Images: {len(image_files)}")
        print(f"Detections: {total_detections}")
        print(f"Avg Confidence: {avg_confidence:.4f}")

    return results_summary

# 사용 예시
test_conditions = {
    'standard_lighting': 'test_data/standard',
    'low_light': 'test_data/low_light',
    'bright_light': 'test_data/bright',
    'angle_45deg': 'test_data/angle_45',
    'mixed_conditions': 'test_data/mixed',
}

model = YOLO("runs/detect/train/weights/best.pt")
summary = evaluate_across_conditions(model, test_conditions, "evaluation_results")
```

---

## 9. Domain Shift 대응 전략

### 9.1 문제 진단

```python
def diagnose_domain_shift(model, source_dir, target_dir):
    """Domain Shift 문제 진단"""

    # 소스 도메인 (학습 환경) 성능
    source_results = model.val(data=source_dir, split="test")
    source_map = source_results.box.map50

    # 타겟 도메인 (새 환경) 성능
    target_results = model.val(data=target_dir, split="test")
    target_map = target_results.box.map50

    # Domain Gap 계산
    domain_gap = source_map - target_map

    print(f"Source Domain mAP50: {source_map:.4f}")
    print(f"Target Domain mAP50: {target_map:.4f}")
    print(f"Domain Gap: {domain_gap:.4f}")

    # 진단 결과
    if domain_gap > 0.3:
        print("심각한 Domain Shift 감지됨 - 재학습 필요")
    elif domain_gap > 0.15:
        print("중간 수준 Domain Shift - Fine-tuning 권장")
    elif domain_gap > 0.05:
        print("경미한 Domain Shift - 증강 강화로 해결 가능")
    else:
        print("Domain Shift 문제 없음")

    return domain_gap
```

### 9.2 해결 전략

#### 전략 1: 데이터 수집 확대

```
우선순위: ★★★★★ (가장 근본적인 해결책)

체크리스트:
□ 새 환경에서 추가 이미지 수집 (최소 500장)
□ 다양한 조명 조건 포함
□ 다양한 카메라 각도 포함
□ 혼합 데이터셋 구성
□ 재라벨링 및 품질 검증
```

#### 전략 2: Fine-Tuning

```python
from ultralytics import YOLO

# 기존 모델 로드
model = YOLO("path/to/original_model.pt")

# 새 환경 데이터로 Fine-tuning
model.train(
    data="new_environment_data.yaml",
    epochs=50,           # 적은 에폭
    lr0=0.0001,          # 낮은 학습률
    freeze=10,           # 초기 10개 레이어 동결
    imgsz=640,
    batch=16,
    patience=10,
)
```

#### 전략 3: 추론 시 적응

```python
class AdaptiveYOLODetector:
    """환경에 적응하는 YOLO 탐지기"""

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.conf_levels = [0.5, 0.35, 0.25, 0.15]

    def preprocess(self, image):
        """조명 정규화"""
        import cv2

        # CLAHE 적용
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def detect_adaptive(self, image, preprocess=True):
        """적응형 탐지"""
        if preprocess:
            image = self.preprocess(image)

        # 단계별 confidence 낮춤
        for conf in self.conf_levels:
            results = self.model.predict(
                source=image,
                conf=conf,
                verbose=False
            )

            if len(results[0].boxes) > 0:
                return results, conf

        return results, self.conf_levels[-1]
```

### 9.3 지속적 개선 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│              지속적 모델 개선 파이프라인                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   [운영 환경]                                                 │
│       │                                                      │
│       ▼                                                      │
│   ┌─────────────────┐                                        │
│   │ 탐지 결과 로깅   │ ← 저 confidence 결과 수집              │
│   └────────┬────────┘                                        │
│            │                                                  │
│            ▼                                                  │
│   ┌─────────────────┐                                        │
│   │ 수동 검증/라벨링 │ ← 정기적으로 (주 1회)                  │
│   └────────┬────────┘                                        │
│            │                                                  │
│            ▼                                                  │
│   ┌─────────────────┐                                        │
│   │ 데이터셋 업데이트│ ← Roboflow 버전 관리                   │
│   └────────┬────────┘                                        │
│            │                                                  │
│            ▼                                                  │
│   ┌─────────────────┐                                        │
│   │ 모델 재학습      │ ← 월 1회 또는 성능 저하 시             │
│   └────────┬────────┘                                        │
│            │                                                  │
│            ▼                                                  │
│   ┌─────────────────┐                                        │
│   │ A/B 테스트      │ ← 새 모델 vs 기존 모델                  │
│   └────────┬────────┘                                        │
│            │                                                  │
│            ▼                                                  │
│   [모델 배포 업데이트]                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. 배포 및 운영

### 10.1 모델 내보내기

#### TensorRT 변환 (Jetson 권장)

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# TensorRT 엔진으로 내보내기
model.export(
    format="engine",
    imgsz=640,
    half=True,      # FP16 사용 (Jetson 권장)
    device=0,
    simplify=True,
    workspace=4,    # GB
)
```

#### ONNX 변환

```python
model.export(
    format="onnx",
    imgsz=640,
    half=False,
    simplify=True,
    opset=12,
)
```

### 10.2 추론 최적화

#### Jetson Orin Nano Super 최적화

```python
from ultralytics import YOLO
import cv2

# TensorRT 모델 로드
model = YOLO("best.engine")

def optimized_inference(frame):
    """최적화된 추론"""
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        half=True,       # FP16
        device=0,
        verbose=False,
    )
    return results

# 배치 추론 (더 높은 처리량)
def batch_inference(frames, batch_size=4):
    """배치 처리"""
    results = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        batch_results = model.predict(
            source=batch,
            imgsz=640,
            conf=0.25,
            batch=len(batch),
        )
        results.extend(batch_results)
    return results
```

### 10.3 실시간 스트리밍

```python
import cv2
from ultralytics import YOLO

model = YOLO("best.engine")

def realtime_detection(source=0):
    """실시간 탐지"""
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 추론
        results = model.predict(
            source=frame,
            conf=0.25,
            verbose=False
        )

        # 시각화
        annotated = results[0].plot()

        # FPS 표시
        # ...

        cv2.imshow("Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

### 10.4 Roboflow 배포

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace").project("project-name")

# 모델 업로드
version = project.version(1)
version.deploy(
    model_type="yolov8",
    model_path="runs/detect/train/weights/best.pt"
)

# 호스팅 API 사용
model = version.model

# 이미지 추론
prediction = model.predict("test_image.jpg", confidence=40, overlap=30)
prediction.save("prediction.jpg")
```

---

## 11. 참고 자료

### 11.1 공식 문서

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [Roboflow Annotate Guide](https://roboflow.com/annotate)

### 11.2 튜토리얼

- [How to Train YOLOv11 on Custom Dataset](https://blog.roboflow.com/yolov11-how-to-train-custom-data/)
- [YOLOv12 Training Guide](https://blog.roboflow.com/train-yolov12-model/)
- [How to Label Data for YOLO](https://roboflow.com/how-to-label/yolov8)

### 11.3 Best Practices

- [Tips for Best YOLO Training Results](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)
- [Data Augmentation Guide](https://docs.ultralytics.com/guides/yolo-data-augmentation/)
- [Industrial Object Detection Best Practices](https://blog.roboflow.com/tips-for-how-to-label-images/)

### 11.4 Domain Adaptation

- [SSDA-YOLO GitHub](https://github.com/hnuzhy/SSDA-YOLO)
- [SF-YOLO GitHub](https://github.com/vs-cv/sf-yolo)

### 11.5 관련 프로젝트 문서

- [yolo_domain_shift_analysis_report.md](./251229/yolo_domain_shift_analysis_report.md)
- [ai_based_velocity_angle_measurement_research.md](./251218/ai_based_velocity_angle_measurement_research.md)

---

## 부록: 체크리스트 요약

### A. 데이터 준비 체크리스트

```
□ 최소 1,500장 이상의 이미지 수집
□ 5가지 이상 조명 조건 포함
□ 5가지 이상 카메라 각도 포함
□ 다양한 배경 포함
□ Roboflow 프로젝트 생성
□ 이미지 업로드 완료
```

### B. 라벨링 체크리스트

```
□ 모든 이미지 라벨링 완료
□ Tight bounding box 적용
□ 일관된 라벨링 기준 적용
□ AI 보조 도구 활용
□ 품질 검증 완료
```

### C. 학습 체크리스트

```
□ 환경 설정 완료 (GPU, 패키지)
□ 데이터셋 분할 (70-20-10)
□ 버전 관리 설정
□ 증강 파라미터 설정
□ 학습 실행 및 모니터링
□ 과적합 확인
```

### D. 평가 체크리스트

```
□ mAP50 > 0.9 달성
□ 다양한 환경에서 테스트
□ Domain Shift 진단
□ Confusion Matrix 분석
□ 추론 속도 확인 (30+ FPS)
```

### E. 배포 체크리스트

```
□ TensorRT 변환 완료
□ Jetson 테스트 완료
□ 실시간 추론 검증
□ 지속적 개선 파이프라인 구축
```

---

*작성: 2025-12-31*
*환경: NVIDIA Jetson Orin Nano Super + Intel RealSense D455*
*목적: YOLO 커스텀 모델 학습 가이드 (Domain Shift 해결)*
