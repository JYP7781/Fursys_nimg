# YOLO 학습 데이터 준비 및 라벨링 가이드

**작성일**: 2025-12-31
**목적**: 현재 보유 데이터를 기반으로 YOLO 커스텀 모델 학습을 위한 데이터 준비 및 라벨링 전략 수립
**관련 문서**: [yolo_roboflow_training_complete_guide.md](./yolo_roboflow_training_complete_guide.md)

---

## 목차

1. [현재 데이터 현황 분석](#1-현재-데이터-현황-분석)
2. [데이터 구조 상세 분석](#2-데이터-구조-상세-분석)
3. [학습 데이터 선별 전략](#3-학습-데이터-선별-전략)
4. [라벨링 도구 비교 및 선택](#4-라벨링-도구-비교-및-선택)
5. [라벨링 실행 가이드](#5-라벨링-실행-가이드)
6. [데이터 증강 전략](#6-데이터-증강-전략)
7. [학습 데이터셋 구성](#7-학습-데이터셋-구성)
8. [권장 실행 계획](#8-권장-실행-계획)
9. [참고 자료](#9-참고-자료)

---

## 1. 현재 데이터 현황 분석

### 1.1 데이터 저장 위치 및 개요

현재 프로젝트에는 3개의 주요 데이터 소스가 있습니다:

| 데이터 소스 | 위치 | 데이터 유형 | 특징 |
|------------|------|------------|------|
| **Output 1** | `/root/fursys_imgprosessing_ws/20251208_155531_output/` | 비디오 프레임 추출 | 고해상도 (1280x720) |
| **Output 2** | `/root/fursys_imgprosessing_ws/20251208_161246_output/` | 비디오 프레임 추출 | 고해상도 (1280x720) |
| **Extraction** | `/root/fursys_img_251229/extraction/` | 각도별 촬영 이미지 | 다양한 각도 (640x480) |

### 1.2 총 데이터 통계

```
┌─────────────────────────────────────────────────────────────────────┐
│                        총 데이터 현황                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Output 폴더 데이터:                                                  │
│  ├── 20251208_155531_output                                          │
│  │   ├── Color 이미지: 12,395장 (1280x720, RGB)                      │
│  │   ├── Depth 이미지: 12,395장 (848x480, 16-bit)                    │
│  │   └── 비디오: 1개 (640MB MP4)                                     │
│  │                                                                   │
│  └── 20251208_161246_output                                          │
│      ├── Color 이미지: 11,187장 (1280x720, RGB)                      │
│      ├── Depth 이미지: 11,187장 (848x480, 16-bit)                    │
│      └── 비디오: 1개 (530MB MP4)                                     │
│                                                                      │
│  Extraction 폴더 데이터 (각도별):                                     │
│  ├── 20251229_093820_front     : 242장 (정면)                        │
│  ├── 20251229_094115_45        : 166장 (45도)                        │
│  ├── 20251229_094410_90        : 136장 (90도)                        │
│  ├── 20251229_094733_135       : 214장 (135도)                       │
│  ├── 20251229_094948_180       : 290장 (180도)                       │
│  ├── 20251229_095222_225       : 262장 (225도)                       │
│  ├── 20251229_095322_270       : 246장 (270도)                       │
│  ├── 20251229_150727_315       : 154장 (315도)                       │
│  ├── 20251229_151239_topfront  : 276장 (상단 정면)                   │
│  ├── 20251229_151338_top90     : 344장 (상단 90도)                   │
│  ├── 20251229_151622_top270    : 210장 (상단 270도)                  │
│  ├── 20251229_152005_bottomfront: 208장 (하단 정면)                  │
│  ├── 20251229_152111_bottom90  : 200장 (하단 90도)                   │
│  ├── 20251229_152156_bottom270 : 308장 (하단 270도)                  │
│  └── 20251229_154835_test      : 59,012장 (테스트용)                 │
│                                                                      │
│  ═══════════════════════════════════════════════════════════════     │
│  총 Color 이미지: 약 85,000장+                                       │
│  학습에 활용 가능한 이미지: 충분                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 데이터 구조 상세 분석

### 2.1 Output 폴더 데이터 (20251208_*)

#### 이미지 특성

| 속성 | Color 이미지 | Depth 이미지 |
|------|-------------|-------------|
| **해상도** | 1280 x 720 | 848 x 480 |
| **색상 모드** | RGB (24-bit) | I;16 (16-bit grayscale) |
| **파일 형식** | PNG | PNG |
| **파일 크기** | ~1.4MB | ~0.5MB |
| **명명 규칙** | `color_XXXXXX.png` | `depth_XXXXXX.png` |

#### 촬영 환경 분석

**20251208_155531_output** (이미지 샘플 분석):
- 환경: 도장 작업장, 밝은 조명
- 객체: 컨베이어 시스템, 도장 프레임 구조물
- 배경: 흰 벽, 창문, 산업 장비
- 특징: 고해상도, 넓은 시야각

**20251208_161246_output**:
- 유사한 환경, 약간 다른 조명 조건
- 동일한 도장 작업장 촬영

### 2.2 Extraction 폴더 데이터 (각도별)

#### 이미지 특성

| 속성 | RGB 이미지 | Depth 이미지 |
|------|-----------|-------------|
| **해상도** | 640 x 480 | 640 x 480 |
| **색상 모드** | RGB | 16-bit |
| **파일 형식** | PNG | PNG |
| **명명 규칙** | `rgb_Color_[timestamp].png` | `depth_[timestamp].png` |

#### 각도별 촬영 데이터 분포

```
수평 각도 (8방향):
┌───────────────────────────────────────────────────────────┐
│                         315° (154장)                       │
│                              │                             │
│              270° (246장)────┼────90° (136장)              │
│                              │                             │
│              225° (262장)────┼────135° (214장)             │
│                              │                             │
│              180° (290장)────┼────45° (166장)              │
│                              │                             │
│                         front (242장)                      │
└───────────────────────────────────────────────────────────┘

수직 각도 (상/중/하):
┌───────────────────────────────────────────────────────────┐
│  상단 (Top):                                               │
│  ├── topfront  : 276장                                     │
│  ├── top90     : 344장                                     │
│  └── top270    : 210장                                     │
│                                                            │
│  중단 (Middle): 수평 8방향 데이터                           │
│                                                            │
│  하단 (Bottom):                                            │
│  ├── bottomfront : 208장                                   │
│  ├── bottom90    : 200장                                   │
│  └── bottom270   : 308장                                   │
└───────────────────────────────────────────────────────────┘
```

### 2.3 데이터 품질 평가

#### 장점

| 항목 | 평가 | 설명 |
|------|------|------|
| **데이터 양** | ★★★★★ | 85,000장+ 이미지로 충분한 학습 데이터 |
| **각도 다양성** | ★★★★★ | 8개 수평 방향 + 3개 수직 레벨 |
| **해상도** | ★★★★☆ | 640x480 ~ 1280x720 다양한 해상도 |
| **Depth 정보** | ★★★★★ | RGB-D 데이터로 추가 활용 가능 |

#### 개선 필요 사항

| 항목 | 평가 | 설명 |
|------|------|------|
| **조명 다양성** | ★★★☆☆ | 유사한 조명 조건, 다양화 필요 |
| **배경 다양성** | ★★★☆☆ | 동일 작업장, 배경 단일 |
| **라벨 데이터** | ★☆☆☆☆ | 라벨링 되지 않은 원본 데이터 |

---

## 3. 학습 데이터 선별 전략

### 3.1 Ultralytics 권장 기준

```yaml
# 최적 학습을 위한 권장 데이터 양 (Ultralytics 공식)
recommended:
  images_per_class: 1500+      # 클래스당 최소 이미지 수
  instances_per_class: 10000+  # 클래스당 최소 객체 인스턴스 수
  background_images: 0-10%     # False Positive 감소용 배경 이미지
```

### 3.2 데이터 선별 우선순위

#### Phase 1: 핵심 학습 데이터 (필수)

**각도별 Extraction 데이터 전체 활용** (약 3,256장):
- 다양한 각도에서 촬영된 도장물 이미지
- Domain Shift 문제 해결에 핵심
- 모든 각도 데이터 라벨링 권장

```python
# 각도별 데이터 선별 스크립트 예시
extraction_data = {
    "front": 242,      # 정면
    "45": 166,         # 45도
    "90": 136,         # 90도
    "135": 214,        # 135도
    "180": 290,        # 180도
    "225": 262,        # 225도
    "270": 246,        # 270도
    "315": 154,        # 315도
    "topfront": 276,   # 상단 정면
    "top90": 344,      # 상단 90도
    "top270": 210,     # 상단 270도
    "bottomfront": 208,# 하단 정면
    "bottom90": 200,   # 하단 90도
    "bottom270": 308,  # 하단 270도
}
total_extraction = sum(extraction_data.values())  # 3,256장
```

#### Phase 2: 보조 학습 데이터 (선택)

**Output 폴더에서 샘플링** (권장: 2,000-5,000장):
- 12,395장 + 11,187장 = 23,582장 중 선별
- 프레임 간격 샘플링 (10-20 프레임마다 1장)
- 도장물이 명확히 보이는 프레임 선별

```python
# 프레임 샘플링 예시
sampling_strategy = {
    "method": "interval",
    "interval": 15,  # 15 프레임마다 1장
    "output_1_samples": 12395 // 15,  # ~826장
    "output_2_samples": 11187 // 15,  # ~746장
    "total_samples": 1572  # 추가 데이터
}
```

### 3.3 최종 데이터셋 목표

```
┌─────────────────────────────────────────────────────────────────────┐
│                    최종 데이터셋 구성 목표                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  핵심 데이터 (Extraction - 전체 라벨링):                              │
│  └── 3,256장 (14개 각도 × ~233장 평균)                               │
│                                                                      │
│  보조 데이터 (Output - 샘플링 라벨링):                                │
│  └── 1,500 ~ 2,000장                                                 │
│                                                                      │
│  배경 이미지 (객체 없는 프레임):                                      │
│  └── 200 ~ 300장 (전체의 5~10%)                                      │
│                                                                      │
│  ═══════════════════════════════════════════════════════════════     │
│  총 라벨링 대상: 약 5,000 ~ 5,500장                                   │
│  데이터 증강 후: 약 15,000 ~ 20,000장                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 라벨링 도구 비교 및 선택

### 4.1 주요 라벨링 도구 비교

| 기능 | Roboflow | CVAT | LabelImg |
|------|----------|------|----------|
| **유형** | 클라우드 (상업용) | 오픈소스 (웹/자체호스팅) | 오픈소스 (데스크탑) |
| **YOLO 내보내기** | ✅ 지원 | ✅ 지원 | ✅ 지원 |
| **AI 보조 라벨링** | ✅ Auto Label, SAM | ✅ 제한적 | ❌ 없음 |
| **팀 협업** | ✅ 우수 | ✅ 지원 | ❌ 미지원 |
| **비용** | 무료 티어 제한 | 무료 | 무료 |
| **학습 곡선** | 쉬움 | 중간 | 쉬움 |
| **배치 처리** | ✅ 우수 | ✅ 지원 | ❌ 미지원 |

### 4.2 권장 도구: Roboflow

#### 선택 이유

1. **AI 보조 라벨링**: Auto Label, Smart Polygon, Box Prompting으로 라벨링 시간 90%+ 단축
2. **SAM 3 통합**: 2025년 11월 업데이트된 Segment Anything Model 3 지원
3. **일관된 워크플로우**: 라벨링 → 증강 → 학습 → 배포까지 단일 플랫폼
4. **YOLO 형식 직접 내보내기**: YOLOv8/v11 형식으로 바로 내보내기 가능

#### Roboflow 무료 티어 제한

```yaml
free_tier_limits:
  projects: 3개
  images_per_project: 10,000장
  team_members: 3명
  model_training: 제한적
  api_calls: 1,000회/월
```

### 4.3 대안 도구: CVAT (자체 호스팅)

#### 장점
- 완전 무료, 이미지 수 제한 없음
- 자체 서버 호스팅으로 데이터 보안
- YOLO 형식 내보내기 지원

#### 설치 방법

```bash
# Docker를 사용한 CVAT 설치
git clone https://github.com/opencv/cvat
cd cvat
docker-compose up -d

# 웹 접속: http://localhost:8080
```

---

## 5. 라벨링 실행 가이드

### 5.1 라벨링 대상 클래스 정의

#### 단일 클래스 접근법 (권장)

```yaml
classes:
  - name: "painting_object"
    description: "컨베이어 위의 도장물 객체"
    id: 0
```

#### 다중 클래스 접근법 (선택사항)

```yaml
classes:
  - name: "painting_object"
    description: "도장물 메인 객체"
    id: 0
  - name: "frame"
    description: "도장 프레임/지지대"
    id: 1
  - name: "hook"
    description: "걸이 부품"
    id: 2
```

### 5.2 Roboflow 라벨링 워크플로우

#### Step 1: 프로젝트 생성

```
1. https://roboflow.com 접속 → 계정 생성/로그인
2. "Create New Project" 클릭
3. 설정:
   - Project Name: fursys-painting-detection
   - Project Type: Object Detection
   - Annotation Group: painting_object
```

#### Step 2: 이미지 업로드

**방법 A: 웹 인터페이스**
```
1. 프로젝트 대시보드 → "Upload" 클릭
2. 폴더 또는 이미지 파일 드래그 앤 드롭
3. 업로드 완료 대기
```

**방법 B: Python SDK**
```python
from roboflow import Roboflow
import os
from pathlib import Path

# Roboflow 초기화
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("fursys-painting-detection")

# 이미지 업로드 함수
def upload_images(image_dir, project, limit=None):
    """디렉토리의 이미지를 Roboflow에 업로드"""
    image_files = list(Path(image_dir).glob("*.png"))

    if limit:
        image_files = image_files[:limit]

    for i, img_path in enumerate(image_files):
        project.upload(str(img_path))
        if (i + 1) % 100 == 0:
            print(f"Uploaded {i + 1}/{len(image_files)}")

    print(f"Total uploaded: {len(image_files)}")

# Extraction 폴더 업로드
extraction_base = "/root/fursys_img_251229/extraction"
folders = [
    "20251229_093820_front",
    "20251229_094115_45",
    "20251229_094410_90",
    # ... 나머지 폴더
]

for folder in folders:
    rgb_path = os.path.join(extraction_base, folder, "rgb")
    upload_images(rgb_path, project)
```

#### Step 3: Auto Label 활용

```
1. 업로드된 이미지 배치 선택
2. "Auto Label" 버튼 클릭
3. 프롬프트 입력:
   - "metal object" 또는 "painted metal panel"
   - "industrial part on conveyor"
4. Confidence threshold 조정 (권장: 0.3-0.5)
5. 결과 미리보기 및 적용
```

#### Step 4: 수동 검증 및 수정

```
1. Annotate 탭에서 이미지 하나씩 검토
2. 단축키:
   - B: 바운딩 박스 도구
   - V: 선택 도구
   - Delete: 선택 항목 삭제
   - Enter: 저장 후 다음 이미지
3. 바운딩 박스 규칙:
   - 객체에 딱 맞게 (tight bounding box)
   - 그림자 제외
   - 반사광은 포함
```

### 5.3 라벨링 품질 기준

#### 바운딩 박스 규칙

```
┌───────────────────────────────────────────────────────────────────┐
│                     올바른 바운딩 박스                              │
│                                                                    │
│   ✅ 좋은 예시:                    ❌ 나쁜 예시:                    │
│   ┌─────────────┐                 ┌─────────────────┐              │
│   │ ┌─────────┐ │                 │                 │              │
│   │ │  객체   │ │  ← 딱 맞음     │   ┌───────┐     │  ← 여백 과다  │
│   │ └─────────┘ │                 │   │ 객체  │     │              │
│   └─────────────┘                 │   └───────┘     │              │
│                                   └─────────────────┘              │
│                                                                    │
│   ❌ 나쁜 예시:                                                     │
│   ┌─────────┐                                                      │
│   │  객 체 ──┼──  ← 객체 잘림                                      │
│   └─────────┘                                                      │
└───────────────────────────────────────────────────────────────────┘
```

#### 체크리스트

```
□ 모든 도장물 객체가 라벨링 되었는가?
□ 바운딩 박스가 객체 경계에 딱 맞는가?
□ 그림자가 포함되지 않았는가?
□ 부분적으로 보이는 객체도 처리했는가? (50% 이상 보이면 라벨링)
□ 이미지 경계에 걸친 객체는 보이는 부분만 라벨링했는가?
□ 중복 라벨링이 없는가?
```

---

## 6. 데이터 증강 전략

### 6.1 Roboflow 내장 증강 설정

#### 전처리 (Preprocessing)

```yaml
preprocessing:
  auto_orient: true        # EXIF 기반 자동 회전
  resize: 640              # YOLO 입력 크기 맞춤
  static_crop: null        # 특정 영역 크롭 불필요
  grayscale: false         # 컬러 유지
```

#### 증강 (Augmentation) - Domain Shift 대응

```yaml
# 조명 변화 대응 (필수 - 가장 중요)
lighting_augmentation:
  brightness:
    enabled: true
    range: [-30%, +30%]    # 넓은 범위로 다양한 조명 시뮬레이션
  exposure:
    enabled: true
    range: [-25%, +25%]
  saturation:
    enabled: true
    range: [-35%, +35%]    # 색상 채도 변화
  hue:
    enabled: true
    range: [-20°, +20°]    # 색조 변화

# 기하학적 변환 (필수)
geometric_augmentation:
  rotation:
    enabled: true
    range: [-15°, +15°]    # 약간의 회전
  flip:
    horizontal: true
    vertical: false        # 수직 뒤집기는 비현실적
  shear:
    enabled: true
    range: [-10°, +10°]
  crop:
    enabled: true
    min_zoom: 0%
    max_zoom: 15%

# 노이즈 및 블러
noise_augmentation:
  blur:
    enabled: true
    max_pixels: 2.0
  noise:
    enabled: true
    max_percent: 2%
  cutout:
    enabled: true
    count: 2
    size: 5%
```

### 6.2 YOLO 학습 시 내장 증강 파라미터

```python
# Ultralytics YOLO 학습 시 증강 파라미터 (권장)
augmentation_params = {
    # HSV 색공간 조정 (조명 변화 대응 핵심)
    'hsv_h': 0.05,     # Hue 변화 범위 (기본 0.015보다 확대)
    'hsv_s': 0.9,      # Saturation 변화 범위
    'hsv_v': 0.6,      # Value(밝기) 변화 범위 (가장 중요)

    # 기하학적 변환
    'degrees': 15.0,   # 회전 범위
    'translate': 0.15, # 이동 범위
    'scale': 0.6,      # 스케일 변화
    'shear': 5.0,      # 전단 변환
    'perspective': 0.001,

    # 뒤집기
    'flipud': 0.0,     # 상하 뒤집기 (사용 안함)
    'fliplr': 0.5,     # 좌우 뒤집기

    # 고급 증강
    'mosaic': 1.0,     # Mosaic 증강
    'mixup': 0.15,     # MixUp 증강
    'copy_paste': 0.1, # Copy-Paste 증강
}
```

### 6.3 증강 후 예상 데이터셋 크기

```
원본 데이터: ~5,000장
증강 배수: 3x (Roboflow 기본)
───────────────────────
증강 후: ~15,000장

추가 YOLO 런타임 증강으로 실제 학습 시:
효과적인 데이터: ~45,000+ 샘플
```

---

## 7. 학습 데이터셋 구성

### 7.1 데이터셋 분할

```yaml
dataset_split:
  train: 70%      # 학습용
  validation: 20% # 검증용 (하이퍼파라미터 튜닝)
  test: 10%       # 테스트용 (최종 성능 평가)
```

### 7.2 YOLO 데이터셋 구조

```
dataset/
├── data.yaml           # 데이터셋 설정 파일
├── train/
│   ├── images/         # 학습 이미지
│   │   ├── img001.png
│   │   ├── img002.png
│   │   └── ...
│   └── labels/         # 학습 라벨 (YOLO 형식)
│       ├── img001.txt
│       ├── img002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 7.3 data.yaml 예시

```yaml
# data.yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1  # 클래스 수
names: ['painting_object']  # 클래스 이름

# 추가 메타데이터 (선택)
# roboflow:
#   workspace: your-workspace
#   project: fursys-painting-detection
#   version: 1
```

### 7.4 YOLO 라벨 형식

```
# 각 이미지에 대응하는 .txt 파일
# 형식: <class_id> <x_center> <y_center> <width> <height>
# 모든 좌표는 0-1 사이로 정규화

# 예시 (img001.txt):
0 0.5234 0.4521 0.2156 0.3872
0 0.7891 0.6234 0.1523 0.2845
```

---

## 8. 권장 실행 계획

### 8.1 단계별 실행 계획

```
┌─────────────────────────────────────────────────────────────────────┐
│                    YOLO 학습 데이터 준비 로드맵                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: 데이터 준비                                                │
│  ├── Step 1: Extraction 폴더 이미지 정리 및 선별                     │
│  ├── Step 2: Output 폴더에서 대표 프레임 샘플링                      │
│  └── Step 3: 배경 이미지 (객체 없는 프레임) 수집                     │
│                                                                      │
│  Phase 2: 라벨링                                                     │
│  ├── Step 4: Roboflow 프로젝트 생성 및 이미지 업로드                 │
│  ├── Step 5: Auto Label로 초기 라벨링                                │
│  ├── Step 6: 수동 검증 및 수정 (품질 보증)                           │
│  └── Step 7: 라벨링 완료 검토                                        │
│                                                                      │
│  Phase 3: 데이터셋 생성                                              │
│  ├── Step 8: 데이터 증강 설정                                        │
│  ├── Step 9: 데이터셋 버전 생성                                      │
│  └── Step 10: YOLOv8/v11 형식으로 내보내기                           │
│                                                                      │
│  Phase 4: 학습                                                       │
│  ├── Step 11: 환경 설정 (GPU, 패키지)                                │
│  ├── Step 12: 기본 학습 실행                                         │
│  ├── Step 13: 학습 결과 평가                                         │
│  └── Step 14: 하이퍼파라미터 튜닝 (필요시)                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 데이터 준비 스크립트

```python
#!/usr/bin/env python3
"""
YOLO 학습 데이터 준비 스크립트
"""

import os
import shutil
import random
from pathlib import Path

# 경로 설정
WORKSPACE = "/root/fursys_imgprosessing_ws"
EXTRACTION_BASE = "/root/fursys_img_251229/extraction"
OUTPUT_DIR = f"{WORKSPACE}/training_data"

# 1. Extraction 폴더 목록
EXTRACTION_FOLDERS = [
    "20251229_093820_front",
    "20251229_094115_45",
    "20251229_094410_90",
    "20251229_094733_135",
    "20251229_094948_180",
    "20251229_095222_225",
    "20251229_095322_270",
    "20251229_150727_315",
    "20251229_151239_topfront",
    "20251229_151338_top90",
    "20251229_151622_top270",
    "20251229_152005_bottomfront",
    "20251229_152111_bottom90",
    "20251229_152156_bottom270",
]

def prepare_extraction_data():
    """Extraction 폴더의 모든 RGB 이미지를 수집"""
    os.makedirs(f"{OUTPUT_DIR}/extraction", exist_ok=True)

    total_copied = 0
    for folder in EXTRACTION_FOLDERS:
        rgb_path = Path(EXTRACTION_BASE) / folder / "rgb"
        if rgb_path.exists():
            images = list(rgb_path.glob("*.png"))
            for img in images:
                # 폴더 이름을 접두어로 추가하여 고유성 보장
                new_name = f"{folder}_{img.name}"
                shutil.copy(img, f"{OUTPUT_DIR}/extraction/{new_name}")
                total_copied += 1

    print(f"Extraction 이미지 복사 완료: {total_copied}장")
    return total_copied

def sample_output_data(sample_interval=15, max_samples=2000):
    """Output 폴더에서 일정 간격으로 이미지 샘플링"""
    os.makedirs(f"{OUTPUT_DIR}/output_samples", exist_ok=True)

    output_folders = [
        f"{WORKSPACE}/20251208_155531_output",
        f"{WORKSPACE}/20251208_161246_output",
    ]

    total_sampled = 0
    for folder in output_folders:
        folder_name = os.path.basename(folder)
        color_images = sorted(Path(folder).glob("color_*.png"))

        # 간격 샘플링
        sampled = color_images[::sample_interval]

        for img in sampled:
            if total_sampled >= max_samples:
                break
            new_name = f"{folder_name}_{img.name}"
            shutil.copy(img, f"{OUTPUT_DIR}/output_samples/{new_name}")
            total_sampled += 1

        if total_sampled >= max_samples:
            break

    print(f"Output 샘플링 완료: {total_sampled}장")
    return total_sampled

def create_data_summary():
    """데이터 요약 생성"""
    extraction_count = len(list(Path(f"{OUTPUT_DIR}/extraction").glob("*.png")))
    output_count = len(list(Path(f"{OUTPUT_DIR}/output_samples").glob("*.png")))

    summary = f"""
    ═══════════════════════════════════════════════════════
    학습 데이터 준비 완료 요약
    ═══════════════════════════════════════════════════════

    Extraction 이미지: {extraction_count}장
    Output 샘플 이미지: {output_count}장
    ───────────────────────────────────────────────────────
    총 이미지: {extraction_count + output_count}장

    다음 단계:
    1. {OUTPUT_DIR} 폴더의 이미지를 Roboflow에 업로드
    2. Auto Label로 초기 라벨링 수행
    3. 수동 검증 및 수정
    4. 데이터셋 버전 생성 및 내보내기
    ═══════════════════════════════════════════════════════
    """
    print(summary)

    with open(f"{OUTPUT_DIR}/data_summary.txt", "w") as f:
        f.write(summary)

if __name__ == "__main__":
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 데이터 준비 실행
    prepare_extraction_data()
    sample_output_data()
    create_data_summary()
```

### 8.3 학습 실행 스크립트

```python
#!/usr/bin/env python3
"""
YOLO 모델 학습 스크립트
"""

from ultralytics import YOLO

def train_yolo_model(data_yaml_path, epochs=100):
    """YOLO 모델 학습"""

    # 모델 초기화 (YOLOv11s 권장)
    model = YOLO("yolo11s.pt")

    # 학습 실행
    results = model.train(
        # 데이터 설정
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16,

        # 최적화 설정
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,

        # 증강 설정 (Domain Shift 대응 강화)
        hsv_h=0.05,
        hsv_s=0.9,
        hsv_v=0.6,       # 밝기 변화 확대
        degrees=15.0,
        translate=0.15,
        scale=0.6,
        shear=5.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,

        # 기타 설정
        device=0,        # GPU 사용
        workers=8,
        patience=30,     # Early Stopping
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    return results

if __name__ == "__main__":
    # Roboflow에서 다운로드한 데이터셋 경로
    data_yaml = "path/to/downloaded/dataset/data.yaml"

    # 학습 실행
    results = train_yolo_model(data_yaml, epochs=150)

    print("학습 완료!")
    print(f"Best model saved to: runs/detect/train/weights/best.pt")
```

---

## 9. 참고 자료

### 9.1 공식 문서

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/) - YOLO 모델 공식 문서
- [Roboflow Documentation](https://docs.roboflow.com/) - Roboflow 플랫폼 가이드
- [Roboflow Auto Label](https://docs.roboflow.com/annotate/ai-labeling/automated-annotation-with-autodistill) - AI 자동 라벨링 가이드
- [YOLO Data Augmentation Guide](https://docs.ultralytics.com/guides/yolo-data-augmentation/) - 데이터 증강 상세 가이드

### 9.2 라벨링 도구 비교

- [Best Data Annotation Platforms 2025](https://blog.roboflow.com/data-annotation-platforms/) - 2025년 최신 라벨링 플랫폼 비교
- [CVAT vs Roboflow](https://averroes.ai/blog/cvat-vs-roboflow-vs-visionrepo) - CVAT, Roboflow 상세 비교
- [CVAT Official](https://www.cvat.ai/) - CVAT 공식 사이트

### 9.3 학습 최적화

- [Tips for Best YOLOv5 Training Results](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/) - 학습 최적화 팁
- [YOLOv8 Best Practices](https://medium.com/internet-of-technology/yolov8-best-practices-for-training-cdb6eacf7e4f) - 학습 모범 사례
- [Albumentations Integration](https://docs.ultralytics.com/integrations/albumentations/) - 고급 증강 라이브러리 연동

### 9.4 산업용 객체 탐지

- [Industrial Object Detection with YOLOv8](https://arxiv.org/html/2503.10356) - 공장 환경 객체 탐지 연구
- [Roboflow Universe](https://universe.roboflow.com/search?q=class:yolo) - YOLO 데이터셋 검색

### 9.5 프로젝트 관련 문서

- [YOLO Domain Shift 분석 리포트](./251229/yolo_domain_shift_analysis_report.md) - Domain Shift 문제 분석
- [YOLO Roboflow Training Complete Guide](./yolo_roboflow_training_complete_guide.md) - 전체 학습 가이드

---

## 부록: 빠른 참조 체크리스트

### A. 데이터 준비

```
□ Extraction 폴더 전체 이미지 수집 (~3,256장)
□ Output 폴더 샘플링 (~1,500-2,000장)
□ 배경 이미지 수집 (~200-300장)
□ 총 이미지 수 확인 (~5,000장+)
```

### B. 라벨링

```
□ Roboflow 프로젝트 생성
□ 이미지 업로드 완료
□ Auto Label 실행
□ 수동 검증 완료 (모든 이미지)
□ 품질 검토 완료
```

### C. 데이터셋 생성

```
□ 전처리 설정 (Auto-Orient, Resize 640)
□ 증강 설정 (Brightness, HSV, Geometric)
□ Train/Valid/Test 분할 (70/20/10)
□ 버전 생성
□ YOLOv8 형식 내보내기
```

### D. 학습

```
□ 환경 설정 완료 (GPU, ultralytics 패키지)
□ data.yaml 경로 확인
□ 학습 실행
□ 학습 모니터링 (TensorBoard)
□ 결과 평가 (mAP50 > 0.9 목표)
```

---

*작성: 2025-12-31*
*환경: NVIDIA Jetson Orin Nano Super + Intel RealSense D455*
*목적: YOLO 커스텀 모델 학습을 위한 데이터 준비 가이드*
