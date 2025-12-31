# nimg_v2 환경 구축 및 실행 가이드

## 목차

1. [시스템 요구사항](#1-시스템-요구사항)
2. [환경 구축](#2-환경-구축)
3. [설치](#3-설치)
4. [실행 방법](#4-실행-방법)
5. [테스트](#5-테스트)
6. [문제 해결](#6-문제-해결)

---

## 1. 시스템 요구사항

### 하드웨어
- **CPU**: Intel Core i5 이상 또는 동급
- **RAM**: 8GB 이상 (16GB 권장)
- **GPU**: NVIDIA GPU with CUDA support (권장, 없어도 CPU로 실행 가능)
- **Storage**: 10GB 이상 (모델 및 데이터용)

### 소프트웨어
- **OS**: Ubuntu 20.04/22.04, Windows 10/11
- **Python**: 3.8 ~ 3.11
- **CUDA**: 11.8 이상 (GPU 사용 시)

---

## 2. 환경 구축

### 2.1 Python 가상환경 생성

```bash
# Conda 사용 (권장)
conda create -n nimg_v2 python=3.10
conda activate nimg_v2

# 또는 venv 사용
python -m venv nimg_v2_env
source nimg_v2_env/bin/activate  # Linux/Mac
# nimg_v2_env\Scripts\activate   # Windows
```

### 2.2 PyTorch 설치

GPU 버전 (CUDA 11.8):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CPU 버전:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2.3 Open3D 설치

```bash
pip install open3d
```

> **참고**: Open3D는 시스템에 따라 설치 문제가 있을 수 있습니다.
> 문제 발생 시 conda로 설치하세요:
> ```bash
> conda install -c open3d-admin open3d
> ```

---

## 3. 설치

### 3.1 의존성 설치

```bash
cd /root/fursys_imgprosessing_ws/src/nimg_v2
pip install -r requirements.txt
```

### 3.2 패키지 설치 (개발 모드)

```bash
pip install -e .
```

### 3.3 설치 확인

```python
import nimg_v2
print(nimg_v2.__version__)  # 2.0.0
```

---

## 4. 실행 방법

### 4.1 명령줄 실행

```bash
# 기본 실행
python3 -m nimg_v2.main \
    --data-dir ./20251208_161246_output \
    --output-dir ./output \
    --model-path /root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt

python3 -m nimg_v2.main \
    --data-dir /root/fursys_imgprosessing_ws/20251208_161246_output \
    --output-dir ./output \
    --model-path /root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt

# 옵션 추가
python -m nimg_v2.main \
    --data-dir ./20251208_155531_output \
    --output-dir ./output \
    --model-path ./models/class187_image85286_v12x_250epochs.pt \
    --intrinsics ./nimg_v2/config/camera_intrinsics.yaml \
    --reference-frame 0 \
    --max-frames 1000 \
    --verbose
```

### 4.2 Python 코드에서 사용

```python
from nimg_v2.main import ImageProcessingPipeline

# 카메라 내부 파라미터
intrinsics = {
    'fx': 636.3392652788157,
    'fy': 636.4266464742717,
    'cx': 654.3418233071645,
    'cy': 399.58963414918554
}

# 파이프라인 초기화
pipeline = ImageProcessingPipeline(
    model_path="./models/class187_image85286_v12x_250epochs.pt",
    intrinsics=intrinsics,
    reference_frame_idx=0
)

# 데이터셋 처리
results_df = pipeline.process_dataset(
    data_dir="./20251208_155531_output",
    output_dir="./output",
    max_frames=1000
)

# 요약 출력
pipeline.print_summary()
```

### 4.3 개별 모듈 사용

```python
# 데이터 로더
from nimg_v2.data.data_loader import DataLoader

loader = DataLoader("./20251208_155531_output")
print(f"총 {len(loader)} 프레임")

for frame in loader:
    print(f"Frame {frame.frame_idx}: {frame.rgb.shape}, IMU: {frame.has_imu}")

# Kalman Filter
from nimg_v2.tracking.kalman_filter_3d import KalmanFilter3D
import numpy as np

kf = KalmanFilter3D(dt=1/30.0)
kf.initialize(np.array([0.0, 0.0, 1.0]))

position, velocity, acceleration = kf.predict_and_update(np.array([0.01, 0.0, 1.0]))
print(f"Position: {position}, Velocity: {velocity}")

# 방향 추정
from nimg_v2.estimation.orientation_estimator import OrientationEstimator

estimator = OrientationEstimator(min_points=100)
# result = estimator.estimate_from_depth(depth, bbox, intrinsics)
```

---

## 5. 테스트

### 5.1 전체 테스트 실행

```bash
cd /root/fursys_imgprosessing_ws/src/nimg_v2
pytest tests/ -v
```

### 5.2 개별 테스트 실행

```bash
# Kalman Filter 테스트
pytest tests/test_kalman_filter.py -v

# 방향 추정 테스트
pytest tests/test_orientation_estimator.py -v

# 전체 파이프라인 테스트
pytest tests/test_full_pipeline.py -v
```

### 5.3 커버리지 확인

```bash
pytest tests/ --cov=nimg_v2 --cov-report=html
# htmlcov/index.html에서 결과 확인
```

---

## 6. 문제 해결

### 6.1 CUDA 관련 오류

```
RuntimeError: CUDA out of memory
```

**해결**:
```python
# 배치 크기 줄이기 또는 CPU 사용
import torch
torch.cuda.empty_cache()

# 또는 환경 변수로 GPU 메모리 제한
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

### 6.2 Open3D 설치 오류

```
ImportError: libGL.so.1: cannot open shared object file
```

**해결 (Ubuntu)**:
```bash
sudo apt-get install libgl1-mesa-glx
```

### 6.3 ultralytics 모델 로드 실패

```
RuntimeError: Couldn't load custom C++ ops
```

**해결**:
```bash
pip uninstall ultralytics
pip install ultralytics==8.0.0  # 특정 버전 설치
```

### 6.4 메모리 부족

**해결**:
```python
# max_frames 옵션으로 처리 프레임 수 제한
results_df = pipeline.process_dataset(
    data_dir="./data",
    output_dir="./output",
    max_frames=500  # 제한
)
```

---

## 출력 파일 설명

처리 완료 후 `output/` 디렉토리에 다음 파일들이 생성됩니다:

| 파일 | 설명 |
|------|------|
| `results.csv` | 프레임별 변화량 데이터 (CSV) |
| `results.json` | 전체 결과 및 메타데이터 (JSON) |
| `report.txt` | 처리 결과 요약 리포트 |
| `results.png` | 속도/위치/각도 변화 그래프 |
| `trajectory_3d.png` | 3D 궤적 그래프 |
| `orientation.png` | Roll/Pitch/Yaw 상세 그래프 |

---

## CSV 출력 컬럼 설명

| 컬럼 | 설명 | 단위 |
|------|------|------|
| `frame_idx` | 프레임 인덱스 | - |
| `timestamp` | 타임스탬프 | 초 |
| `dx`, `dy`, `dz` | 기준 대비 위치 변화 | 미터 |
| `vx`, `vy`, `vz` | 3D 속도 | m/s |
| `speed` | 속력 | m/s |
| `roll_change` | Roll 각도 변화 | 도 |
| `pitch_change` | Pitch 각도 변화 | 도 |
| `yaw_change` | Yaw 각도 변화 | 도 |
| `overall_confidence` | 전체 신뢰도 | 0-1 |

---

## 참고 문서

- [implementation_design_guide.md](../../research/251205/implementation_design_guide.md): 상세 설계 문서
- [comprehensive_improvement_research_2025.md](../../research/251205/comprehensive_improvement_research_2025.md): 개선 연구 문서

---

*작성일: 2025-12-17*
*버전: 2.0.0*
