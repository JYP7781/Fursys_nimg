# 6DoF 자세 추정에서 오일러 각도 vs 쿼터니언 변환 심층 분석

**작성일**: 2025-12-31
**버전**: v1.0
**목적**: nimg_v3 프로젝트에서 FoundationPose 6DoF 출력을 오일러 각도 또는 쿼터니언으로 변환 시 최적의 선택 분석
**관련 문서**: `nimg_v3_foundationpose_comprehensive_design_guide.md` 섹션 6.2

---

## 목차

1. [개요 및 배경](#1-개요-및-배경)
2. [오일러 각도(Euler Angles) 상세 분석](#2-오일러-각도euler-angles-상세-분석)
3. [쿼터니언(Quaternion) 상세 분석](#3-쿼터니언quaternion-상세-분석)
4. [핵심 비교: 오일러 vs 쿼터니언](#4-핵심-비교-오일러-vs-쿼터니언)
5. [짐벌 락(Gimbal Lock) 문제 심층 분석](#5-짐벌-락gimbal-lock-문제-심층-분석)
6. [6D 연속 회전 표현 (최신 대안)](#6-6d-연속-회전-표현-최신-대안)
7. [nimg_v3 프로젝트 적용 분석](#7-nimg_v3-프로젝트-적용-분석)
8. [구현 권장사항](#8-구현-권장사항)
9. [코드 구현 예시](#9-코드-구현-예시)
10. [결론 및 최종 권장사항](#10-결론-및-최종-권장사항)
11. [참고 자료](#11-참고-자료)

---

## 1. 개요 및 배경

### 1.1 문제 정의

`nimg_v3_foundationpose_comprehensive_design_guide.md`의 섹션 6.2에서는 FoundationPose의 6DoF 자세 추정 결과(4x4 변환 행렬)를 **오일러 각도(Roll, Pitch, Yaw)**로 변환하는 `pose_to_euler.py` 모듈을 제안하고 있습니다.

본 문서에서는 다음 질문에 대해 심층 분석합니다:

> **오일러 각도 대신 쿼터니언으로 변환이 가능한가? 두 방식 중 어떤 것이 더 나으며, 실제 구현에서 어떤 방식을 채택해야 하는가?**

### 1.2 6DoF 자세 표현의 기본 개념

6DoF (Six Degrees of Freedom) 자세는 3D 공간에서 객체의 위치와 방향을 완전히 기술합니다:

```
6DoF = 이동(Translation) + 회전(Rotation)
     = [tx, ty, tz]     + [R] (3x3 회전 행렬)
     = SE(3) 그룹의 원소
```

**회전을 표현하는 주요 방법:**

| 표현 방식 | 파라미터 수 | 특징 |
|----------|-----------|------|
| 회전 행렬 (Rotation Matrix) | 9 (실제 DoF: 3) | 가장 기본적, 직접 적용 가능 |
| 오일러 각도 (Euler Angles) | 3 | 직관적, 짐벌 락 문제 |
| 쿼터니언 (Quaternion) | 4 (정규화 필요) | 짐벌 락 없음, 효율적 보간 |
| 축-각도 (Axis-Angle) | 4 (또는 3) | 회전축과 각도로 표현 |
| 6D 연속 표현 | 6 | 신경망 학습에 최적 |

---

## 2. 오일러 각도(Euler Angles) 상세 분석

### 2.1 정의 및 개념

오일러 각도는 3D 회전을 세 개의 연속적인 축 회전으로 분해합니다:

```
R = Rz(yaw) × Ry(pitch) × Rx(roll)   # ZYX 순서 (항공/로봇공학 표준)
```

- **Roll (φ)**: X축 중심 회전 (좌우 기울기)
- **Pitch (θ)**: Y축 중심 회전 (앞뒤 기울기)
- **Yaw (ψ)**: Z축 중심 회전 (방향 전환)

### 2.2 장점

| 장점 | 설명 |
|------|------|
| **직관성** | 사람이 이해하기 가장 쉬운 표현. "Yaw 45°" 같은 표현이 자연스러움 |
| **최소 저장 공간** | 3개의 숫자만 필요 (메모리 효율적) |
| **독립적 제어** | 각 축의 회전을 개별적으로 제어 가능 |
| **디버깅 용이** | 값을 보고 직접 회전 상태 파악 가능 |
| **압축 용이** | 고정 소수점 시스템으로 쉽게 압축 가능 |

### 2.3 단점

| 단점 | 설명 |
|------|------|
| **짐벌 락(Gimbal Lock)** | Pitch가 ±90°일 때 Roll과 Yaw 구분 불가 |
| **불연속성** | α와 α+360°가 같은 회전을 나타냄 |
| **비고유성** | 동일 회전에 대해 여러 표현 존재 |
| **보간 어려움** | 두 자세 사이 자연스러운 보간이 복잡 |
| **회전 합성 복잡** | 여러 회전 결합 시 순서에 민감 |
| **회귀 학습 부적합** | 신경망 학습에서 불연속성 문제 |

### 2.4 수학적 표현

```python
# ZYX 오일러 각도에서 회전 행렬로 변환
def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    roll: X축 회전 (라디안)
    pitch: Y축 회전 (라디안)
    yaw: Z축 회전 (라디안)
    """
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx  # ZYX 순서
```

---

## 3. 쿼터니언(Quaternion) 상세 분석

### 3.1 정의 및 개념

쿼터니언은 4차원 복소수 확장으로, 3D 회전을 표현합니다:

```
q = w + xi + yj + zk

여기서:
- w: 스칼라 부분 (실수)
- (x, y, z): 벡터 부분 (허수)
- i² = j² = k² = ijk = -1
```

**축-각도 관계:**
```
q = cos(θ/2) + sin(θ/2)(ax·i + ay·j + az·k)

여기서:
- θ: 회전 각도
- (ax, ay, az): 단위 회전 축
```

### 3.2 장점

| 장점 | 설명 |
|------|------|
| **짐벌 락 없음** | 모든 회전을 안정적으로 표현 |
| **수치적 안정성** | 부동소수점 오류에 강건 |
| **효율적 연산** | 회전 합성이 곱셈으로 간단 |
| **부드러운 보간** | SLERP로 자연스러운 회전 보간 |
| **정규화 용이** | 단위 쿼터니언 유지가 간단 |
| **메모리 효율** | 회전 행렬(9개)보다 적은 4개 파라미터 |

### 3.3 단점

| 단점 | 설명 |
|------|------|
| **직관성 부족** | 사람이 이해하기 어려움 |
| **이중 표현(Antipodal)** | q와 -q가 같은 회전 표현 |
| **정규화 필요** | 항상 단위 길이 유지해야 함 |
| **디버깅 어려움** | 값만 보고 회전 상태 파악 곤란 |
| **4D 불연속성** | 신경망 학습 시 여전히 불연속점 존재 |

### 3.4 수학적 표현

```python
# 쿼터니언에서 회전 행렬로 변환
def quaternion_to_rotation_matrix(q):
    """
    q = [x, y, z, w] (scipy 형식) 또는 [w, x, y, z]
    """
    w, x, y, z = q[3], q[0], q[1], q[2]  # scipy 형식 가정

    # 정규화
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n

    R = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])

    return R
```

### 3.5 SLERP (Spherical Linear Interpolation)

쿼터니언의 핵심 장점 중 하나는 두 회전 사이의 부드러운 보간:

```python
def slerp(q0, q1, t):
    """
    q0, q1: 시작/끝 쿼터니언
    t: 보간 파라미터 [0, 1]
    """
    dot = np.dot(q0, q1)

    # 최단 경로 선택
    if dot < 0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        # 선형 보간 (거의 같은 방향)
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    s0 = np.sin((1 - t) * theta) / sin_theta
    s1 = np.sin(t * theta) / sin_theta

    return s0 * q0 + s1 * q1
```

---

## 4. 핵심 비교: 오일러 vs 쿼터니언

### 4.1 종합 비교표

| 특성 | 오일러 각도 | 쿼터니언 | 승자 |
|------|-----------|---------|------|
| **파라미터 수** | 3 | 4 | 오일러 |
| **직관성** | 매우 높음 | 낮음 | 오일러 |
| **짐벌 락** | 있음 | 없음 | **쿼터니언** |
| **수치 안정성** | 낮음 | 높음 | **쿼터니언** |
| **보간 품질** | 어려움 | SLERP (우수) | **쿼터니언** |
| **연산 효율** | 삼각함수 필요 | 곱셈만 필요 | **쿼터니언** |
| **회전 합성** | 복잡 | 간단 (곱셈) | **쿼터니언** |
| **디버깅** | 용이 | 어려움 | 오일러 |
| **저장 공간** | 최소 (3) | 중간 (4) | 오일러 |
| **ROS 표준** | 변환용 | 기본 표현 | **쿼터니언** |
| **신경망 학습** | 불연속성 | 불연속성 | 무승부 |

### 4.2 사용 사례별 권장 표현

| 사용 사례 | 권장 표현 | 이유 |
|----------|----------|------|
| **ROS 통신** | 쿼터니언 | ROS 표준 (geometry_msgs/Pose) |
| **내부 계산** | 쿼터니언 | 수치 안정성, 연산 효율 |
| **사용자 표시** | 오일러 | 직관적 이해 |
| **설정 파일** | 오일러 | 편집 용이 |
| **로깅/디버깅** | 둘 다 | 오일러로 표시, 쿼터니언으로 저장 |
| **보간/애니메이션** | 쿼터니언 | SLERP 지원 |
| **신경망 출력** | 6D 표현 | 연속성 보장 |

### 4.3 ROS 2 표준 관행

ROS 2 공식 문서에 따르면:

> "ROS 2 uses quaternions to track and apply rotations."
> "For representing orientation in ROS, quaternions are used. They do not represent a human-readable format like e.g. euler angles."

**ROS 2 권장 워크플로우:**
```
입력(오일러) → 내부(쿼터니언) → 계산(쿼터니언) → 출력(오일러 for 사람)
```

---

## 5. 짐벌 락(Gimbal Lock) 문제 심층 분석

### 5.1 짐벌 락이란?

짐벌 락은 3D 회전을 세 개의 연속 회전으로 분해할 때 발생하는 특이점 문제입니다:

```
발생 조건: Pitch = ±90° (또는 회전 순서에 따라 다른 축)

결과:
- 두 축이 정렬되어 자유도 손실
- Roll과 Yaw 구분 불가
- 회전 제어 불능
```

### 5.2 역사적 사례: Apollo 11

> "A well-known gimbal lock incident happened in the Apollo 11 Moon mission. On this spacecraft, a set of gimbals was used on an inertial measurement unit (IMU)."

현대적 해결:
> "Modern practice is to avoid the use of gimbals entirely by mounting the inertial sensors directly to the body of the vehicle (strapdown system) and integrating sensed rotation and acceleration digitally using quaternion methods."

### 5.3 nimg_v3에서의 영향 분석

**도장물 회전 범위 고려:**

| Pitch 범위 | 짐벌 락 위험 | 영향 |
|-----------|-------------|------|
| ±30° | 매우 낮음 | 무시 가능 |
| ±60° | 낮음 | 주의 필요 |
| ±75° | 중간 | 대응 권장 |
| ±85° 이상 | 높음 | 반드시 쿼터니언 사용 |

**실제 도장물 측정 시나리오:**
- 일반적으로 Pitch 범위: -30° ~ +30°
- 극단적 경우에도 ±60° 이내
- **결론: 짐벌 락 위험 낮음, 그러나 대비 권장**

### 5.4 짐벌 락 발생 시 증상

```python
# 짐벌 락 감지 코드
def detect_gimbal_lock(euler_angles, threshold_deg=85):
    """
    Pitch가 ±threshold_deg 근처일 때 경고
    """
    pitch = euler_angles[1]  # 도 단위

    if abs(abs(pitch) - 90) < (90 - threshold_deg):
        return True, f"Warning: Pitch={pitch:.1f}° is near gimbal lock region"
    return False, "OK"
```

---

## 6. 6D 연속 회전 표현 (최신 대안)

### 6.1 배경: Zhou et al. (CVPR 2019)

Yi Zhou 등의 연구 ["On the Continuity of Rotation Representations in Neural Networks"](https://arxiv.org/abs/1812.07035)에서 핵심 발견:

> **"For 3D rotations, all representations are discontinuous in the real Euclidean spaces of four or fewer dimensions."**

**의미:**
- 오일러 각도 (3D): 불연속
- 쿼터니언 (4D): 불연속
- 축-각도 (3D/4D): 불연속
- **5D/6D 표현: 연속** ← 신경망 학습에 최적

### 6.2 6D 표현 방법

```python
def rotation_matrix_to_6d(R):
    """
    3x3 회전 행렬에서 6D 표현 추출
    (처음 두 열 사용)
    """
    return R[:, :2].flatten()  # [r1, r2] = 6개 값

def sixd_to_rotation_matrix(sixd):
    """
    6D 표현에서 회전 행렬 복원
    Gram-Schmidt 직교화 사용
    """
    a1 = sixd[:3]
    a2 = sixd[3:6]

    # 정규화
    b1 = a1 / np.linalg.norm(a1)

    # 직교화
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)

    # 세 번째 축
    b3 = np.cross(b1, b2)

    return np.column_stack([b1, b2, b3])
```

### 6.3 성능 비교 (Zhou et al. 논문 결과)

| 표현 방식 | 평균 오차 | 95% 오차 < 5° |
|----------|----------|--------------|
| **6D 연속** | 최저 | ~95% |
| 5D 연속 | 낮음 | ~90% |
| 쿼터니언 (4D) | 중간 | ~75% |
| 축-각도 (3D) | 높음 | ~60% |
| 오일러 (3D) | **최고** | ~10% |

> "The 6D representation has the lowest mean and standard deviation of errors with around 95% of errors lower than 5°, while Euler representation is the worst with around 10% of errors higher than 25°."

### 6.4 nimg_v3에서의 적용 가능성

| 시나리오 | 6D 표현 적용 여부 | 이유 |
|----------|-----------------|------|
| **FoundationPose 내부** | 이미 사용 중 (가능성 높음) | 신경망 출력 |
| **Kalman Filter 입력** | 권장하지 않음 | 복잡성 증가 |
| **사용자 출력** | 권장하지 않음 | 직관성 부재 |
| **중간 처리** | 선택적 | 특정 최적화 시 |

---

## 7. nimg_v3 프로젝트 적용 분석

### 7.1 현재 설계 분석 (pose_to_euler.py)

```python
# 현재 설계 (nimg_v3_foundationpose_comprehensive_design_guide.md 6.2절)
@dataclass
class EulerAngles:
    """오일러 각도 (degrees)"""
    roll: float   # X축 회전
    pitch: float  # Y축 회전
    yaw: float    # Z축 회전

def pose_to_euler(pose_matrix: np.ndarray) -> Tuple[np.ndarray, EulerAngles]:
    # 회전 행렬 → 오일러 각도 변환
    r = Rotation.from_matrix(rotation_matrix)
    euler = r.as_euler('xyz', degrees=True)
    ...
```

### 7.2 쿼터니언 지원 필요성 평가

**고려 요소:**

| 요소 | 현재 상태 | 쿼터니언 필요성 |
|------|----------|---------------|
| **ROS 2 통합** | ros2_publisher.py 계획 | 높음 (표준 형식) |
| **Kalman Filter** | 오일러 기반 | 중간 (쿼터니언 KF 가능) |
| **결과 저장** | CSV/JSON | 낮음 (오일러로 충분) |
| **시각화** | 오일러 표시 | 낮음 |
| **자세 보간** | 현재 없음 | 필요시 높음 |
| **도장물 회전 범위** | 일반적 ±30° | 낮음 |

### 7.3 하이브리드 접근법 제안

**권장 아키텍처:**

```
┌─────────────────────────────────────────────────────────────┐
│                      FoundationPose                          │
│               (내부: 회전 행렬 / 6D 표현)                      │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│              pose_converter.py (제안)                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  입력: 4x4 Pose Matrix                                │  │
│  │                                                        │  │
│  │  출력:                                                 │  │
│  │   ├── EulerAngles (roll, pitch, yaw)  ← 사용자 표시   │  │
│  │   ├── Quaternion (x, y, z, w)         ← ROS, 내부 계산│  │
│  │   └── RotationMatrix (3x3)            ← 필요시        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Kalman Filter │  │  ROS Publisher  │  │   CSV Export    │
│  (쿼터니언 권장) │  │   (쿼터니언)    │  │   (오일러)      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 8. 구현 권장사항

### 8.1 최종 권장: 하이브리드 접근법

| 용도 | 권장 표현 | 이유 |
|------|----------|------|
| **내부 저장/처리** | 쿼터니언 | 수치 안정성, 연산 효율 |
| **ROS 메시지** | 쿼터니언 | ROS 표준 (geometry_msgs/Pose) |
| **Kalman Filter** | 쿼터니언 | 짐벌 락 방지, 안정성 |
| **사용자 출력** | 오일러 | 직관적 이해 |
| **CSV/JSON 로깅** | 둘 다 | 호환성 + 디버깅 |
| **변화량 계산** | 쿼터니언 → 오일러 | 정확한 계산 후 변환 |

### 8.2 구현 우선순위

```
1단계: 기본 변환 모듈 (필수)
   ├── pose_to_euler() - 현재 설계 유지
   ├── pose_to_quaternion() - 새로 추가
   └── quaternion_to_euler() - 양방향 변환

2단계: Kalman Filter 개선 (권장)
   ├── 쿼터니언 기반 상태 표현 옵션
   └── 짐벌 락 감지 및 경고

3단계: ROS 통합 (필요시)
   └── geometry_msgs/Pose 형식 지원
```

### 8.3 짐벌 락 대응 전략

```python
class RotationConverter:
    """하이브리드 회전 변환기"""

    GIMBAL_LOCK_THRESHOLD = 85.0  # 도

    def __init__(self, warn_gimbal_lock: bool = True):
        self.warn_gimbal_lock = warn_gimbal_lock

    def matrix_to_all(self, R: np.ndarray) -> Dict[str, Any]:
        """회전 행렬에서 모든 표현으로 변환"""
        rotation = Rotation.from_matrix(R)

        euler = rotation.as_euler('xyz', degrees=True)
        quat = rotation.as_quat()  # [x, y, z, w]

        # 짐벌 락 감지
        gimbal_lock = False
        if self.warn_gimbal_lock:
            if abs(abs(euler[1]) - 90) < (90 - self.GIMBAL_LOCK_THRESHOLD):
                gimbal_lock = True
                print(f"Warning: Approaching gimbal lock (pitch={euler[1]:.1f}°)")

        return {
            'euler': EulerAngles(roll=euler[0], pitch=euler[1], yaw=euler[2]),
            'quaternion': Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
            'matrix': R,
            'gimbal_lock_warning': gimbal_lock
        }
```

---

## 9. 코드 구현 예시

### 9.1 개선된 pose_converter.py

```python
"""
pose_converter.py - 통합 자세 변환 모듈

6DoF 자세 행렬에서 다양한 회전 표현으로 변환 지원:
- 오일러 각도 (Roll, Pitch, Yaw)
- 쿼터니언 (x, y, z, w)
- 회전 행렬 (3x3)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class RotationOrder(Enum):
    """오일러 각도 회전 순서"""
    XYZ = 'xyz'  # Roll-Pitch-Yaw (항공/로봇공학 표준)
    ZYX = 'zyx'  # Yaw-Pitch-Roll
    ZXY = 'zxy'


@dataclass
class EulerAngles:
    """오일러 각도 (degrees)"""
    roll: float   # X축 회전
    pitch: float  # Y축 회전
    yaw: float    # Z축 회전

    def to_radians(self) -> 'EulerAngles':
        return EulerAngles(
            roll=np.deg2rad(self.roll),
            pitch=np.deg2rad(self.pitch),
            yaw=np.deg2rad(self.yaw)
        )

    def to_array(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.yaw])


@dataclass
class Quaternion:
    """쿼터니언 (x, y, z, w) - scipy/ROS 형식"""
    x: float
    y: float
    z: float
    w: float

    def to_array(self) -> np.ndarray:
        """[x, y, z, w] 형식"""
        return np.array([self.x, self.y, self.z, self.w])

    def to_array_wxyz(self) -> np.ndarray:
        """[w, x, y, z] 형식 (일부 라이브러리용)"""
        return np.array([self.w, self.x, self.y, self.z])

    def normalize(self) -> 'Quaternion':
        arr = self.to_array()
        norm = np.linalg.norm(arr)
        arr = arr / norm
        return Quaternion(x=arr[0], y=arr[1], z=arr[2], w=arr[3])

    def conjugate(self) -> 'Quaternion':
        return Quaternion(x=-self.x, y=-self.y, z=-self.z, w=self.w)


@dataclass
class PoseComponents:
    """자세 구성요소 통합"""
    translation: np.ndarray      # [x, y, z] 미터
    euler: EulerAngles          # Roll, Pitch, Yaw (도)
    quaternion: Quaternion      # (x, y, z, w)
    rotation_matrix: np.ndarray # 3x3
    gimbal_lock_warning: bool   # 짐벌 락 경고


class PoseConverter:
    """
    통합 자세 변환 클래스

    FoundationPose 출력(4x4 변환 행렬)을
    다양한 회전 표현으로 변환합니다.
    """

    GIMBAL_LOCK_THRESHOLD = 85.0  # 도

    def __init__(
        self,
        rotation_order: RotationOrder = RotationOrder.XYZ,
        warn_gimbal_lock: bool = True,
        use_degrees: bool = True
    ):
        """
        Args:
            rotation_order: 오일러 각도 회전 순서
            warn_gimbal_lock: 짐벌 락 경고 활성화
            use_degrees: 오일러 각도를 도 단위로 출력
        """
        self.rotation_order = rotation_order
        self.warn_gimbal_lock = warn_gimbal_lock
        self.use_degrees = use_degrees

    def pose_matrix_to_components(
        self,
        pose_matrix: np.ndarray
    ) -> PoseComponents:
        """
        4x4 자세 행렬에서 모든 구성요소 추출

        Args:
            pose_matrix: 4x4 변환 행렬 [R|t; 0 1]

        Returns:
            PoseComponents: 이동, 오일러, 쿼터니언, 회전 행렬
        """
        # 이동 추출
        translation = pose_matrix[:3, 3].copy()

        # 회전 행렬 추출
        rotation_matrix = pose_matrix[:3, :3].copy()

        # scipy Rotation 객체 생성
        rot = Rotation.from_matrix(rotation_matrix)

        # 오일러 각도 추출
        euler_array = rot.as_euler(self.rotation_order.value, degrees=self.use_degrees)
        euler = EulerAngles(
            roll=euler_array[0],
            pitch=euler_array[1],
            yaw=euler_array[2]
        )

        # 쿼터니언 추출 [x, y, z, w]
        quat_array = rot.as_quat()
        quaternion = Quaternion(
            x=quat_array[0],
            y=quat_array[1],
            z=quat_array[2],
            w=quat_array[3]
        )

        # 짐벌 락 감지
        gimbal_lock = self._check_gimbal_lock(euler.pitch)

        return PoseComponents(
            translation=translation,
            euler=euler,
            quaternion=quaternion,
            rotation_matrix=rotation_matrix,
            gimbal_lock_warning=gimbal_lock
        )

    def pose_to_euler(
        self,
        pose_matrix: np.ndarray
    ) -> Tuple[np.ndarray, EulerAngles]:
        """
        기존 API 호환: 4x4 행렬에서 이동과 오일러 각도 추출
        """
        components = self.pose_matrix_to_components(pose_matrix)
        return components.translation, components.euler

    def pose_to_quaternion(
        self,
        pose_matrix: np.ndarray
    ) -> Tuple[np.ndarray, Quaternion]:
        """
        4x4 행렬에서 이동과 쿼터니언 추출
        """
        components = self.pose_matrix_to_components(pose_matrix)
        return components.translation, components.quaternion

    def quaternion_to_euler(
        self,
        quat: Quaternion
    ) -> EulerAngles:
        """
        쿼터니언에서 오일러 각도로 변환
        """
        rot = Rotation.from_quat(quat.to_array())
        euler_array = rot.as_euler(self.rotation_order.value, degrees=self.use_degrees)
        return EulerAngles(
            roll=euler_array[0],
            pitch=euler_array[1],
            yaw=euler_array[2]
        )

    def euler_to_quaternion(
        self,
        euler: EulerAngles
    ) -> Quaternion:
        """
        오일러 각도에서 쿼터니언으로 변환
        """
        euler_array = euler.to_array()
        rot = Rotation.from_euler(
            self.rotation_order.value,
            euler_array,
            degrees=self.use_degrees
        )
        quat_array = rot.as_quat()
        return Quaternion(
            x=quat_array[0],
            y=quat_array[1],
            z=quat_array[2],
            w=quat_array[3]
        )

    def _check_gimbal_lock(self, pitch: float) -> bool:
        """
        짐벌 락 근접 여부 확인
        """
        if not self.warn_gimbal_lock:
            return False

        # 도 단위 기준
        pitch_deg = pitch if self.use_degrees else np.rad2deg(pitch)

        if abs(abs(pitch_deg) - 90) < (90 - self.GIMBAL_LOCK_THRESHOLD):
            print(f"⚠️ Warning: Approaching gimbal lock (pitch={pitch_deg:.1f}°)")
            return True

        return False

    @staticmethod
    def slerp(
        q0: Quaternion,
        q1: Quaternion,
        t: float
    ) -> Quaternion:
        """
        두 쿼터니언 사이 구면 선형 보간 (SLERP)

        Args:
            q0: 시작 쿼터니언
            q1: 끝 쿼터니언
            t: 보간 파라미터 [0, 1]

        Returns:
            보간된 쿼터니언
        """
        from scipy.spatial.transform import Slerp

        key_times = [0, 1]
        key_rots = Rotation.from_quat([q0.to_array(), q1.to_array()])
        slerp = Slerp(key_times, key_rots)

        result = slerp(t).as_quat()
        return Quaternion(x=result[0], y=result[1], z=result[2], w=result[3])

    @staticmethod
    def compute_rotation_difference(
        q1: Quaternion,
        q2: Quaternion
    ) -> Tuple[float, np.ndarray]:
        """
        두 쿼터니언 사이의 회전 차이 계산

        Returns:
            angle: 회전 각도 (도)
            axis: 회전 축 (단위 벡터)
        """
        rot1 = Rotation.from_quat(q1.to_array())
        rot2 = Rotation.from_quat(q2.to_array())

        diff = rot2 * rot1.inv()
        rotvec = diff.as_rotvec()

        angle = np.linalg.norm(rotvec)
        axis = rotvec / angle if angle > 1e-6 else np.array([0, 0, 1])

        return np.rad2deg(angle), axis


# 편의 함수들
def pose_to_euler(pose_matrix: np.ndarray) -> Tuple[np.ndarray, EulerAngles]:
    """
    기존 API 호환 함수
    """
    converter = PoseConverter()
    return converter.pose_to_euler(pose_matrix)


def pose_to_quaternion(pose_matrix: np.ndarray) -> Tuple[np.ndarray, Quaternion]:
    """
    4x4 자세 행렬에서 이동과 쿼터니언 추출
    """
    converter = PoseConverter()
    return converter.pose_to_quaternion(pose_matrix)


def compute_angle_change(
    prev_pose: np.ndarray,
    curr_pose: np.ndarray,
    use_quaternion: bool = True
) -> np.ndarray:
    """
    두 자세 간의 각도 변화 계산

    Args:
        prev_pose: 이전 자세 (4x4)
        curr_pose: 현재 자세 (4x4)
        use_quaternion: True면 쿼터니언으로 정확히 계산 후 오일러 변환

    Returns:
        delta_angles: [Δroll, Δpitch, Δyaw] 도 단위
    """
    converter = PoseConverter()

    if use_quaternion:
        # 쿼터니언 기반 정확한 계산
        _, q_prev = converter.pose_to_quaternion(prev_pose)
        _, q_curr = converter.pose_to_quaternion(curr_pose)

        # 회전 차이 계산
        rot_prev = Rotation.from_quat(q_prev.to_array())
        rot_curr = Rotation.from_quat(q_curr.to_array())

        rot_diff = rot_curr * rot_prev.inv()
        delta_euler = rot_diff.as_euler('xyz', degrees=True)
    else:
        # 오일러 기반 직접 계산 (짐벌 락 근처에서 부정확)
        _, prev_euler = converter.pose_to_euler(prev_pose)
        _, curr_euler = converter.pose_to_euler(curr_pose)

        delta_euler = np.array([
            curr_euler.roll - prev_euler.roll,
            curr_euler.pitch - prev_euler.pitch,
            curr_euler.yaw - prev_euler.yaw
        ])

    # -180 ~ 180 범위로 정규화
    delta_euler = np.where(delta_euler > 180, delta_euler - 360, delta_euler)
    delta_euler = np.where(delta_euler < -180, delta_euler + 360, delta_euler)

    return delta_euler
```

### 9.2 개선된 Kalman Filter (쿼터니언 옵션)

```python
"""
pose_kalman_filter.py - 쿼터니언 지원 Kalman Filter

두 가지 모드 지원:
1. 오일러 기반 (기존 호환)
2. 쿼터니언 기반 (권장, 짐벌 락 안전)
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from .pose_converter import EulerAngles, Quaternion, PoseConverter


class FilterMode(Enum):
    EULER = "euler"
    QUATERNION = "quaternion"


@dataclass
class PoseKalmanState:
    """Kalman Filter 상태"""
    position: np.ndarray       # [x, y, z] 미터
    velocity: np.ndarray       # [vx, vy, vz] m/s
    orientation_euler: EulerAngles  # Roll, Pitch, Yaw (도)
    orientation_quat: Quaternion    # 쿼터니언
    angular_velocity: np.ndarray    # [ωx, ωy, ωz] °/s
    speed: float                    # 선속도 크기 m/s


class PoseKalmanFilter:
    """
    6DoF 자세 추정을 위한 Kalman Filter

    오일러 모드:
        상태: [x, y, z, vx, vy, vz, roll, pitch, yaw, ωx, ωy, ωz]

    쿼터니언 모드:
        상태: [x, y, z, vx, vy, vz, qx, qy, qz, qw, ωx, ωy, ωz]
    """

    def __init__(
        self,
        dt: float = 1/30.0,
        mode: FilterMode = FilterMode.QUATERNION,
        process_noise_pos: float = 0.01,
        process_noise_vel: float = 0.1,
        process_noise_orient: float = 0.1,
        process_noise_angular_vel: float = 1.0,
        measurement_noise_pos: float = 0.005,
        measurement_noise_orient: float = 0.5
    ):
        self.dt = dt
        self.mode = mode
        self._converter = PoseConverter()

        if mode == FilterMode.EULER:
            self._init_euler_filter(
                process_noise_pos, process_noise_vel,
                process_noise_orient, process_noise_angular_vel,
                measurement_noise_pos, measurement_noise_orient
            )
        else:
            self._init_quaternion_filter(
                process_noise_pos, process_noise_vel,
                process_noise_orient, process_noise_angular_vel,
                measurement_noise_pos, measurement_noise_orient
            )

        self._initialized = False

    def _init_euler_filter(self, pn_pos, pn_vel, pn_orient, pn_angvel, mn_pos, mn_orient):
        """오일러 기반 12-상태 필터"""
        # 상태: [x, y, z, vx, vy, vz, roll, pitch, yaw, ωx, ωy, ωz]
        self.kf = KalmanFilter(dim_x=12, dim_z=6)

        # 상태 전이 행렬
        self.kf.F = np.eye(12)
        for i in range(3):
            self.kf.F[i, i+3] = self.dt     # pos += vel * dt
            self.kf.F[i+6, i+9] = self.dt   # angle += angvel * dt

        # 측정 행렬
        self.kf.H = np.zeros((6, 12))
        self.kf.H[0, 0] = self.kf.H[1, 1] = self.kf.H[2, 2] = 1
        self.kf.H[3, 6] = self.kf.H[4, 7] = self.kf.H[5, 8] = 1

        # 노이즈 설정
        self.kf.Q = np.diag([pn_pos]*3 + [pn_vel]*3 + [pn_orient]*3 + [pn_angvel]*3)
        self.kf.R = np.diag([mn_pos]*3 + [mn_orient]*3)
        self.kf.P *= 1.0

    def _init_quaternion_filter(self, pn_pos, pn_vel, pn_orient, pn_angvel, mn_pos, mn_orient):
        """쿼터니언 기반 13-상태 필터"""
        # 상태: [x, y, z, vx, vy, vz, qx, qy, qz, qw, ωx, ωy, ωz]
        self.kf = KalmanFilter(dim_x=13, dim_z=7)

        # 상태 전이 행렬 (선형 근사)
        self.kf.F = np.eye(13)
        for i in range(3):
            self.kf.F[i, i+3] = self.dt  # pos += vel * dt
        # 쿼터니언 업데이트는 predict()에서 별도 처리

        # 측정 행렬
        self.kf.H = np.zeros((7, 13))
        for i in range(7):
            self.kf.H[i, i] = 1 if i < 3 else 1  # pos + quat
        self.kf.H[3, 6] = self.kf.H[4, 7] = self.kf.H[5, 8] = self.kf.H[6, 9] = 1

        # 노이즈 설정
        self.kf.Q = np.diag([pn_pos]*3 + [pn_vel]*3 + [pn_orient]*4 + [pn_angvel]*3)
        self.kf.R = np.diag([mn_pos]*3 + [mn_orient]*4)
        self.kf.P *= 1.0

    def initialize(self, position: np.ndarray, orientation):
        """
        필터 초기화

        Args:
            position: [x, y, z]
            orientation: EulerAngles 또는 Quaternion
        """
        if self.mode == FilterMode.EULER:
            if isinstance(orientation, Quaternion):
                orientation = self._converter.quaternion_to_euler(orientation)

            self.kf.x = np.array([
                position[0], position[1], position[2],
                0, 0, 0,
                orientation.roll, orientation.pitch, orientation.yaw,
                0, 0, 0
            ])
        else:
            if isinstance(orientation, EulerAngles):
                orientation = self._converter.euler_to_quaternion(orientation)

            self.kf.x = np.array([
                position[0], position[1], position[2],
                0, 0, 0,
                orientation.x, orientation.y, orientation.z, orientation.w,
                0, 0, 0
            ])

        self._initialized = True

    def predict(self) -> PoseKalmanState:
        """예측 단계"""
        if not self._initialized:
            raise RuntimeError("Filter not initialized")

        self.kf.predict()

        # 쿼터니언 정규화 (quaternion 모드)
        if self.mode == FilterMode.QUATERNION:
            quat = self.kf.x[6:10]
            norm = np.linalg.norm(quat)
            if norm > 1e-6:
                self.kf.x[6:10] = quat / norm

        return self.get_state()

    def update(self, position: np.ndarray, orientation) -> PoseKalmanState:
        """
        업데이트 단계

        Args:
            position: [x, y, z]
            orientation: EulerAngles 또는 Quaternion
        """
        if not self._initialized:
            self.initialize(position, orientation)
            return self.get_state()

        if self.mode == FilterMode.EULER:
            if isinstance(orientation, Quaternion):
                orientation = self._converter.quaternion_to_euler(orientation)
            measurement = np.concatenate([
                position,
                [orientation.roll, orientation.pitch, orientation.yaw]
            ])
        else:
            if isinstance(orientation, EulerAngles):
                orientation = self._converter.euler_to_quaternion(orientation)
            measurement = np.concatenate([
                position,
                [orientation.x, orientation.y, orientation.z, orientation.w]
            ])

        self.kf.update(measurement)

        # 쿼터니언 정규화
        if self.mode == FilterMode.QUATERNION:
            quat = self.kf.x[6:10]
            norm = np.linalg.norm(quat)
            if norm > 1e-6:
                self.kf.x[6:10] = quat / norm

        return self.get_state()

    def predict_and_update(self, position: np.ndarray, orientation) -> PoseKalmanState:
        """예측 + 업데이트"""
        self.predict()
        return self.update(position, orientation)

    def get_state(self) -> PoseKalmanState:
        """현재 상태 반환"""
        state = self.kf.x.flatten()

        if self.mode == FilterMode.EULER:
            euler = EulerAngles(
                roll=state[6], pitch=state[7], yaw=state[8]
            )
            quat = self._converter.euler_to_quaternion(euler)
            angular_vel = state[9:12]
        else:
            quat = Quaternion(
                x=state[6], y=state[7], z=state[8], w=state[9]
            )
            euler = self._converter.quaternion_to_euler(quat)
            angular_vel = state[10:13]

        return PoseKalmanState(
            position=state[0:3],
            velocity=state[3:6],
            orientation_euler=euler,
            orientation_quat=quat,
            angular_velocity=angular_vel,
            speed=np.linalg.norm(state[3:6])
        )

    def reset(self):
        """필터 리셋"""
        dim_x = 12 if self.mode == FilterMode.EULER else 13
        self.kf.x = np.zeros(dim_x)
        self.kf.P = np.eye(dim_x) * 1.0
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized
```

---

## 10. 결론 및 최종 권장사항

### 10.1 핵심 결론

| 질문 | 답변 |
|------|------|
| **쿼터니언 변환 가능 여부** | **가능** - scipy.spatial.transform 사용 |
| **더 나은 방식** | **용도에 따라 다름** - 하이브리드 권장 |
| **구현 필요성** | **두 방식 모두 구현 권장** |

### 10.2 최종 권장사항

```
┌────────────────────────────────────────────────────────────────────┐
│                    nimg_v3 회전 표현 권장 사항                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 내부 처리: 쿼터니언                                             │
│     └── Kalman Filter, 자세 보간, 연산                              │
│                                                                     │
│  2. 외부 인터페이스: 오일러                                         │
│     └── 사용자 출력, 설정 파일, 디버깅                              │
│                                                                     │
│  3. ROS 통신: 쿼터니언                                              │
│     └── geometry_msgs/Pose 표준 준수                                │
│                                                                     │
│  4. 저장/로깅: 둘 다                                                │
│     └── 쿼터니언(정확성) + 오일러(가독성)                           │
│                                                                     │
│  5. 신경망 학습 (해당시): 6D 연속 표현                              │
│     └── 커스텀 모델 학습 시에만 고려                                │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### 10.3 구현 우선순위

1. **1순위 (필수)**: `pose_converter.py` 모듈 - 양방향 변환 지원
2. **2순위 (권장)**: Kalman Filter 쿼터니언 모드 옵션
3. **3순위 (선택)**: 짐벌 락 감지 및 경고 시스템

### 10.4 도장물 측정 특화 고려사항

- **일반적 Pitch 범위 ±30°**: 짐벌 락 위험 낮음
- **그러나 예외 상황 대비**: 쿼터니언 내부 처리 권장
- **사용자 친화성**: 최종 출력은 오일러 유지

---

## 11. 참고 자료

### 11.1 학술 논문

- [On the Continuity of Rotation Representations in Neural Networks (Zhou et al., CVPR 2019)](https://arxiv.org/abs/1812.07035)
- [Pose estimation using linearized rotations and quaternion algebra (Barfoot et al.)](https://furgalep.github.io/sbib/barfoot_aa10.pdf)
- [WQuatNet: Wide range quaternion-based head pose estimation](https://link.springer.com/article/10.1007/s44443-025-00034-1)
- [QuatNet: Quaternion-Based Head Pose Estimation With Multiregression Loss](https://www.researchgate.net/publication/327169137_QuatNet_Quaternion-Based_Head_Pose_Estimation_With_Multiregression_Loss)

### 11.2 공식 문서

- [SciPy Rotation Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)
- [ROS 2 Quaternion Fundamentals](https://docs.ros.org/en/iron/Tutorials/Intermediate/Tf2/Quaternion-Fundamentals.html)
- [Gimbal Lock - Wikipedia](https://en.wikipedia.org/wiki/Gimbal_lock)

### 11.3 기술 블로그 및 튜토리얼

- [Better rotation representations for accurate pose estimation (Towards Data Science)](https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f/)
- [How to Convert a Quaternion Into Euler Angles in Python](https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/)
- [Euler and Quaternion Angles: Differences and Why it Matters (Inertial Labs)](https://inertiallabs.com/euler-and-quaternion-angles-differences-and-why-it-matters/)
- [Quaternions & Gimbal Lock](https://jackmin.home.blog/2019/05/25/quaternions-gimbal-lock/)

### 11.4 관련 프로젝트 문서

- `research/251231/nimg_v3_foundationpose_comprehensive_design_guide.md`
- `research/251231/foundationpose_implementation_guide.md`
- `research/251218/ai_based_velocity_angle_measurement_research.md`

---

*작성: 2025-12-31*
*버전: v1.0*
*목적: nimg_v3 프로젝트 6DoF 자세 표현 방식 결정을 위한 심층 분석*
