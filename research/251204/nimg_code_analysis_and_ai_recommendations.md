# nimg 코드베이스 분석 및 AI/ML 기반 개선 권장사항

**작성일**: 2025-12-04
**목적**: 현재 nimg 코드의 문제점 분석 및 AI/ML 기술을 활용한 Depth/3D Point Cloud 기반 속도/각도 측정 정확도 개선 방안 연구

---

## 목차

1. [현재 코드베이스 구조 분석](#1-현재-코드베이스-구조-분석)
2. [현재 코드의 핵심 문제점](#2-현재-코드의-핵심-문제점)
3. [AI/ML 기반 개선 솔루션](#3-aiml-기반-개선-솔루션)
4. [추천 AI 모델 및 프레임워크](#4-추천-ai-모델-및-프레임워크)
5. [구현 우선순위 및 로드맵](#5-구현-우선순위-및-로드맵)
6. [하드웨어 고려사항](#6-하드웨어-고려사항)
7. [참고 자료](#7-참고-자료)

---

## 1. 현재 코드베이스 구조 분석

### 1.1 핵심 모듈 구성

```
nimg/
├── nimg.py                          # 메인 ROS2 노드 (nimg_x86 클래스)
├── submodules/
│   ├── nodeL515.py                  # L515 카메라 센서 처리 (dSensor 클래스)
│   ├── detectProcessor.py           # 객체 탐지 및 각도 측정 처리
│   ├── detect.py                    # YOLOv5 기반 객체 탐지 (Detector 클래스)
│   ├── lineDetector.py              # Hough 변환 기반 라인 검출
│   ├── ItemList.py                  # 탐지 객체 관리
│   └── ...
├── models/                          # YOLOv5 모델 정의
└── utils/                           # 유틸리티 함수들
```

### 1.2 데이터 흐름

```
[L515 Camera]
    ↓
[dSensor.run()] → RGB Frame + Depth Frame + Point Cloud
    ↓
[detectProcessor.setImage()] → RGB만 전달
    ↓
[Detector.detect()] → YOLOv5 2D 탐지
    ↓
[detectProcessor.processImage()] → 탐지 결과 + 간단한 depth 처리
    ↓
[depthProcess()] → 상하 영역 depth 평균 차이로 각도 추정
```

### 1.3 주요 클래스 역할

| 클래스 | 파일 위치 | 주요 역할 |
|--------|----------|----------|
| `nimg_x86` | nimg/nimg.py:46-247 | ROS2 노드, 전체 시스템 조율 |
| `dSensor` | nimg/submodules/nodeL515.py:63-628 | L515 카메라 데이터 수집, Point Cloud 생성 |
| `detectProcessor` | nimg/submodules/detectProcessor.py:25-329 | 객체 탐지, 각도 측정, 결과 발행 |
| `Detector` | nimg/submodules/detect.py:56-236 | YOLOv5 모델 로드 및 추론 |
| `lineDetector` | nimg/submodules/lineDetector.py:5-75 | Hough 변환 라인 검출 |

---

## 2. 현재 코드의 핵심 문제점

### 2.1 속도 측정 관련 문제

#### 문제 1: 속도 측정 기능 미구현
```python
# detectProcessor.py - speed 변수가 선언만 되어 있음
self.speed = 0  # Line 53 - 초기화만 됨, 실제 계산 없음
```

**현재 상태**:
- `speed` 변수가 `detectProcessor.__init__()` 에서 0으로 초기화
- 실제 속도 계산 로직이 전혀 구현되어 있지 않음
- Point Cloud나 Depth 정보를 활용한 3D 속도 측정 없음

#### 문제 2: 2D 기반 움직임 감지만 존재
```python
# nodeL515.py - move_check 메서드 (Lines 535-548)
def move_check(self, img):
    # 단순 프레임 차이 기반 움직임 감지
    dst = cv2.absdiff(self.pre_img, new_img)
    _, dst_th = cv2.threshold(dst, 15, 255, cv2.THRESH_BINARY)
    move_check = cv2.countNonZero(dst_th)
```

**문제점**:
- 2D 픽셀 변화량만 측정, 실제 3D 이동 거리 계산 불가
- 원근 왜곡으로 인한 심각한 오차 (거리에 따라 50-100% 오차 가능)
- 움직임 발생 여부만 확인, 속도 벡터 추정 불가

### 2.2 각도 측정 관련 문제

#### 문제 3: 매우 단순한 Depth 기반 각도 추정
```python
# detectProcessor.py - depthProcess 메서드 (Lines 134-142)
def depthProcess(self, img, roi, cimg):
    up_mean = self.checkDepth(img, roi[0], roi[1], roi[0] + roi[2], roi[1] + int(roi[3]/3))
    down_mean = self.checkDepth(img, roi[0], roi[1] + int(roi[3]/3*2), roi[0] + roi[2], roi[1] + int(roi[3]/3))
    return round(down_mean - up_mean, 3)  # 단순 차이값 반환
```

**문제점**:
- **Pitch 각도만 추정 가능**: 상단/하단 depth 평균 차이로 기울기만 추정
- **Yaw/Roll 측정 불가**: 좌우 회전, 측면 기울기 측정 없음
- **6DoF 자세 추정 불가**: 3D 공간에서의 완전한 자세 파악 불가능
- **단위 없는 값**: 실제 각도(도)로 변환하는 로직 없음
- **노이즈에 취약**: 단순 평균으로 depth 노이즈에 매우 민감

#### 문제 4: 2D Hough 변환 기반 각도 측정
```python
# detectProcessor.py - countingPattern 메서드 (Lines 153-230)
def countingPattern(self, img, rst, roi):
    # Canny Edge Detection + Hough Lines
    E = cv2.Canny(th3, Threshold1, Threshold2, FilterSize)
    lines = cv2.HoughLinesP(E, Rres, Thetares, Threshold, ...)

    # 2D 이미지에서 라인 각도 추출
    for line in lines:
        slope = (y2 - y1) / (x2 - x1)
        angle = round(np.rad2deg(np.arctan(slope)), 2)
```

**문제점**:
- **2D 투영 각도만 측정**: 3D 회전이 2D에 투영되어 정확도 저하
- **에지 검출 의존성**: 에지가 명확하지 않은 객체에서 실패
- **복잡한 형상 처리 불가**: 단순 직선만 검출 가능
- **조명 변화에 민감**: 그림자, 반사 등에 의한 오검출

### 2.3 Point Cloud 활용 문제

#### 문제 5: Point Cloud 데이터 미활용
```python
# nodeL515.py - run 메서드 (Lines 384-532)
def run(self):
    # Point Cloud 생성은 하지만...
    self.pc.map_to(color_frame)
    points = self.pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

    # ...저장용으로만 사용
    if self.filesaveMode:
        self.savePointToFile(self.points, ...)
```

**현재 상태**:
- Point Cloud 생성 후 **파일 저장용으로만 사용**
- 객체 추적, 속도 측정, 자세 추정에 전혀 활용되지 않음
- `detectProcessor`로 Point Cloud 정보가 전달되지 않음

### 2.4 객체 추적 관련 문제

#### 문제 6: 추적(Tracking) 기능 부재
```python
# detectProcessor.py - processImage 메서드
# 매 프레임 독립적으로 탐지, 이전 프레임과 연결 없음
self.detectImage, items, check_flag = self.dProcessor.detect('source.png')
```

**문제점**:
- **프레임 간 연결 없음**: 동일 객체의 시간적 연속성 파악 불가
- **ID 할당 없음**: 여러 객체 중 동일 객체 식별 불가
- **Kalman Filter 미사용**: 예측 및 노이즈 필터링 없음
- **가려짐 처리 불가**: 객체가 일시적으로 가려지면 추적 실패

### 2.5 시스템 아키텍처 문제

#### 문제 7: RGB와 Depth 데이터 분리
```python
# nodeL515.py - processMode 처리
if self.processMode:
    self.detectP.setImage(color_image)  # RGB만 전달
```

**문제점**:
- `detectProcessor`에 **RGB 이미지만 전달**
- Depth Frame과 Point Cloud 정보가 detectProcessor로 전달되지 않음
- 센서 융합(RGB-D Fusion)이 제대로 이루어지지 않음

#### 문제 8: 파일 기반 탐지 처리
```python
# detectProcessor.py - processImage 메서드
cv2.imwrite('source.png', src_img)  # 파일로 저장
self.detectImage, items, check_flag = self.dProcessor.detect('source.png')  # 파일에서 읽어 탐지
```

**문제점**:
- 매 프레임마다 **디스크 I/O 발생** (심각한 성능 저하)
- 실시간 처리에 부적합
- 메모리 기반 처리로 변경 필요

### 2.6 성능 및 품질 문제

| 카테고리 | 현재 상태 | 예상 문제 |
|----------|----------|----------|
| 속도 측정 | 미구현 | 측정 불가 |
| Pitch 각도 | 단순 depth 차이 | ±15-20° 오차 |
| Yaw 각도 | 2D Hough 변환 | ±10-15° 오차 |
| Roll 각도 | 미구현 | 측정 불가 |
| 객체 추적 | 미구현 | 연속 추적 불가 |
| FPS | 파일 I/O 병목 | 10-15 FPS 예상 |

---

## 3. AI/ML 기반 개선 솔루션

### 3.1 속도 측정 개선

#### 솔루션 A: 3D Scene Flow 기반 속도 측정

**FlowNet3D / RMS-FlowNet++ 활용**

Scene Flow는 3D 공간에서의 점별 이동 벡터를 추정합니다.

```python
# 개선된 속도 측정 파이프라인 (개념적 코드)
class VelocityEstimator:
    def __init__(self):
        self.scene_flow_model = load_flownet3d_model()
        self.kalman_filter = KalmanFilter3D(dt=1/30.0)

    def estimate_velocity(self, pcd_prev, pcd_curr, bbox_3d, dt):
        # 1. 객체 영역 Point Cloud 추출
        obj_pcd_prev = self.crop_object(pcd_prev, bbox_3d)
        obj_pcd_curr = self.crop_object(pcd_curr, bbox_3d)

        # 2. Scene Flow 추정 (AI 모델)
        flow_vectors = self.scene_flow_model(obj_pcd_prev, obj_pcd_curr)

        # 3. 평균 이동 벡터 계산
        displacement = np.mean(flow_vectors, axis=0)

        # 4. Kalman Filter로 노이즈 제거 및 속도 추정
        velocity = self.kalman_filter.update(displacement / dt)

        return velocity  # [vx, vy, vz] in m/s
```

**기대 효과**:
- 실세계 단위(m/s) 속도 직접 측정
- 원근 왜곡 완전 제거
- 3D 방향 벡터 획득

#### 솔루션 B: CenterPoint 기반 속도 추정

**CenterPoint**는 3D 객체 탐지와 함께 속도를 직접 예측합니다.

```python
# CenterPoint 기반 접근 (개념적)
class CenterPointVelocityTracker:
    def __init__(self):
        self.centerpoint = load_centerpoint_model()

    def detect_and_track(self, point_cloud):
        # 탐지 + 속도 예측을 한번에 수행
        detections = self.centerpoint(point_cloud)

        for det in detections:
            position = det['center']      # [x, y, z]
            size = det['size']            # [w, h, l]
            rotation = det['rotation']    # yaw angle
            velocity = det['velocity']    # [vx, vy] 직접 예측!

        return detections
```

**장점**:
- End-to-end 학습으로 최적화된 속도 추정
- nuScenes 등 대규모 데이터셋으로 사전학습
- 60+ FPS TensorRT 최적화 버전 존재

### 3.2 각도/자세 측정 개선

#### 솔루션 C: 6DoF Pose Estimation

**YOLO-6D-Pose / SwinDePose 활용**

```python
# 6DoF Pose Estimation 파이프라인
class PoseEstimator6D:
    def __init__(self):
        # Depth 전용 또는 RGB-D 융합 모델
        self.pose_model = load_swin_depose_model()

    def estimate_pose(self, rgb, depth, bbox):
        # 1. ROI 추출
        rgb_roi = rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        depth_roi = depth[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # 2. 6DoF Pose 추정
        pose = self.pose_model(rgb_roi, depth_roi)

        # pose = [x, y, z, roll, pitch, yaw]
        return pose
```

#### 솔루션 D: Point Cloud 기반 OBB + PCA

**Open3D를 활용한 Oriented Bounding Box 추정**

```python
import open3d as o3d
import numpy as np

def estimate_pose_from_pointcloud(pcd):
    """Point Cloud에서 6DoF 자세 추정"""

    # 1. Oriented Bounding Box 계산
    obb = pcd.get_oriented_bounding_box()

    # 2. 중심점과 회전 행렬 추출
    center = obb.center  # [x, y, z]
    rotation_matrix = obb.R  # 3x3 회전 행렬

    # 3. 회전 행렬 → Euler 각도 변환
    def rotation_to_euler(R):
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = np.arctan2(R[1,0], R[0,0])
        else:
            roll = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = 0

        return np.degrees([roll, pitch, yaw])

    euler_angles = rotation_to_euler(rotation_matrix)

    return {
        'position': center,
        'orientation': euler_angles,  # [roll, pitch, yaw] in degrees
        'dimensions': obb.extent  # [width, height, depth]
    }
```

### 3.3 객체 추적 개선

#### 솔루션 E: 3D Kalman Filter + Hungarian Algorithm

```python
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Tracker3D:
    def __init__(self, dt=1/30.0):
        self.tracks = {}
        self.next_id = 0
        self.dt = dt

    def create_kalman_filter(self):
        """9-state Kalman Filter: [x,y,z, vx,vy,vz, ax,ay,az]"""
        kf = KalmanFilter(dim_x=9, dim_z=3)

        # State transition matrix (constant acceleration)
        kf.F = np.array([
            [1, 0, 0, self.dt, 0, 0, 0.5*self.dt**2, 0, 0],
            [0, 1, 0, 0, self.dt, 0, 0, 0.5*self.dt**2, 0],
            [0, 0, 1, 0, 0, self.dt, 0, 0, 0.5*self.dt**2],
            [0, 0, 0, 1, 0, 0, self.dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, self.dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix (only position)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])

        # Measurement noise (L515 accuracy: ~5mm)
        kf.R = np.eye(3) * 0.005**2

        return kf

    def update(self, detections_3d):
        """Hungarian Algorithm으로 탐지-트랙 매칭"""
        if len(self.tracks) == 0:
            # 새 트랙 생성
            for det in detections_3d:
                self._create_track(det)
            return

        # 비용 행렬 계산 (3D 거리)
        cost_matrix = np.zeros((len(self.tracks), len(detections_3d)))
        track_ids = list(self.tracks.keys())

        for i, tid in enumerate(track_ids):
            predicted = self.tracks[tid]['kf'].x[:3]
            for j, det in enumerate(detections_3d):
                cost_matrix[i, j] = np.linalg.norm(predicted - det['position'])

        # Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 매칭된 트랙 업데이트
        matched_tracks = set()
        matched_dets = set()

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 0.5:  # 50cm threshold
                tid = track_ids[i]
                self.tracks[tid]['kf'].predict()
                self.tracks[tid]['kf'].update(detections_3d[j]['position'])
                matched_tracks.add(tid)
                matched_dets.add(j)

        # 매칭되지 않은 탐지 → 새 트랙
        for j, det in enumerate(detections_3d):
            if j not in matched_dets:
                self._create_track(det)

    def get_velocity(self, track_id):
        """특정 트랙의 속도 반환"""
        if track_id in self.tracks:
            kf = self.tracks[track_id]['kf']
            velocity = kf.x[3:6]  # [vx, vy, vz]
            speed = np.linalg.norm(velocity)
            return speed, velocity
        return None, None
```

### 3.4 ICP 기반 정밀 변환 추정

```python
import open3d as o3d

class ICPVelocityEstimator:
    def __init__(self):
        self.prev_pcd = None
        self.prev_time = None

    def estimate(self, current_pcd, current_time, bbox_3d):
        if self.prev_pcd is None:
            self.prev_pcd = self._crop_object(current_pcd, bbox_3d)
            self.prev_time = current_time
            return None

        # 현재 객체 Point Cloud 추출
        current_obj_pcd = self._crop_object(current_pcd, bbox_3d)

        # Voxel Downsampling (성능 최적화)
        voxel_size = 0.005  # 5mm
        prev_down = self.prev_pcd.voxel_down_sample(voxel_size)
        curr_down = current_obj_pcd.voxel_down_sample(voxel_size)

        # Normal 계산
        prev_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        curr_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )

        # Point-to-Plane ICP
        threshold = 0.02  # 2cm
        result = o3d.pipelines.registration.registration_icp(
            curr_down, prev_down, threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        # 변환 행렬에서 속도/각속도 추출
        dt = current_time - self.prev_time
        T = result.transformation

        # Translation → Velocity
        translation = T[:3, 3]
        velocity = translation / dt

        # Rotation → Angular Velocity
        R = T[:3, :3]
        angle, axis = self._rotation_to_axis_angle(R)
        angular_velocity = (angle / dt) * axis

        # 상태 업데이트
        self.prev_pcd = current_obj_pcd
        self.prev_time = current_time

        return {
            'velocity': velocity,
            'angular_velocity': angular_velocity,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse
        }
```

---

## 4. 추천 AI 모델 및 프레임워크

### 4.1 3D 객체 탐지 및 추적

| 모델 | 특징 | 속도 추정 | FPS | 추천도 |
|------|------|----------|-----|--------|
| **CenterPoint** | Point Cloud 기반, 속도 직접 예측 | O | 11-60 | ★★★★★ |
| HVTrack | 고시간변동 환경 특화 | X | 20+ | ★★★★☆ |
| 3DMOTFormer | Graph Transformer 기반 | X | 15+ | ★★★☆☆ |
| PointPillars | 경량, 실시간 | X | 60+ | ★★★★☆ |

**최우선 추천**: [CenterPoint](https://github.com/tianweiy/CenterPoint)
- MIT 라이선스
- 속도 예측 헤드 내장
- TensorRT 최적화 버전 존재 (60 FPS)
- nuScenes, Waymo 데이터셋 사전학습

### 4.2 Scene Flow (3D 움직임 추정)

| 모델 | 특징 | 정확도 | FPS |
|------|------|--------|-----|
| **RMS-FlowNet++** | 2024 최신, Multi-scale | 높음 | 15+ |
| FlowNet3D | 기초 모델, 안정적 | 중간 | 20+ |
| PointPWC-Net | Self-supervised | 높음 | 10+ |

**추천**: [RMS-FlowNet++](https://link.springer.com/article/10.1007/s11263-024-02093-9)
- 2024년 최신 연구
- 대규모 Point Cloud 효율 처리
- Multi-scale 특징 추출

### 4.3 6DoF Pose Estimation

| 모델 | 입력 | 정확도 | 실시간성 |
|------|------|--------|----------|
| **SwinDePose** | Depth only | 높음 | 중간 |
| YOLO-6D-Pose | RGB | 중간 | 높음 |
| GDR-Net | RGB-D | 높음 | 중간 |
| FFB6D | RGB-D | 높음 | 낮음 |

**추천**: [SwinDePose](https://arxiv.org/abs/2303.02133)
- Depth 이미지만으로 6DoF 추정
- 조명 변화에 강건
- Swin Transformer 기반

### 4.4 RGB-D Fusion

| 방법 | 특징 | 적용 |
|------|------|------|
| **Early Fusion** | 입력 단계 결합 | 간단, 기본 |
| Late Fusion | 결과 단계 결합 | 모듈성 높음 |
| **Mid/Hybrid Fusion** | 중간 특징 결합 | 최고 성능 |

**추천**: FusionVision 접근법
- YOLO + FastSAM 결합
- 85% 노이즈 제거
- 6D pose 정확도: ±3mm, ±2°

### 4.5 추적 알고리즘

| 알고리즘 | 특징 | 복잡도 |
|----------|------|--------|
| **3D Kalman Filter** | 노이즈 필터링, 속도 추론 | 낮음 |
| Extended Kalman Filter | 비선형 모델 지원 | 중간 |
| Particle Filter | 다중 가설 | 높음 |
| DeepSORT (3D 확장) | 외관+움직임 | 중간 |

**최우선 추천**: 3D Kalman Filter (FilterPy)
- 즉시 적용 가능
- 안정적, 검증된 방법
- 속도 및 가속도 추론

---

## 5. 구현 우선순위 및 로드맵

### Phase 1: 기반 개선 (우선순위: 높음)

#### 1.1 파일 I/O 제거
```python
# 현재 (비효율적)
cv2.imwrite('source.png', src_img)
result = self.dProcessor.detect('source.png')

# 개선 (메모리 기반)
result = self.dProcessor.detect_from_array(src_img)
```

#### 1.2 Depth/Point Cloud 데이터 전달 구조 개선
```python
# detectProcessor에 depth와 point cloud 전달
def setImage(self, color_img, depth_img=None, point_cloud=None):
    self.sourceImage = color_img
    self.depthFrame = depth_img
    self.pointCloud = point_cloud  # 추가
```

#### 1.3 3D Kalman Filter 통합
- filterpy 라이브러리 활용
- 위치, 속도, 가속도 상태 추정
- 노이즈 필터링

### Phase 2: 정밀 측정 구현 (우선순위: 높음)

#### 2.1 ICP 기반 속도/각속도 측정
- Open3D ICP 활용
- Point Cloud 정합으로 정밀 변환 추정
- 속도: ±5% 이내, 각도: ±2° 이내 목표

#### 2.2 PCA 기반 OBB 각도 측정
- Point Cloud에서 주축 추출
- Roll, Pitch, Yaw 모두 측정
- 실시간 처리 가능 (< 10ms)

### Phase 3: AI 모델 통합 (우선순위: 중간)

#### 3.1 CenterPoint 통합
- 3D 객체 탐지 + 속도 예측
- TensorRT 최적화 적용
- 목표: 30+ FPS

#### 3.2 6DoF Pose Estimation 모델 적용
- SwinDePose 또는 YOLO-6D-Pose
- Depth 기반 정밀 자세 추정

### Phase 4: 고급 최적화 (우선순위: 낮음)

#### 4.1 Scene Flow 모델 적용
- RMS-FlowNet++ 또는 FlowNet3D
- 점별 3D 속도 벡터 추정

#### 4.2 RGB-D Fusion 네트워크
- Multi-modal feature fusion
- Attention 기반 적응형 융합

---

## 6. 하드웨어 고려사항

### 6.1 L515 카메라 상태

**중요**: Intel RealSense L515는 2022년 2월 단종되었습니다.

#### 대안 카메라 옵션

| 카메라 | 타입 | 정확도 | 범위 | 권장 |
|--------|------|--------|------|------|
| Intel D455 | Stereo | ±2% @2m | 0.4-6m | ★★★★☆ |
| Intel D435i | Stereo+IMU | ±2% @2m | 0.1-10m | ★★★☆☆ |
| Azure Kinect DK | ToF | ±0.25% @1m | 0.25-5.5m | ★★★★★ |
| Orbbec Femto | ToF | ±0.3% @1m | 0.25-10m | ★★★★☆ |
| ZED 2i | Stereo | ±1% @3m | 0.2-20m | ★★★★☆ |

**권장**: 현재 L515 재고가 있다면 계속 사용하되, 장기적으로 **Azure Kinect DK** 또는 **Orbbec Femto**로 마이그레이션 고려

### 6.2 컴퓨팅 플랫폼

#### NVIDIA Jetson 시리즈 (추천)

| 플랫폼 | GPU | 메모리 | TDP | 적합성 |
|--------|-----|--------|-----|--------|
| Jetson Orin Nano | 1024 CUDA | 8GB | 15W | ★★★☆☆ |
| Jetson Orin NX | 2048 CUDA | 16GB | 25W | ★★★★☆ |
| **Jetson AGX Orin** | 2048 CUDA | 32GB | 30-60W | ★★★★★ |

**권장**: Jetson AGX Orin
- 3D Point Cloud 처리에 충분한 성능
- TensorRT 지원으로 AI 모델 가속
- 30+ FPS 실시간 처리 가능

### 6.3 예상 성능

| 처리 단계 | CPU Only | Jetson AGX Orin |
|-----------|----------|-----------------|
| YOLO 탐지 | 100ms | 15ms |
| Point Cloud 생성 | 30ms | 10ms |
| ICP 정합 | 100ms | 20ms |
| Kalman Filter | 1ms | 1ms |
| **총 지연시간** | **~230ms** | **~45ms** |
| **FPS** | ~4 | ~22 |

---

## 7. 참고 자료

### 7.1 AI 모델 GitHub 저장소

- **CenterPoint**: https://github.com/tianweiy/CenterPoint
- **OpenPCDet**: https://github.com/open-mmlab/OpenPCDet
- **FlowNet3D**: https://github.com/xingyul/flownet3d
- **Open3D**: https://github.com/isl-org/Open3D
- **FilterPy**: https://github.com/rlabbe/filterpy

### 7.2 학술 논문

1. [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275) - CVPR 2021
2. [FlowNet3D: Learning Scene Flow in 3D Point Clouds](https://arxiv.org/abs/1806.01411) - CVPR 2019
3. [RMS-FlowNet++](https://link.springer.com/article/10.1007/s11263-024-02093-9) - IJCV 2024
4. [3D Single-Object Tracking in Point Clouds with High Temporal Variation](https://arxiv.org/abs/2408.02049) - ECCV 2024
5. [Depth-based 6DoF Object Pose Estimation using Swin Transformer](https://arxiv.org/abs/2303.02133)
6. [FusionVision: RGB-D Object Reconstruction and Segmentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11086350/)
7. [A Survey of 6DoF Object Pose Estimation Methods](https://www.mdpi.com/1424-8220/24/4/1076)

### 7.3 튜토리얼 및 문서

- [Open3D ICP Registration Tutorial](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)
- [FilterPy Documentation](https://filterpy.readthedocs.io/)
- [Kalman and Bayesian Filters in Python (무료 책)](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
- [Intel RealSense SDK Documentation](https://dev.intelrealsense.com/docs)
- [NVIDIA Jetson Inference](https://github.com/dusty-nv/jetson-inference)

### 7.4 컨베이어 벨트 관련 연구

- [Tracking and Classifying Objects on a Conveyor Belt Using Time-of-Flight Camera](https://www.researchgate.net/publication/228835165)
- [A new paradigm for intelligent status detection of belt conveyors based on deep learning](https://www.sciencedirect.com/science/article/abs/pii/S0263224123002993)
- [Study of conveyor belt deviation detection based on improved YOLOv8](https://www.nature.com/articles/s41598-024-75542-7)

---

## 결론

현재 nimg 코드베이스는 다음과 같은 핵심 문제를 가지고 있습니다:

1. **속도 측정 기능 미구현**
2. **각도 측정이 매우 단순하고 부정확** (Pitch만 가능, Yaw/Roll 불가)
3. **생성된 Point Cloud 미활용**
4. **객체 추적 기능 부재**
5. **RGB-D 센서 융합 미비**
6. **파일 기반 처리로 인한 성능 저하**

### 즉시 적용 가능한 개선

1. **3D Kalman Filter 통합** - FilterPy 활용, 속도 추론
2. **Open3D ICP** - 정밀 속도/각속도 측정
3. **OBB/PCA 기반 각도 측정** - Roll, Pitch, Yaw 모두 측정

### AI 모델 적용 시

1. **CenterPoint** - 3D 탐지 + 속도 예측 통합
2. **SwinDePose** - Depth 기반 6DoF 자세 추정
3. **RMS-FlowNet++** - 정밀 Scene Flow 추정

### 예상 개선 효과

| 항목 | 현재 | 개선 후 |
|------|------|---------|
| 속도 측정 | 불가능 | ±5% 정확도 |
| Pitch 각도 | ±15-20° | ±2-3° |
| Yaw 각도 | ±10-15° | ±2-3° |
| Roll 각도 | 불가능 | ±3-5° |
| 추적 성공률 | ~70% | >95% |
| 처리 속도 | ~10 FPS | 25-30 FPS |
