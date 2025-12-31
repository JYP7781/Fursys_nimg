# nimg_v2 파이프라인 테스트 보고서

**테스트 시간**: 2025-12-24 12:56:11
**완료 시간**: 2025-12-24 12:56:20

---

## 1. 테스트 환경

| 항목 | 값 |
|------|-----|
| AI 모델 | `class187_image85286_v12x_250epochs.pt` |
| 데이터 소스 | 20240525, 20240529 |
| 카메라 | Intel RealSense D455 (시뮬레이션) |

### 카메라 내부 파라미터
- fx: 636.3393
- fy: 636.4266
- cx: 654.3418
- cy: 399.5896

---

## 2. 테스트 결과 요약

### 2.1 객체 탐지

| 지표 | 값 |
|------|-----|
| 총 탐지 수 | 5 |
| 고유 클래스 | 2 |
| 평균 신뢰도 | 0.541 |
| 평균 추론 시간 | 25.4 ms |

### 2.2 Kalman Filter 추적

| 지표 | 값 |
|------|-----|
| 테스트 프레임 | 99 |
| 평균 위치 오차 | 14.91 mm |
| 최대 위치 오차 | 38.26 mm |
| 최종 속도 (X) | 0.3682 m/s |

### 2.3 변화량 측정

| 지표 | 값 |
|------|-----|
| 테스트 프레임 | 30 |
| 최대 거리 변화 | 0.300 m |
| 최대 회전량 | 18.49 deg |

---

## 3. 생성된 파일

### 데이터 파일 (data/)
- `detection_results.csv` - 객체 탐지 결과
- `position_results.csv` - 3D 위치 추정 결과
- `orientation_results.csv` - 방향 추정 결과
- `kalman_results.csv` - Kalman Filter 추적 결과
- `change_results.csv` - 변화량 측정 결과
- `test_results_summary.json` - 전체 요약

### 시각화 (plots/)
- `detection_stats.png` - 탐지 통계 차트
- `kalman_tracking.png` - Kalman Filter 추적 그래프
- `change_over_time.png` - 시간별 변화량 그래프
- `position_distribution.png` - 위치 분포 그래프

### 탐지 이미지 (images/)
- 탐지된 객체가 표시된 이미지들

### 로그 (logs/)
- 테스트 실행 로그 파일

---

## 4. 결론

nimg_v2 파이프라인의 모든 핵심 컴포넌트가 정상 동작합니다:

1. **YOLODetector**: 원본 데이터에서 가구 제품 탐지 성공
2. **PositionEstimator**: 3D 위치 추정 정상 동작
3. **OrientationEstimator**: PCA 기반 방향 추정 정상 동작
4. **KalmanFilter3D**: 속도/위치 추적 정확도 우수 (오차 ~15mm)
5. **ChangeCalculator**: 기준 대비 변화량 계산 정상 동작

---

*보고서 자동 생성: nimg_v2 테스트 시스템*
