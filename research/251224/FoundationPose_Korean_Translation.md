# FoundationPose: 새로운 객체의 통합 6D 자세 추정 및 추적

**Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield**

**NVIDIA**

---

## 초록

본 논문에서는 6D 객체 자세 추정 및 추적을 위한 통합 기반 모델인 FoundationPose를 제시한다. 이 모델은 모델 기반(model-based)과 모델 프리(model-free) 설정을 모두 지원한다. 우리의 접근 방식은 CAD 모델이 제공되거나 소수의 참조 이미지가 촬영되는 한, 미세 조정(fine-tuning) 없이 테스트 시점에 새로운 객체에 즉시 적용될 수 있다. 통합 프레임워크 덕분에 하위 자세 추정 모듈은 두 설정에서 동일하며, CAD 모델이 없을 때는 효율적인 새로운 뷰 합성을 위해 신경 암시적 표현(neural implicit representation)을 사용한다. 강력한 일반화 능력은 대규모 합성 훈련을 통해 달성되며, 대규모 언어 모델(LLM), 새로운 트랜스포머 기반 아키텍처, 대조 학습(contrastive learning) 공식화의 도움을 받는다. 도전적인 시나리오와 객체를 포함하는 여러 공개 데이터셋에 대한 광범위한 평가는 우리의 통합 접근 방식이 각 작업에 특화된 기존 방법들을 큰 차이로 능가함을 보여준다. 또한, 가정을 줄였음에도 불구하고 인스턴스 수준 방법과 비교할 만한 결과를 달성한다.

**프로젝트 페이지:** https://nvlabs.github.io/FoundationPose/

---

## 1. 서론

객체에서 카메라로의 강체 6D 변환을 계산하는 것, 즉 객체 자세 추정은 로봇 조작[29, 63, 64] 및 혼합 현실[40]과 같은 다양한 응용 분야에 매우 중요하다. 고전적인 방법들[19, 20, 30, 47, 62]은 훈련 시점에 결정된 특정 객체 인스턴스에서만 작동하기 때문에 인스턴스 수준(instance-level)이라고 알려져 있다. 이러한 방법들은 일반적으로 훈련 데이터 생성을 위해 텍스처가 있는 CAD 모델이 필요하며, 테스트 시점에 보지 못한 새로운 객체에는 적용할 수 없다. 카테고리 수준 방법들[5, 32, 54, 58, 69]은 이러한 가정(인스턴스별 훈련 및 CAD 모델)을 제거하지만, 훈련된 미리 정의된 카테고리 내의 객체로 제한된다. 더욱이, 카테고리 수준 훈련 데이터를 얻는 것은 적용해야 하는 추가적인 자세 정규화 및 검사 단계[58] 때문에 매우 어려운 것으로 알려져 있다.

이러한 한계를 해결하기 위해, 최근의 노력은 임의의 새로운 객체에 대한 즉각적인 자세 추정 문제에 초점을 맞추고 있다[18, 31, 38, 50, 53]. 테스트 시점에 어떤 정보가 사용 가능한지에 따라 두 가지 다른 설정이 고려된다: **모델 기반(model-based)**은 객체의 텍스처가 있는 3D CAD 모델이 제공되는 경우이고, **모델 프리(model-free)**는 객체의 참조 이미지 세트가 제공되는 경우이다. 두 설정 모두에서 많은 진전이 있었지만, 서로 다른 실제 응용 프로그램이 서로 다른 유형의 정보를 제공하기 때문에 두 설정을 통합된 방식으로 다루는 단일 방법이 필요하다.

단일 프레임 객체 자세 추정과 직교하여, 자세 추적 방법[8, 28, 34, 37, 51, 57, 61, 66]은 시간적 단서를 활용하여 비디오 시퀀스에서 보다 효율적이고 부드러우며 정확한 자세 추정을 가능하게 한다. 이러한 방법들은 객체 지식에 대한 가정에 따라 자세 추정의 대응 방법과 유사한 앞서 언급한 문제를 공유한다.

본 논문에서 우리는 RGBD 이미지를 사용하여 모델 기반 및 모델 프리 설정 모두에서 새로운 객체의 자세 추정과 추적을 수행하는 **FoundationPose**라는 통합 프레임워크를 제안한다. 그림 1에서 볼 수 있듯이, 우리의 방법은 이 네 가지 작업 각각에 특화된 기존 최신 방법들을 능가한다. 우리의 강력한 일반화 능력은 대규모 합성 훈련, 대규모 언어 모델(LLM)의 도움, 새로운 트랜스포머 기반 아키텍처 및 대조 학습을 통해 달성된다. 우리는 소수(~16개)의 참조 이미지로 효과적인 새로운 뷰 합성을 가능하게 하는 신경 암시적 표현을 통해 모델 기반과 모델 프리 설정 간의 격차를 해소하며, 이전의 렌더 앤 비교(render-and-compare) 방법[31, 34, 61]보다 훨씬 빠른 렌더링 속도를 달성한다.

### 우리의 기여는 다음과 같이 요약할 수 있다:

- 모델 기반 및 모델 프리 설정을 모두 지원하는 새로운 객체의 자세 추정 및 추적을 위한 통합 프레임워크를 제시한다. 효과적인 새로운 뷰 합성을 위한 객체 중심 신경 암시적 표현이 두 설정 간의 격차를 해소한다.

- 다양한 텍스처 증강을 통해 3D 훈련 자산의 다양성을 확장하는 LLM 지원 합성 데이터 생성 파이프라인을 제안한다.

- 트랜스포머 기반 네트워크 아키텍처와 대조 학습 공식화의 새로운 설계는 합성 데이터만으로 훈련할 때 강력한 일반화를 이끈다.

- 우리의 방법은 여러 공개 데이터셋에서 각 작업에 특화된 기존 방법들을 큰 차이로 능가한다. 가정을 줄였음에도 불구하고 인스턴스 수준 방법과 비교할 만한 결과를 달성한다.

이 연구에서 개발된 코드와 데이터는 공개될 예정이다.

---

## 2. 관련 연구

### CAD 모델 기반 객체 자세 추정

인스턴스 수준 자세 추정 방법[19, 20, 30, 47]은 객체에 대한 텍스처가 있는 CAD 모델이 주어진다고 가정한다. 훈련과 테스트는 정확히 동일한 인스턴스에서 수행된다. 객체 자세는 종종 직접 회귀[35, 67], 또는 PnP가 뒤따르는 2D-3D 대응 구축[47, 55], 또는 최소 제곱 피팅이 뒤따르는 3D-3D 대응[19, 20]으로 해결된다. 객체 지식에 대한 가정을 완화하기 위해, 카테고리 수준 방법[5, 32, 54, 58, 69, 71]은 동일한 카테고리의 새로운 객체 인스턴스에 적용될 수 있지만, 미리 정의된 카테고리를 넘어 임의의 새로운 객체로 일반화할 수 없다. 이러한 한계를 해결하기 위해, 최근의 노력[31, 50]은 테스트 시점에 CAD 모델이 제공되는 한 임의의 새로운 객체의 즉각적인 자세 추정을 목표로 한다.

### 퓨샷 모델 프리 객체 자세 추정

모델 프리 방법은 명시적인 텍스처가 있는 모델의 요구 사항을 제거한다. 대신, 대상 객체를 촬영한 여러 참조 이미지가 제공된다[18, 21, 48, 53]. RLLG[3]와 NeRF-Pose[33]는 객체 CAD 모델 없이 인스턴스별 훈련을 제안한다. 특히, [33]은 객체 좌표 맵과 마스크에 대한 반지도 학습을 제공하기 위해 신경 방사장(neural radiance field)을 구축한다. 다르게, 우리는 모델 기반과 모델 프리 시나리오 간의 격차를 해소하기 위해 효율적인 RGB 및 깊이 렌더링을 위한 SDF 표현을 기반으로 구축된 신경 객체 필드를 도입한다. 또한, 우리는 [3, 33]의 경우와 달리 이 연구에서 일반화 가능한 새로운 객체 자세 추정에 초점을 맞춘다. 새로운 객체를 처리하기 위해, Gen6D[38]는 검출, 검색 및 정제 파이프라인을 설계한다. 그러나, 분포 외(out-of-distribution) 테스트 세트의 어려움을 피하기 위해 미세 조정이 필요하다. OnePose[53]와 그 확장 OnePose++[18]는 객체 모델링을 위해 구조적 모션(SfM)을 활용하고 대응으로부터 자세를 해결하기 위해 2D-3D 매칭 네트워크를 사전 훈련한다. FS6D[21]는 유사한 체계를 채택하고 RGBD 모달리티에 초점을 맞춘다. 그럼에도 불구하고, 대응에 대한 의존성은 텍스처가 없는 객체에 적용되거나 심한 폐색에서 취약해진다.

### 객체 자세 추적

6D 객체 자세 추적은 시간적 단서를 활용하여 비디오 시퀀스에서 보다 효율적이고 부드러우며 정확한 자세 예측을 가능하게 하는 것을 목표로 한다. 신경 렌더링을 통해, 우리의 방법은 높은 효율성으로 자세 추적 작업으로 사소하게 확장될 수 있다. 단일 프레임 자세 추정과 유사하게, 기존 추적 방법은 객체 지식에 대한 가정에 따라 대응 방법으로 분류될 수 있다. 여기에는 인스턴스 수준 방법[8, 11, 34, 61], 카테고리 수준 방법[37, 57], 모델 기반 새로운 객체 추적[28, 51, 66] 및 모델 프리 새로운 객체 추적[60, 65]이 포함된다. 모델 기반 및 모델 프리 설정 모두에서, 우리는 공개 데이터셋에서 새로운 벤치마크 기록을 세우며, 인스턴스 수준 훈련이 필요한 최신 방법[8, 34, 61]까지도 능가한다.

---

## 3. 접근 방법

우리 시스템의 전체 구조는 그림 2에 설명되어 있으며, 다음 하위 섹션에서 설명하는 다양한 구성 요소 간의 관계를 보여준다.

### 3.1. 대규모 언어 지원 데이터 생성

강력한 일반화를 달성하기 위해서는 훈련에 대규모 다양한 객체와 장면이 필요하다. 실제 세계에서 그러한 데이터를 얻고 정확한 6D 자세 실측값을 주석 처리하는 것은 시간과 비용이 많이 든다. 반면에, 합성 데이터는 종종 3D 자산의 크기와 다양성이 부족하다. 우리는 최근 등장한 리소스와 기술인 대규모 3D 모델 데이터베이스[6, 10], 대규모 언어 모델(LLM), 확산 모델[4, 23, 49]을 활용하여 훈련을 위한 새로운 합성 데이터 생성 파이프라인을 개발했다. 이 접근 방식은 이전 연구[21, 25, 31]와 비교하여 데이터의 양과 다양성을 극적으로 확장한다.

#### 3D 자산

우리는 Objaverse[6]와 GSO[10]를 포함한 최근 대규모 3D 데이터베이스에서 훈련 자산을 얻는다. Objaverse[6]의 경우 1156개의 LVIS[13] 카테고리에 속하는 40,000개 이상의 객체로 구성된 Objaverse-LVIS 하위 집합에서 객체를 선택했다. 이 목록에는 합리적인 품질과 형상 및 외관의 다양성을 갖춘 가장 관련성 있는 일상 생활 객체가 포함되어 있다. 또한 각 객체에 대한 카테고리를 설명하는 태그를 제공하여 다음 LLM 지원 텍스처 증강 단계에서 자동 언어 프롬프트 생성에 도움이 된다.

#### LLM 지원 텍스처 증강

대부분의 Objaverse 객체는 고품질 형상을 가지고 있지만, 텍스처 충실도는 크게 다르다. FS6D[21]는 ImageNet[7] 또는 MS-COCO[36]의 이미지를 무작위로 붙여 객체 텍스처를 증강할 것을 제안한다. 그러나, 무작위 UV 매핑으로 인해 이 방법은 결과 텍스처 메시에 이음새와 같은 아티팩트를 생성하고(그림 3 상단), 전체적인 장면 이미지를 객체에 적용하면 비현실적인 결과가 발생한다. 대조적으로, 우리는 더 현실적인(그리고 완전 자동화된) 텍스처 증강을 위해 대규모 언어 모델과 확산 모델의 최근 발전을 어떻게 활용할 수 있는지 탐구한다. 구체적으로, 우리는 텍스트 프롬프트, 객체 형상, 무작위로 초기화된 노이즈 텍스처를 TexFusion[4]에 제공하여 증강된 텍스처 모델을 생성한다. 물론, 다양한 스타일로 많은 객체를 다른 프롬프트 안내 하에 증강하려면 수동으로 이러한 프롬프트를 제공하는 것은 확장할 수 없다. 결과적으로, 우리는 2단계 계층적 프롬프트 전략을 도입한다. 그림 2 좌측 상단에 설명된 것처럼, 먼저 ChatGPT에 프롬프트를 제공하여 객체의 가능한 외관을 설명하도록 요청한다; 이 프롬프트는 템플릿화되어 있어 매번 Objaverse-LVIS 목록에서 제공하는 객체와 쌍을 이루는 태그만 교체하면 된다. 그러면 ChatGPT의 답변이 텍스처 합성을 위해 확산 모델에 제공되는 텍스트 프롬프트가 된다. 이 접근 방식은 텍스처 증강의 완전 자동화를 가능하게 하므로 대규모 다양화된 데이터 생성을 촉진한다. 그림 3은 동일한 객체에 대한 다른 스타일화를 포함한 더 많은 예를 제시한다.

#### 데이터 생성

우리의 합성 데이터 생성은 고충실도 사실적 렌더링을 위한 경로 추적을 활용하여 NVIDIA Isaac Sim에서 구현된다. 우리는 물리적으로 그럴듯한 장면을 생성하기 위해 중력 및 물리 시뮬레이션을 수행한다. 각 장면에서, 원본 및 텍스처 증강 버전을 포함한 객체를 무작위로 샘플링한다. 객체 크기, 재질, 카메라 자세 및 조명도 무작위화된다; 자세한 내용은 부록에서 찾을 수 있다.

### 3.2. 신경 객체 모델링

모델 프리 설정에서 3D CAD 모델을 사용할 수 없을 때, 핵심 과제 중 하나는 하위 모듈에 충분한 품질의 이미지를 효과적으로 렌더링하기 위해 객체를 표현하는 것이다. 신경 암시적 표현은 새로운 뷰 합성에 효과적이고 GPU에서 병렬화 가능하므로, 그림 2에 표시된 것처럼 하위 자세 추정 모듈에 대해 여러 자세 가설을 렌더링할 때 높은 계산 효율성을 제공한다. 이를 위해, 우리는 이전 연구[42, 59, 65, 68]에서 영감을 받아 객체 모델링을 위한 객체 중심 신경 필드 표현을 도입한다.

#### 필드 표현

우리는 그림 2에 표시된 것처럼 두 함수[68]로 객체를 표현한다. 첫째, 기하학 함수 Ω : x ↦ s는 3D 점 x ∈ ℝ³를 입력으로 받아 부호 있는 거리 값 s ∈ ℝ를 출력한다. 둘째, 외관 함수 Φ : (f_Ω(x), n, d) ↦ c는 기하학 네트워크의 중간 특징 벡터 f_Ω(x), 점 법선 n ∈ ℝ³, 시선 방향 d ∈ ℝ³를 받아 색상 c ∈ ℝ³₊를 출력한다. 실제로, 네트워크로 전달하기 전에 x에 다중 해상도 해시 인코딩[42]을 적용한다. n과 d 모두 고정된 2차 구면 조화 계수 집합으로 임베딩된다. 암시적 객체 표면은 부호 있는 거리 필드(SDF)의 영 레벨 셋을 취하여 얻는다: S = {x ∈ ℝ³ | Ω(x) = 0}. NeRF[41]와 비교하여, SDF 표현 Ω는 밀도 임계값을 수동으로 선택할 필요를 제거하면서 더 높은 품질의 깊이 렌더링을 제공한다.

#### 필드 학습

텍스처 학습을 위해, 우리는 절단된 표면 근처 영역에 대한 볼륨 렌더링을 따른다[65]:

$$c(r) = \int_{z(r)-\lambda}^{z(r)+0.5\lambda} w(x_i)\Phi(f_{\Omega(x_i)}, n(x_i), d(x_i)) dt \quad (1)$$

$$w(x_i) = \frac{1}{1+e^{-\alpha\Omega(x_i)}} \cdot \frac{1}{1+e^{\alpha\Omega(x_i)}} \quad (2)$$

여기서 w(x_i)는 점에서 암시적 객체 표면까지의 부호 있는 거리 Ω(x_i)에 의존하는 종 모양의 확률 밀도 함수[59]이고, α는 분포의 부드러움을 조정한다. 확률은 표면 교차점에서 최고점에 달한다. 식 (1)에서, z(r)은 깊이 이미지로부터의 광선의 깊이 값이고, λ는 절단 거리이다. 우리는 더 효율적인 훈련을 위해 표면에서 λ보다 멀리 떨어진 빈 공간의 기여를 무시하고, 자기 폐색을 모델링하기 위해 0.5λ 침투 거리까지만 적분한다[59]. 훈련 중에, 색상 감독을 위해 이 양을 참조 RGB 이미지와 비교한다:

$$\mathcal{L}_c = \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} \|c(r) - \bar{c}(r)\|_2 \quad (3)$$

여기서 c̄(r)은 광선 r이 통과하는 픽셀에서의 실측 색상을 나타낸다.

기하학 학습을 위해, 우리는 공간을 두 영역으로 나누어 SDF를 학습하는 하이브리드 SDF 모델[65]을 채택하여 빈 공간 손실과 표면 근처 손실을 이끈다. 또한 표면 근처 SDF에 아이코날 정규화[12]를 적용한다:

$$\mathcal{L}_e = \frac{1}{|\mathcal{X}_e|} \sum_{x \in \mathcal{X}_e} |\Omega(x) - \lambda| \quad (4)$$

$$\mathcal{L}_s = \frac{1}{|\mathcal{X}_s|} \sum_{x \in \mathcal{X}_s} (\Omega(x) + d_x - d_D)^2 \quad (5)$$

$$\mathcal{L}_{eik} = \frac{1}{|\mathcal{X}_s|} \sum_{x \in \mathcal{X}_s} (\|\nabla\Omega(x)\|_2 - 1)^2 \quad (6)$$

여기서 x는 분할된 공간에서 광선을 따라 샘플링된 3D 점을 나타낸다; d_x와 d_D는 각각 광선 원점에서 샘플 점까지의 거리와 관측된 깊이 점까지의 거리이다. 템플릿 이미지가 모델 프리 설정에서 오프라인으로 사전 캡처되므로 불확실한 자유 공간 손실[65]을 사용하지 않는다. 총 훈련 손실은 다음과 같다:

$$\mathcal{L} = w_c\mathcal{L}_c + w_e\mathcal{L}_e + w_s\mathcal{L}_s + w_{eik}\mathcal{L}_{eik} \quad (7)$$

학습은 사전 지식 없이 객체별로 최적화되며 몇 초 내에 효율적으로 수행될 수 있다. 신경 필드는 새로운 객체에 대해 한 번만 훈련하면 된다.

#### 렌더링

훈련이 완료되면, 신경 필드는 후속 렌더 앤 비교 반복을 위해 객체의 효율적인 렌더링을 수행하는 기존 그래픽 파이프라인의 드롭인 대체물로 사용될 수 있다. 원본 NeRF[41]에서와 같은 색상 렌더링 외에도, RGBD 기반 자세 추정 및 추적을 위해 깊이 렌더링도 필요하다. 이를 위해, 우리는 색상 투영과 결합하여 SDF의 영 레벨 셋에서 텍스처가 있는 메시를 추출하기 위해 마칭 큐브[39]를 수행한다. 이것은 각 객체에 대해 한 번만 수행하면 된다. 추론 시, 객체 자세가 주어지면 래스터화 프로세스를 따라 RGBD 이미지를 렌더링한다. 대안으로, 구 추적(sphere tracing)[14]을 사용하여 온라인으로 Ω를 사용하여 깊이 이미지를 직접 렌더링할 수 있지만, 특히 병렬로 렌더링할 자세 가설이 많을 때 효율성이 떨어지는 것으로 나타났다.

### 3.3. 자세 가설 생성

#### 자세 초기화

RGBD 이미지가 주어지면, 객체는 Mask R-CNN[17] 또는 CNOS[44]와 같은 기성 방법을 사용하여 검출된다. 검출된 2D 바운딩 박스 내 중앙 깊이에 위치한 3D 점을 사용하여 이동을 초기화한다. 회전을 초기화하기 위해, 카메라가 중심을 향하도록 객체를 중심으로 한 정이십면체(icosphere)에서 N_s개의 시점을 균일하게 샘플링한다. 이러한 카메라 자세는 N_i개의 이산화된 평면 내 회전으로 추가 증강되어, 자세 정제기에 입력으로 전송되는 N_s · N_i개의 전역 자세 초기화가 생성된다.

#### 자세 정제

이전 단계의 대략적인 자세 초기화는 종종 상당히 노이즈가 많으므로, 자세 품질을 향상시키기 위한 정제 모듈이 필요하다. 구체적으로, 우리는 대략적인 자세에 조건화된 객체의 렌더링과 카메라로부터의 입력 관측 크롭을 입력으로 받는 자세 정제 네트워크를 구축한다; 네트워크는 자세 품질을 향상시키는 자세 업데이트를 출력한다. 앵커 포인트를 찾기 위해 대략적인 자세 주변의 여러 뷰를 렌더링하는 MegaPose[31]와 달리, 대략적인 자세에 해당하는 단일 뷰 렌더링으로 충분함을 관찰했다. 입력 관측의 경우, 일정한 2D 검출을 기반으로 크롭하는 대신, 이동 업데이트에 대한 피드백을 제공하기 위해 자세 조건화 크롭 전략을 수행한다. 구체적으로, 크롭 중심을 결정하기 위해 객체 원점을 이미지 공간에 투영한다. 그런 다음 자세 가설 주변의 객체와 인근 맥락을 둘러싸는 크롭 크기를 결정하기 위해 약간 확대된 객체 직경(객체 표면의 임의의 점 쌍 사이의 최대 거리)을 투영한다. 따라서 이 크롭은 대략적인 자세에 조건화되어 네트워크가 크롭을 관측과 더 잘 정렬하도록 이동을 업데이트하도록 장려한다. 정제 프로세스는 최신 업데이트된 자세를 다음 추론의 입력으로 공급하여 자세 품질을 반복적으로 향상시키기 위해 여러 번 반복될 수 있다.

정제 네트워크 아키텍처는 그림 2에 설명되어 있다; 자세한 내용은 부록에 있다. 먼저 단일 공유 CNN 인코더로 두 RGBD 입력 브랜치에서 특징 맵을 추출한다. 특징 맵은 연결되어 잔차 연결[16]이 있는 CNN 블록에 공급되고, 위치 임베딩과 함께 패치[9]로 나누어 토큰화된다. 마지막으로, 네트워크는 이동 업데이트 Δt ∈ ℝ³와 회전 업데이트 ΔR ∈ SO(3)를 예측하며, 각각 트랜스포머 인코더[56]에 의해 개별적으로 처리되고 출력 차원으로 선형 투영된다. 더 구체적으로, Δt는 카메라 프레임에서 객체의 이동 변화를 나타내고, ΔR은 카메라 프레임에서 표현된 객체의 방향 업데이트를 나타낸다. 실제로, 회전은 축-각도 표현으로 매개변수화된다. 유사한 결과를 달성하는 6D 표현[72]도 실험했다. 입력 대략적인 자세 [R | t] ∈ SE(3)는 다음과 같이 업데이트된다:

$$\mathbf{t}^+ = \mathbf{t} + \Delta\mathbf{t} \quad (8)$$

$$\mathbf{R}^+ = \Delta\mathbf{R} \otimes \mathbf{R} \quad (9)$$

여기서 ⊗는 SO(3)에서의 업데이트를 나타낸다. 단일 동차 자세 업데이트를 사용하는 대신, 이 분리된 표현은 이동 업데이트를 적용할 때 업데이트된 방향에 대한 의존성을 제거한다. 이것은 업데이트와 입력 관측 모두를 카메라 좌표 프레임에서 통합하여 학습 프로세스를 단순화한다. 네트워크 훈련은 L2 손실로 감독된다:

$$\mathcal{L}_{\text{refine}} = w_1\|\Delta\mathbf{t} - \Delta\bar{\mathbf{t}}\|_2 + w_2\|\Delta\mathbf{R} - \Delta\bar{\mathbf{R}}\|_2 \quad (10)$$

여기서 t̄와 R̄은 실측값이고; w₁과 w₂는 손실의 균형을 맞추는 가중치로, 경험적으로 1로 설정된다.

### 3.4. 자세 선택

정제된 자세 가설 목록이 주어지면, 계층적 자세 순위 네트워크를 사용하여 점수를 계산한다. 최고 점수를 가진 자세가 최종 추정치로 선택된다.

#### 계층적 비교

네트워크는 2단계 비교 전략을 사용한다. 첫째, 각 자세 가설에 대해, 렌더링된 이미지는 섹션 3.3에서 도입된 자세 조건화 크롭 연산을 사용하여 크롭된 입력 관측과 비교된다. 이 비교(그림 2 좌측 하단)는 정제 네트워크에서와 동일한 백본 아키텍처를 특징 추출에 활용하는 자세 순위 인코더로 수행된다. 추출된 특징은 연결되고, 토큰화되어 비교를 위한 전역 이미지 맥락을 더 잘 활용하기 위해 다중 헤드 셀프 어텐션 모듈로 전달된다. 자세 순위 인코더는 렌더링과 관측 사이의 정렬 품질을 설명하는 특징 임베딩 F ∈ ℝ⁵¹²를 출력하기 위해 평균 풀링을 수행한다(그림 2 하단 중앙). 이 시점에서, 일반적으로 수행되는 것처럼[2, 31, 43] F를 유사도 스칼라로 직접 투영할 수 있다. 그러나, 이것은 다른 자세 가설을 무시하여 네트워크가 학습하기 어려운 절대 점수 할당을 출력하도록 강제한다.

보다 정보에 입각한 결정을 내리기 위해 모든 자세 가설의 전역 맥락을 활용하기 위해, 우리는 모든 K개의 자세 가설 간의 두 번째 수준의 비교를 도입한다. 연결된 특징 임베딩 **F** = [F₀, ..., F_{K-1}]^⊤ ∈ ℝ^{K×512}에 대해 다중 헤드 셀프 어텐션이 수행되며, 이는 모든 자세로부터의 자세 정렬 정보를 인코딩한다. **F**를 시퀀스로 취급함으로써, 이 접근 방식은 자연스럽게 다양한 K 길이로 일반화된다[56]. 순열에 구애받지 않도록 **F**에 위치 인코딩을 적용하지 않는다. 어텐션된 특징은 그런 다음 자세 가설에 할당될 점수 **S** ∈ ℝ^K로 선형 투영된다. 이 계층적 비교 전략의 효과는 그림 4의 전형적인 예에서 보여진다.

#### 대조 검증

자세 순위 네트워크를 훈련하기 위해, 우리는 자세 조건화 삼중 손실을 제안한다:

$$\mathcal{L}(i^+, i^-) = \max(\mathbf{S}(i^-) - \mathbf{S}(i^+) + \alpha, 0) \quad (11)$$

여기서 α는 대조 마진을 나타낸다; i⁻와 i⁺는 각각 음성 및 양성 자세 샘플을 나타내며, 실측값을 사용하여 ADD 메트릭[67]을 계산하여 결정된다. 표준 삼중 손실[26]과 달리, 입력이 이동을 고려하여 각 자세 가설에 따라 크롭되므로 앵커 샘플은 양성 및 음성 샘플 간에 공유되지 않는다. 목록의 각 쌍에 대해 이 손실을 계산할 수 있지만, 두 자세가 모두 실측값에서 멀면 비교가 모호해진다. 따라서, 양성 샘플이 비교를 의미 있게 만들 만큼 실측값에 충분히 가까운 시점에서 온 자세 쌍만 유지한다:

$$\mathbb{V}^+ = \{i : D(\mathbf{R}_i, \bar{\mathbf{R}}) < d\} \quad (12)$$

$$\mathbb{V}^- = \{0, 1, 2, ..., K-1\} \quad (13)$$

$$\mathcal{L}_{\text{rank}} = \sum_{i^+, i^-} \mathcal{L}(i^+, i^-) \quad (14)$$

여기서 합은 i⁺ ∈ V⁺, i⁻ ∈ V⁻, i⁺ ≠ i⁻에 대해 수행된다; R_i와 R̄은 각각 가설과 실측값의 회전이다; D(·)는 회전 간의 측지 거리를 나타낸다; d는 미리 정의된 임계값이다. [43]에서 사용된 InfoNCE 손실[46]도 실험했지만 더 나쁜 성능을 관찰했다(섹션 4.5). 이것은 우리 설정의 경우가 아닌 [43]에서 이루어진 완벽한 이동 가정 때문이라고 본다.

---

## 4. 실험

### 4.1. 데이터셋 및 설정

우리는 5개의 데이터셋을 고려한다: LINEMOD[22], Occluded-LINEMOD[1], YCB-Video[67], T-LESS[24], YCBInEOAT[61]. 이들은 다양한 도전적인 시나리오(밀집된 혼잡, 다중 인스턴스, 정적 또는 동적 장면, 테이블 탑 또는 로봇 조작)와 다양한 속성을 가진 객체(텍스처 없음, 광택, 대칭, 다양한 크기)를 포함한다.

우리의 프레임워크가 통합되었으므로, 두 설정(모델 프리 및 모델 기반)과 두 자세 예측 작업(6D 자세 추정 및 추적)의 조합을 고려하여 총 4개의 작업이 생성된다. 모델 프리 설정의 경우, [21]에 따라 실측 객체 자세 주석이 포함된 데이터셋의 훈련 분할에서 새로운 객체를 촬영한 여러 참조 이미지가 선택된다. 모델 기반 설정의 경우, 새로운 객체에 대한 CAD 모델이 제공된다. 절제 연구를 제외한 모든 평가에서, 우리의 방법은 미세 조정 없이 항상 동일한 훈련된 모델과 구성을 추론에 사용한다.

### 4.2. 메트릭

각 설정의 기준 프로토콜을 면밀히 따르기 위해, 다음 메트릭을 고려한다:

- ADD 및 ADD-S의 곡선 아래 면적(AUC)[67].
- [18, 21]에서 사용된 것처럼 객체 직경의 0.1보다 작은 ADD의 재현율(ADD-0.1d).
- BOP 챌린지[25]에서 도입된 VSD, MSSD 및 MSPD 메트릭의 평균 재현율(AR).

### 4.3. 자세 추정 비교

#### 모델 프리

표 1은 YCB-Video 데이터셋에서 최신 RGBD 방법[21, 27, 52]과의 비교 결과를 제시한다. 기준 결과는 [21]에서 채택되었다. [21]에 따라, 모든 방법은 공정한 비교를 위해 섭동된 실측 바운딩 박스를 2D 검출로 제공받는다. 표 2는 LINEMOD 데이터셋에서의 비교 결과를 제시한다. 기준 결과는 [18, 21]에서 채택되었다. RGB 기반 방법[18, 38, 53]은 깊이 부족을 보상하기 위해 훨씬 많은 참조 이미지의 특권을 받는다. RGBD 방법 중, FS6D[21]는 대상 데이터셋에서 미세 조정이 필요하다. 우리의 방법은 대상 데이터셋에서 미세 조정이나 ICP 정제 없이 두 데이터셋에서 기존 방법을 크게 능가한다.

그림 5는 정성적 비교를 시각화한다. 코드가 공개되지 않았으므로 FS6D[21]의 자세 예측에 대한 정성적 결과에 접근할 수 없다. 풀의 심한 자기 폐색과 텍스처 부족은 OnePose++[18]와 LatentFusion[48]에 크게 도전하지만, 우리의 방법은 자세를 성공적으로 추정한다.

#### 모델 기반

표 3은 BOP의 3개 핵심 데이터셋인 Occluded-LINEMOD[1], YCB-Video[67] 및 T-LESS[24]에서 RGBD 방법 간의 비교 결과를 제시한다. 모든 방법은 2D 검출에 Mask R-CNN[17]을 사용한다. 우리의 방법은 새로운 객체를 다루는 기존 모델 기반 방법과 인스턴스 수준 방법[15]을 큰 차이로 능가한다.

### 4.4. 자세 추적 비교

달리 명시되지 않는 한, 장기 추적 견고성을 평가하기 위해 추적 손실의 경우 평가된 방법에 재초기화가 적용되지 않는다. 정성적 결과는 보충 자료를 참조한다.

급격한 평면 외 회전, 동적 외부 폐색 및 분리된 카메라 모션의 도전에 대한 포괄적인 비교를 위해, 동적 로봇 조작의 비디오를 포함하는 YCBInEOAT[61] 데이터셋에서 자세 추적 방법을 평가한다. 모델 기반 설정 하의 결과는 표 4에 제시된다. 우리의 방법은 최고의 성능을 달성하며 실측 자세 초기화를 가진 인스턴스별 훈련 방법[61]까지도 능가한다. 또한, 우리의 통합 프레임워크는 외부 자세 초기화 없이 종단 간 자세 추정 및 추적도 가능하게 하며, 이는 표에서 Ours†로 표시된 유일한 그러한 기능을 가진 방법이다.

표 5는 YCB-Video[67] 데이터셋에서 자세 추적의 비교 결과를 제시한다. 기준 중, DeepIM[34], se(3)-TrackNet[61] 및 PoseRBPF[8]는 동일한 객체 인스턴스에서 훈련이 필요하고, Wüthrich et al.[66], RGF[28], ICG[51] 및 우리의 방법은 CAD 모델이 제공될 때 새로운 객체에 즉시 적용될 수 있다.

### 4.5. 분석

#### 절제 연구

표 6은 중요한 설계 선택의 절제 연구를 제시한다. 결과는 YCB-Video 데이터셋에서 ADD 및 ADD-S 메트릭의 AUC로 평가된다. **Ours (proposed)**는 모델 프리(16개 참조 이미지) 설정 하의 기본 버전이다. **W/o LLM texture augmentation**은 합성 훈련을 위한 LLM 지원 텍스처 증강을 제거한다. **W/o transformer**에서는 유사한 수의 매개변수를 유지하면서 트랜스포머 기반 아키텍처를 합성곱 및 선형 레이어로 대체한다. **W/o hierarchical comparison**은 2단계 계층적 비교 없이 자세 조건화 삼중 손실(식 11)로 훈련된 렌더링과 크롭된 입력만 비교한다. 테스트 시, 각 자세 가설을 입력 관측과 독립적으로 비교하고 최고 점수를 가진 자세를 출력한다. 예시 정성적 결과는 그림 4에 나와 있다. **Ours-InfoNCE**는 대조 검증된 쌍별 손실(식 14)을 [43]에서 사용된 InfoNCE 손실로 대체한다.

#### 참조 이미지 수의 영향

그림 6에 표시된 것처럼 참조 이미지 수가 YCB-Video 데이터셋에서 ADD 및 ADD-S의 AUC로 측정된 결과에 어떤 영향을 미치는지 연구한다. 전반적으로, 우리의 방법은 특히 ADD-S 메트릭에서 참조 이미지 수에 견고하며, 두 메트릭 모두 12개 이미지에서 포화된다. 주목할 만한 점은, 4개의 참조 이미지만 제공되어도 우리의 방법은 16개의 참조 이미지를 갖춘 FS6D[21]보다 더 강한 성능을 여전히 산출한다(표 1).

#### 훈련 데이터 스케일링 법칙

이론적으로, 무한한 양의 합성 데이터가 훈련을 위해 생성될 수 있다. 그림 7은 훈련 데이터 양이 YCB-Video 데이터셋에서 ADD 및 ADD-S 메트릭의 AUC로 측정된 결과에 어떤 영향을 미치는지 제시한다. 이득은 약 100만에서 포화된다.

#### 실행 시간

Intel i9-10980XE CPU와 NVIDIA RTX 3090 GPU 하드웨어에서 실행 시간을 측정한다. 자세 추정은 한 객체에 대해 약 1.3초가 걸리며, 자세 초기화에 4ms, 정제에 0.88초, 자세 선택에 0.42초가 소요된다. 추적은 자세 정제만 필요하고 여러 자세 가설이 없으므로 ~32Hz로 훨씬 빠르게 실행된다. 실제로, 초기화를 위해 자세 추정을 한 번 실행하고 실시간 성능을 위해 추적 모드로 전환할 수 있다.

---

## 5. 결론

우리는 모델 기반 및 모델 프리 설정을 모두 지원하는 새로운 객체의 6D 자세 추정 및 추적을 위한 통합 기반 모델을 제시한다. 4가지 다른 작업의 조합에 대한 광범위한 실험은 그것이 다재다능할 뿐만 아니라 각 작업에 특별히 설계된 기존 최신 방법을 상당한 차이로 능가함을 나타낸다. 인스턴스 수준 훈련이 필요한 방법과 비교할 만한 결과도 달성한다. 향후 연구에서, 단일 강체 객체를 넘어선 상태 추정을 탐구하는 것이 흥미로울 것이다.

---

## 참고문헌

[1] Eric Brachmann, Alexander Krull, Frank Michel, Stefan Gumhold, Jamie Shotton, and Carsten Rother. Learning 6D object pose estimation using 3d object coordinates. In 13th European Conference on Computer Vision (ECCV), pages 536–551, 2014.

[2] Dingding Cai, Janne Heikkilä, and Esa Rahtu. OVE6D: Object viewpoint encoding for depth-based 6D object pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6803–6813, 2022.

[3] Ming Cai and Ian Reid. Reconstruct locally, localize globally: A model free method for object pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3153–3163, 2020.

[4] Tianshi Cao, Karsten Kreis, Sanja Fidler, Nicholas Sharp, and Kangxue Yin. TexFusion: Synthesizing 3D textures with text-guided image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4169–4181, 2023.

[5] Dengsheng Chen, Jun Li, Zheng Wang, and Kai Xu. Learning canonical shape space for category-level 6D object pose and size estimation. In Proceedings of the IEEE International Conference on Computer Vision (CVPR), pages 11973–11982, 2020.

[6] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3D objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13142–13153, 2023.

[7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 248–255, 2009.

[8] Xinke Deng, Arsalan Mousavian, Yu Xiang, Fei Xia, Timothy Bretl, and Dieter Fox. PoseRBPF: A Rao-Blackwellized particle filter for 6D object pose tracking. In Robotics: Science and Systems (RSS), 2019.

[9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021.

[10] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3D scanned household items. In International Conference on Robotics and Automation (ICRA), pages 2553–2560, 2022.

[11] Mathieu Garon, Denis Laurendeau, and Jean-François Lalonde. A framework for evaluating 6-dof object trackers. In Proceedings of the European Conference on Computer Vision (ECCV), pages 582–597, 2018.

[12] Amos Gropp, Lior Yariv, Niv Haim, Matan Atzmon, and Yaron Lipman. Implicit geometric regularization for learning shapes. In International Conference on Machine Learning (ICML), pages 3789–3799, 2020.

[13] Agrim Gupta, Piotr Dollar, and Ross Girshick. LVIS: A dataset for large vocabulary instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5356–5364, 2019.

[14] John C Hart. Sphere tracing: A geometric method for the antialiased ray tracing of implicit surfaces. The Visual Computer, 12(10):527–545, 1996.

[15] Rasmus Laurvig Haugaard and Anders Glent Buch. Surfemb: Dense and continuous correspondence distributions for object pose estimation with learnt surface embeddings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6749–6758, 2022.

[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.

[17] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask R-CNN. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2961–2969, 2017.

[18] Xingyi He, Jiaming Sun, Yuang Wang, Di Huang, Hujun Bao, and Xiaowei Zhou. OnePose++: Keypoint-free one-shot object pose estimation without CAD models. Advances in Neural Information Processing Systems (NeurIPS), 35:35103–35115, 2022.

[19] Yisheng He, Wei Sun, Haibin Huang, Jianran Liu, Haoqiang Fan, and Jian Sun. PVN3D: A deep point-wise 3D keypoints voting network for 6DoF pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11632–11641, 2020.

[20] Yisheng He, Haibin Huang, Haoqiang Fan, Qifeng Chen, and Jian Sun. FFB6D: A full flow bidirectional fusion network for 6D pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3003–3013, 2021.

[21] Yisheng He, Yao Wang, Haoqiang Fan, Jian Sun, and Qifeng Chen. FS6D: Few-shot 6D pose estimation of novel objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6814–6824, 2022.

[22] Stefan Hinterstoisser, Stefan Holzer, Cedric Cagniart, Slobodan Ilic, Kurt Konolige, Nassir Navab, and Vincent Lepetit. Multimodal templates for real-time detection of texture-less objects in heavily cluttered scenes. In International Conference on Computer Vision (ICCV), pages 858–865, 2011.

[23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NeurIPS), 33:6840–6851, 2020.

[24] Tomáš Hodan, Pavel Haluza, Štepán Obdržálek, Jiri Matas, Manolis Lourakis, and Xenophon Zabulis. T-LESS: An RGB-D dataset for 6D pose estimation of texture-less objects. In IEEE Winter Conference on Applications of Computer Vision (WACV), pages 880–888, 2017.

[25] Tomas Hodan, Frank Michel, Eric Brachmann, Wadim Kehl, Anders GlentBuch, Dirk Kraft, Bertram Drost, Joel Vidal, Stephan Ihrke, Xenophon Zabulis, et al. BOP: Benchmark for 6D object pose estimation. In Proceedings of the European Conference on Computer Vision (ECCV), pages 19–34, 2018.

[26] Elad Hoffer and Nir Ailon. Deep metric learning using triplet network. In Third International Workshop on Similarity-Based Pattern Recognition (SIMBAD), pages 84–92, 2015.

[27] Shengyu Huang, Zan Gojcic, Mikhail Usvyatsov, Andreas Wieser, and Konrad Schindler. PREDATOR: Registration of 3D point clouds with low overlap. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4267–4276, 2021.

[28] Jan Issac, Manuel Wüthrich, Cristina Garcia Cifuentes, Jeannette Bohg, Sebastian Trimpe, and Stefan Schaal. Depth-based object tracking using a robust gaussian filter. In IEEE International Conference on Robotics and Automation (ICRA), pages 608–615, 2016.

[29] Daniel Kappler, Franziska Meier, Jan Issac, Jim Mainprice, Cristina Garcia Cifuentes, Manuel Wüthrich, Vincent Berenz, Stefan Schaal, Nathan Ratliff, and Jeannette Bohg. Real-time perception meets reactive motion generation. IEEE Robotics and Automation Letters, 3(3):1864–1871, 2018.

[30] Yann Labbé, Justin Carpentier, Mathieu Aubry, and Josef Sivic. CosyPose: Consistent multi-view multi-object 6D pose estimation. In European Conference on Computer Vision (ECCV), pages 574–591, 2020.

[31] Yann Labbé, Lucas Manuelli, Arsalan Mousavian, Stephen Tyree, Stan Birchfield, Jonathan Tremblay, Justin Carpentier, Mathieu Aubry, Dieter Fox, and Josef Sivic. MegaPose: 6D pose estimation of novel objects via render & compare. In 6th Annual Conference on Robot Learning (CoRL), 2022.

[32] Taeyeop Lee, Jonathan Tremblay, Valts Blukis, Bowen Wen, Byeong-Uk Lee, Inkyu Shin, Stan Birchfield, In So Kweon, and Kuk-Jin Yoon. TTA-COPE: Test-time adaptation for category-level object pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21285–21295, 2023.

[33] Fu Li, Shishir Reddy Vutukur, Hao Yu, Ivan Shugurov, Benjamin Busam, Shaowu Yang, and Slobodan Ilic. NeRF-Pose: A first-reconstruct-then-regress approach for weakly-supervised 6D object pose estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2123–2133, 2023.

[34] Yi Li, Gu Wang, Xiangyang Ji, Yu Xiang, and Dieter Fox. DeepIM: Deep iterative matching for 6D pose estimation. In Proceedings of the European Conference on Computer Vision (ECCV), pages 683–698, 2018.

[35] Zhigang Li, Gu Wang, and Xiangyang Ji. CDPN: Coordinates-based disentangled pose network for real-time RGB-based 6-DoF object pose estimation. In CVF International Conference on Computer Vision (ICCV), pages 7677–7686, 2019.

[36] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In 13th European Conference on Computer Vision (ECCV), pages 740–755, 2014.

[37] Yunzhi Lin, Jonathan Tremblay, Stephen Tyree, Patricio A Vela, and Stan Birchfield. Keypoint-based category-level object pose tracking from an RGB sequence with uncertainty estimation. In International Conference on Robotics and Automation (ICRA), 2022.

[38] Yuan Liu, Yilin Wen, Sida Peng, Cheng Lin, Xiaoxiao Long, Taku Komura, and Wenping Wang. Gen6D: Generalizable model-free 6-DoF object pose estimation from RGB images. ECCV, 2022.

[39] William E Lorensen and Harvey E Cline. Marching cubes: A high resolution 3d surface construction algorithm. In Seminal graphics: pioneering efforts that shaped the field, pages 347–353. 1998.

[40] Eric Marchand, Hideaki Uchiyama, and Fabien Spindler. Pose estimation for augmented reality: A hands-on survey. IEEE Transactions on Visualization and Computer Graphics (TVCG), 22(12):2633–2651, 2015.

[41] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.

[42] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4):102:1–102:15, 2022.

[43] Van Nguyen Nguyen, Yinlin Hu, Yang Xiao, Mathieu Salzmann, and Vincent Lepetit. Templates for 3D object pose estimation revisited: Generalization to new objects and robustness to occlusions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6771–6780, 2022.

[44] Van Nguyen Nguyen, Thibault Groueix, Georgy Ponimatkin, Vincent Lepetit, and Tomas Hodan. Cnos: A strong baseline for cad-based novel object segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2134–2140, 2023.

[45] Brian Okorn, Qiao Gu, Martial Hebert, and David Held. Zephyr: Zero-shot pose hypothesis rating. In IEEE International Conference on Robotics and Automation (ICRA), pages 14141–14148, 2021.

[46] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[47] Kiru Park, Timothy Patten, and Markus Vincze. Pix2Pose: Pixel-wise coordinate regression of objects for 6D pose estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 7668–7677, 2019.

[48] Keunhong Park, Arsalan Mousavian, Yu Xiang, and Dieter Fox. LatentFusion: End-to-end differentiable reconstruction and rendering for unseen object pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10710–10719, 2020.

[49] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022.

[50] Ivan Shugurov, Fu Li, Benjamin Busam, and Slobodan Ilic. OSOP: A multi-stage one shot object pose estimation framework. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6835–6844, 2022.

[51] Manuel Stoiber, Martin Sundermeyer, and Rudolph Triebel. Iterative corresponding geometry: Fusing region and depth for highly efficient 3D tracking of textureless objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6855–6865, 2022.

[52] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. LoFTR: Detector-free local feature matching with transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8922–8931, 2021.

[53] Jiaming Sun, Zihao Wang, Siyu Zhang, Xingyi He, Hongcheng Zhao, Guofeng Zhang, and Xiaowei Zhou. OnePose: One-shot object pose estimation without CAD models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6825–6834, 2022.

[54] Meng Tian, Marcelo H Ang, and Gim Hee Lee. Shape prior deformation for categorical 6D object pose and size estimation. In Proceedings of the European Conference on Computer Vision (ECCV), pages 530–546, 2020.

[55] Jonathan Tremblay, Thang To, Balakumar Sundaralingam, Yu Xiang, Dieter Fox, and Stan Birchfield. Deep object pose estimation for semantic robotic grasping of household objects. In Conference on Robot Learning (CoRL), pages 306–316, 2018.

[56] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems (NeurIPS), 30, 2017.

[57] Chen Wang, Roberto Martín-Martín, Danfei Xu, Jun Lv, Cewu Lu, Li Fei-Fei, Silvio Savarese, and Yuke Zhu. 6-PACK: Category-level 6D pose tracker with anchor-based keypoints. In IEEE International Conference on Robotics and Automation (ICRA), pages 10059–10066, 2020.

[58] He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin, Shuran Song, and Leonidas J Guibas. Normalized object coordinate space for category-level 6D object pose and size estimation. In Proceedings of the IEEE International Conference on Computer Vision (CVPR), pages 2642–2651, 2019.

[59] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. NeuS: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

[60] Bowen Wen and Kostas Bekris. BundleTrack: 6D pose tracking for novel objects without instance or category-level 3D models. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 8067–8074, 2021.

[61] Bowen Wen, Chaitanya Mitash, Baozhang Ren, and Kostas E Bekris. se(3)-TrackNet: Data-driven 6D pose tracking by calibrating image residuals in synthetic domains. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 10367–10373, 2020.

[62] Bowen Wen, Chaitanya Mitash, Sruthi Soorian, Andrew Kimmel, Avishai Sintov, and Kostas E Bekris. Robust, occlusion-aware pose estimation for objects grasped by adaptive hands. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 6210–6217. IEEE, 2020.

[63] Bowen Wen, Wenzhao Lian, Kostas Bekris, and Stefan Schaal. CatGrasp: Learning category-level task-relevant grasping in clutter from simulation. In International Conference on Robotics and Automation (ICRA), pages 6401–6408, 2022.

[64] Bowen Wen, Wenzhao Lian, Kostas Bekris, and Stefan Schaal. You only demonstrate once: Category-level manipulation from single visual demonstration. RSS, 2022.

[65] Bowen Wen, Jonathan Tremblay, Valts Blukis, Stephen Tyree, Thomas Müller, Alex Evans, Dieter Fox, Jan Kautz, and Stan Birchfield. BundleSDF: Neural 6-DoF tracking and 3D reconstruction of unknown objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 606–617, 2023.

[66] Manuel Wüthrich, Peter Pastor, Mrinal Kalakrishnan, Jeannette Bohg, and Stefan Schaal. Probabilistic object tracking using a range camera. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3195–3202, 2013.

[67] Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, and Dieter Fox. PoseCNN: A convolutional neural network for 6D object pose estimation in cluttered scenes. In Robotics: Science and Systems (RSS), 2018.

[68] Lior Yariv, Yoni Kasten, Dror Moran, Meirav Galun, Matan Atzmon, Basri Ronen, and Yaron Lipman. Multiview neural surface reconstruction by disentangling geometry and appearance. Advances in Neural Information Processing Systems (NeurIPS), 33:2492–2502, 2020.

[69] Ruida Zhang, Yan Di, Fabian Manhardt, Federico Tombari, and Xiangyang Ji. SSP-Pose: Symmetry-aware shape prior deformation for direct category-level object pose estimation. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7452–7459, 2022.

[70] Heng Zhao, Shenxing Wei, Dahu Shi, Wenming Tan, Zheyang Li, Ye Ren, Xing Wei, Yi Yang, and Shiliang Pu. Learning symmetry-aware geometry correspondences for 6D object pose estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 14045–14054, 2023.

[71] Linfang Zheng, Chen Wang, Yinghan Sun, Esha Dasgupta, Hua Chen, Aleš Leonardis, Wei Zhang, and Hyung Jin Chang. HS-Pose: Hybrid scope feature extraction for category-level object pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17163–17173, 2023.

[72] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5745–5753, 2019.

---

## 표

### 표 1. 모델 프리 자세 추정 결과 (YCB-Video 데이터셋, ADD 및 ADD-S의 AUC)

| 참조 이미지 | PREDATOR [27] | LoFTR [52] | FS6D-DPM [21] | Ours |
|------------|---------------|------------|---------------|------|
| | 16 | 16 | 16 | 16 |
| 미세조정 불필요 | ✓ | ✓ | ✗ | ✓ |
| **평균** ADD-S / ADD | 71.0 / 24.3 | 52.5 / 26.2 | 88.4 / 42.1 | **97.4 / 91.5** |

### 표 2. 모델 프리 자세 추정 결과 (LINEMOD 데이터셋, ADD-0.1d)

| 방법 | 모달리티 | 미세조정 불필요 | 참조 이미지 | 평균 |
|------|----------|----------------|------------|------|
| Gen6D [38] | RGB | ✗ | 200 | - |
| OnePose [53] | RGB | ✓ | 200 | 63.6 |
| OnePose++ [18] | RGB | ✓ | 200 | 76.9 |
| LatentFusion [48] | RGBD | ✓ | 16 | 87.1 |
| FS6D [21] | RGBD | ✗ | 16 | 88.9 |
| FS6D [21] + ICP | RGBD | ✗ | 16 | 91.5 |
| **Ours** | RGBD | ✓ | 16 | **99.9** |

### 표 3. 모델 기반 자세 추정 결과 (BOP 데이터셋, AR 점수)

| 방법 | 새로운 객체 | LM-O | T-LESS | YCB-V | 평균 |
|------|------------|------|--------|-------|------|
| SurfEmb [15] + ICP | ✗ | 75.8 | 82.8 | 80.6 | 79.7 |
| MegaPose-RGBD [31] | ✓ | 58.3 | 54.3 | 63.3 | 58.6 |
| **Ours** | ✓ | **78.8** | **83.0** | **88.0** | **83.3** |

### 표 4. 자세 추적 결과 (YCBInEOAT 데이터셋, ADD 및 ADD-S의 AUC)

| | se(3)-TrackNet [61] | RGF [28] | BundleTrack [60] | BundleSDF [65] | Wüthrich [66] | Ours | Ours† |
|---|---------------------|----------|------------------|----------------|---------------|------|-------|
| 새로운 객체 | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 초기 자세 | GT | GT | GT | GT | GT | GT | Est. |
| **전체** ADD-S / ADD | 95.53 / 92.66 | 39.90 / 29.98 | 92.53 / 87.34 | 93.77 / 86.95 | 89.18 / 78.28 | **96.42 / 93.09** | 96.40 / 93.22 |

### 표 5. 자세 추적 결과 (YCB-Video 데이터셋, ADD 및 ADD-S의 AUC)

모델 기반 설정에서 Ours는 ADD 96.0 / ADD-S 97.9를 달성하여 모든 기존 방법을 능가함.

### 표 6. 주요 설계 선택의 절제 연구

| | ADD | ADD-S |
|---|-----|-------|
| Ours (제안) | 91.52 | 97.40 |
| W/o LLM 텍스처 증강 | 90.83 | 97.38 |
| W/o 트랜스포머 | 90.77 | 97.33 |
| W/o 계층적 비교 | 89.05 | 96.67 |
| Ours-InfoNCE | 89.39 | 97.29 |

---

*이 논문은 CVPR 2024에서 발표되었습니다.*
