# MiLSF HetNet Simulator

이 프로젝트는 IEEE 논문 "A Base Station Sleeping Strategy in Heterogeneous Cellular Networks Based on User Traffic Prediction"의 MiLSF (Minimum Load Sleep First) 전략을 구현한 시뮬레이터입니다.

## 📁 프로젝트 구조

```
milsf_hetnet/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   └── simulation_config.py
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── enhanced_cell.py
│   │   ├── traffic_aware_ue.py
│   │   └── hetnet_base.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── milsf_ric.py
│   │   └── traffic_prediction.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   └── metrics.py
│   └── scenarios/
│       ├── __init__.py
│       └── hetnet_scenarios.py
├── examples/
│   ├── __init__.py
│   ├── basic_milsf_demo.py
│   ├── paper_reproduction.py
│   └── custom_scenarios.py
├── tests/
│   ├── __init__.py
│   ├── test_enhanced_cell.py
│   ├── test_milsf_ric.py
│   └── test_traffic_prediction.py
├── data/
│   ├── traffic_patterns/
│   └── results/
└── docs/
    ├── installation.md
    ├── api_reference.md
    └── tutorial.md
```

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/milsf_hetnet.git
cd milsf_hetnet

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 패키지 설치
pip install -e .
```

### 2. 기본 실행

```bash
# 기본 MiLSF 데모
python examples/basic_milsf_demo.py

# 논문 재현 실험
python examples/paper_reproduction.py
```

## 📋 주요 컴포넌트

### 🏗️ Core Components

#### `EnhancedCell` (src/core/enhanced_cell.py)
- Macro/Micro 기지국 구현
- SINR 계산 및 전력 소모 모델
- 3GPP 표준 경로손실 모델

#### `TrafficAwareUE` (src/core/traffic_aware_ue.py)
- 트래픽 패턴 생성 및 예측
- 시간별 트래픽 변화 모델링

### 🧠 Algorithms

#### `MiLSF_RIC` (src/algorithms/milsf_ric.py)
- MiLSF 알고리즘 핵심 구현
- 사용자 재할당 전략
- 에너지 절약 계산

#### `TrafficPredictionBLSTM` (src/algorithms/traffic_prediction.py)
- Bidirectional LSTM 구현
- 트래픽 패턴 학습 및 예측

## 🎯 사용 예제

### 기본 시뮬레이션

```python
from milsf_hetnet import create_hetnet_simulation
from milsf_hetnet.algorithms import MiLSF_RIC

# 네트워크 생성
sim = create_hetnet_simulation(
    n_macro_cells=7,
    n_micro_cells=10,
    n_users=25
)

# MiLSF RIC 추가
ric = MiLSF_RIC(sim, interval=30.0)
sim.add_ric(ric)

# 시뮬레이션 실행 (24시간)
sim.run(until=86400)
```

### 커스텀 시나리오

```python
from milsf_hetnet.scenarios import CustomHetNetScenario

# 커스텀 시나리오 생성
scenario = CustomHetNetScenario(
    area_size=15000,  # 15km x 15km
    macro_positions=[(5000, 5000), (10000, 10000)],
    n_micro_cells=20,
    user_density=50
)

sim = scenario.create_simulation()
sim.run(until=172800)  # 48시간
```

## ⚙️ 설정 옵션

### 시뮬레이션 파라미터 (config/simulation_config.py)

```python
SIMULATION_CONFIG = {
    # 네트워크 파라미터
    'area_size': 10000,  # meters
    'n_macro_cells': 7,
    'n_micro_cells': 10,
    'n_users': 25,
    
    # MiLSF 파라미터
    'low_load_start': 22,  # 10 PM
    'low_load_end': 6,     # 6 AM
    'sinr_threshold': -6,  # dB
    'ric_interval': 30.0,  # seconds
    
    # 트래픽 파라미터
    'base_traffic_rate': 1.0,  # Mbps
    'traffic_variation': 0.5,
    'prediction_window': 24,   # hours
}
```

## 📊 결과 분석

### 로그 데이터 구조

```
timestamp  cell_id  bs_type  state   load    power_W  throughput  energy_savings_%
0.0        0        macro    active  0.456   180.2    15.6        0.00
60.0       1        micro    active  0.234   25.4     8.2         0.00
1320.0     8        micro    sleep   0.000   2.0      0.0         11.26
```

### 성능 메트릭

- **에너지 절약률**: 전체 네트워크 전력 소모 감소 비율
- **수면 셀 수**: 저부하 기간 동안 수면 상태인 MiBS 수
- **QoS 보장**: SINR 임계값 위반 사용자 수
- **트래픽 예측 정확도**: BLSTM 모델의 MAE, RMSE

## 🔬 실험 시나리오

### 1. 논문 재현 실험

```bash
python examples/paper_reproduction.py --scenario all
```

- Scenario I: PPP vs MHCPP 배치 비교
- Scenario II: 사용자 수 변화에 따른 성능
- Scenario III: SINR 임계값 영향 분석
- Scenario IV: 수면 셀 수와 에너지 절약 관계

### 2. 확장 실험

```bash
python examples/custom_scenarios.py --config extended
```

- 더 큰 네트워크 규모
- 다양한 트래픽 패턴
- 동적 사용자 이동성

## 🛠️ 개발 가이드

### 새로운 알고리즘 추가

1. `src/algorithms/` 디렉토리에 새 파일 생성
2. `RIC` 클래스를 상속받아 구현
3. `examples/` 디렉토리에 테스트 스크립트 추가

```python
# src/algorithms/my_algorithm.py
from milsf_hetnet.core import RIC

class MyAlgorithm(RIC):
    def __init__(self, sim, **kwargs):
        super().__init__(sim, **kwargs)
        
    def loop(self):
        while True:
            # 알고리즘 로직 구현
            yield self.sim.wait(self.interval)
```

### 새로운 메트릭 추가

1. `src/utils/metrics.py`에 메트릭 함수 추가
2. `Logger` 클래스에 로깅 로직 추가

```python
# src/utils/metrics.py
def calculate_spectral_efficiency(sinr_values):
    """Calculate average spectral efficiency"""
    return sum(log2(1 + sinr) for sinr in sinr_values) / len(sinr_values)
```

## 🐛 트러블슈팅

### 일반적인 문제들

1. **AIMM_simulator 의존성 오류**
   ```bash
   pip install AIMM-simulator
   # 또는 로컬 설치
   make install_local
   ```

2. **TensorFlow GPU 설정**
   ```bash
   pip install tensorflow-gpu
   # CUDA 설정 확인
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

3. **메모리 부족 오류**
   - 네트워크 크기 줄이기
   - 배치 크기 조정
   - 예측 윈도우 단축

### 성능 최적화

1. **시뮬레이션 속도 향상**
   - RIC 간격 늘리기 (`ric_interval` 증가)
   - 로깅 빈도 줄이기
   - 불필요한 계산 제거

2. **메모리 사용량 감소**
   - 트래픽 히스토리 길이 제한
   - 예측 모델 크기 축소

## 📚 추가 자료

- [논문 원문](https://ieeexplore.ieee.org/document/10285284)
- [AIMM Simulator 문서](https://aimm-simulator.readthedocs.io/)
- [API 레퍼런스](docs/api_reference.md)
- [튜토리얼](docs/tutorial.md)

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.