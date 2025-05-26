# MiLSF HetNet Simulator - 설치 및 실행 가이드

## 📦 설치 방법

### 1. 기본 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/milsf_hetnet.git
cd milsf_hetnet

# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 패키지 설치 (개발 모드)
pip install -e .
```

### 2. AIMM Simulator 설치

AIMM Simulator가 필요합니다. 다음 중 하나의 방법을 선택하세요:

```bash
# 방법 1: PyPI에서 설치
pip install AIMM-simulator

# 방법 2: 소스에서 설치 (최신 버전)
git clone https://github.com/keithbriggs/AIMM-simulator.git
cd AIMM-simulator
make install_local
cd ..
```

### 3. TensorFlow 설치 (선택사항)

BLSTM 트래픽 예측을 사용하려면:

```bash
# CPU 버전
pip install tensorflow

# GPU 버전 (CUDA 설정 필요)
pip install tensorflow-gpu
```

## 🚀 빠른 시작

### 1. 기본 데모 실행

```bash
python examples/basic_milsf_demo.py
```

**예상 출력:**
```
=== MiLSF Basic Demonstration ===

Creating simple heterogeneous network...
  Deployed Macro BS 0 at (2000, 2000)
  Deployed Macro BS 1 at (6000, 2000)
  Deployed Macro BS 2 at (4000, 5000)
  Deployed Micro BS 3 at (3000, 3000)
  ...

Network created with:
  - 3 Macro cells
  - 5 Micro cells
  - 15 UEs

MiLSF RIC started at t=0.0
t=79200.0 MiLSF Decisions: MiBS[5] sleeping (load=0.123)
Energy savings: 8.45%
```

### 2. 프로그래밍 방식 사용

```python
# your_simulation.py
import sys
sys.path.append('src')

from core.enhanced_cell import EnhancedCell, BSType
from core.traffic_aware_ue import TrafficAwareUE
from algorithms.milsf_ric import MiLSF_RIC
from AIMM_simulator import Sim

# 시뮬레이션 생성
sim = Sim()

# 매크로 셀 배치
macro = EnhancedCell(sim, bs_type=BSType.MACRO, xyz=(1000, 1000, 25))

# 마이크로 셀 배치  
micro = EnhancedCell(sim, bs_type=BSType.MICRO, xyz=(1200, 1200, 10))

# 사용자 배치
ue = TrafficAwareUE(sim, xyz=(1100, 1100, 2))
ue.attach(macro)

# MiLSF RIC 추가
ric = MiLSF_RIC(sim, interval=30.0)
sim.add_ric(ric)

# 시뮬레이션 실행
sim.run(until=86400)  # 24시간
```

## 📁 프로젝트 구조 이해

```
milsf_hetnet/
├── src/                    # 소스 코드
│   ├── core/              # 핵심 컴포넌트
│   ├── algorithms/        # MiLSF 및 예측 알고리즘
│   └── utils/             # 유틸리티 함수들
├── examples/              # 예제 코드
├── config/                # 설정 파일
├── tests/                 # 테스트 코드
└── docs/                  # 문서
```

## ⚙️ 설정 커스터마이징

### 1. 기본 설정 수정

```python
# config/simulation_config.py에서 설정 수정
from config.simulation_config import get_config

# 기본 설정 로드
config = get_config('default')

# 네트워크 크기 변경
config['area_size'] = 15000  # 15km x 15km
config['n_micro_cells'] = 20
config['n_users'] = 50

# MiLSF 파라미터 조정
config['milsf']['sinr_threshold_dB'] = -9.0
config['milsf']['low_load_start_hour'] = 23
```

### 2. 커스텀 시나리오 생성

```python
# 밀집 네트워크 시나리오
dense_config = get_config('dense_network')

# 대규모 네트워크 시나리오  
large_config = get_config('large_network')
```

## 📊 결과 분석

### 1. 로그 파일 분석

시뮬레이션 실행 후 로그 파일이 생성됩니다:

```
#time	cell_id	bs_type	state	load	power_W	users
0.0	0	macro	active	0.456	180.2	5
0.0	3	micro	active	0.234	25.4	2
79200.0	3	micro	sleep	0.000	2.0	0
```

### 2. 성능 메트릭

주요 성능 지표:
- **에너지 절약률**: 전체 네트워크 전력 소모 감소 비율
- **수면 셀 수**: 저부하 기간 동안 수면 상태인 MiBS 수  
- **QoS 보장**: SINR 임계값 위반 없이 서비스 제공
- **사용자 재할당 성공률**: 수면 전 사용자 재할당 성공 비율

## 🐛 문제 해결

### 1. AIMM Simulator 설치 오류

```bash
# 직접 빌드
cd AIMM-simulator
python -m build
pip install dist/*.whl
```

### 2. TensorFlow 오류

```bash
# 호환성 확인
python -c "import tensorflow as tf; print(tf.__version__)"

# 구버전 설치
pip install "tensorflow==2.12.0"
```

### 3. 메모리 부족

```python
# 네트워크 크기 줄이기
config['n_users'] = 15
config['n_micro_cells'] = 5

# 로깅 간격 늘리기
logger = BasicLogger(sim, logging_interval=600)  # 10분마다
```

## 📈 성능 최적화

### 1. 시뮬레이션 속도 향상

```python
# RIC 간격 늘리기
ric = MiLSF_RIC(sim, interval=120.0)  # 2분마다

# 로깅 빈도 줄이기
logger = BasicLogger(sim, logging_interval=300)

# Verbosity 낮추기
ric = MiLSF_RIC(sim, verbosity=0)
```

### 2. 메모리 사용량 감소

```python
# 트래픽 히스토리 단축
ue = TrafficAwareUE(sim)
ue.traffic_history = deque(maxlen=24)  # 24시간만 저장

# 예측 윈도우 축소
predictor = TrafficPredictionBLSTM(sequence_length=12)
```

## 📚 다음 단계

1. **예제 코드 확인**: `examples/` 디렉토리의 다른 예제들
2. **API 문서**: `docs/api_reference.md` 참조
3. **튜토리얼**: `docs/tutorial.md`에서 자세한 사용법
4. **논문 재현**: `examples/paper_reproduction.py`로 논문 결과 재현

## 💡 팁

- 첫 실행 시 작은 네트워크로 테스트해보세요
- 로그 출력을 확인하여 MiLSF 동작을 이해하세요
- `verbosity=1`로 설정하여 상세한 진행 상황을 확인하세요
- GPU가 있다면 TensorFlow-GPU를 설치하여 트래픽 예측 속도를 향상시키세요

## 🔧 고급 사용법

### 1. 커스텀 알고리즘 구현

```python
# algorithms/my_algorithm.py
from algorithms.milsf_ric import MiLSF_RIC

class MyCustomRIC(MiLSF_RIC):
    def milsf_algorithm(self):
        # 커스텀 로직 구현
        decisions = []
        # ... 알고리즘 구현
        return decisions
```

### 2. 커스텀 메트릭 추가

```python
# utils/custom_metrics.py
def calculate_spectral_efficiency(cells):
    """커스텀 메트릭 계산"""
    total_se = 0
    for cell in cells:
        if hasattr(cell, 'attached'):
            for ue_id in cell.attached:
                sinr = cell.get_serving_quality(cell.sim.UEs[ue_id])
                se = log2(1 + 10**(sinr/10))
                total_se += se
    return total_se
```

### 3. 배치 실험 실행

```python
# experiments/batch_run.py
import itertools
from config.simulation_config import get_config

# 파라미터 그리드 정의
user_counts = [15, 25, 35, 50]
sinr_thresholds = [-9, -6, -3]
scenarios = ['small_network', 'default', 'dense_network']

results = []

for users, sinr, scenario in itertools.product(user_counts, sinr_thresholds, scenarios):
    config = get_config(scenario)
    config['n_users'] = users
    config['milsf']['sinr_threshold_dB'] = sinr
    
    # 시뮬레이션 실행
    sim = create_simulation_from_config(config)
    sim.run(until=86400)
    
    # 결과 저장
    results.append({
        'users': users,
        'sinr_threshold': sinr,
        'scenario': scenario,
        'energy_savings': get_energy_savings(sim)
    })
```

## 📊 결과 시각화

### 1. 기본 플롯 생성

```python
import matplotlib.pyplot as plt
import pandas as pd

# 로그 데이터 읽기
df = pd.read_csv('simulation_log.csv', sep='\t')

# 에너지 절약 시계열 플롯
plt.figure(figsize=(12, 6))
df_energy = df[df['energy_savings_%'] > 0]
plt.plot(df_energy['time']/3600, df_energy['energy_savings_%'])
plt.xlabel('Time (hours)')
plt.ylabel('Energy Savings (%)')
plt.title('MiLSF Energy Savings Over Time')
plt.grid(True)
plt.show()
```

### 2. 대화형 대시보드 (Streamlit)

```python
# dashboard/milsf_dashboard.py
import streamlit as st
import plotly.express as px

st.title('MiLSF Simulation Dashboard')

# 사이드바 설정
st.sidebar.header('Simulation Parameters')
n_users = st.sidebar.slider('Number of Users', 10, 100, 25)
sinr_threshold = st.sidebar.slider('SINR Threshold (dB)', -12, 0, -6)

# 실시간 시뮬레이션 실행
if st.button('Run Simulation'):
    # 시뮬레이션 실행 및 결과 표시
    results = run_simulation(n_users, sinr_threshold)
    
    # 결과 시각화
    fig = px.line(results, x='time', y='energy_savings', 
                  title='Energy Savings Over Time')
    st.plotly_chart(fig)
```

실행:
```bash
streamlit run dashboard/milsf_dashboard.py
```

## 🧪 테스트 실행

```bash
# 모든 테스트 실행
pytest tests/

# 특정 테스트만 실행
pytest tests/test_milsf_ric.py

# 커버리지 포함
pytest --cov=src tests/
```

## 📝 논문 재현 실험

```bash
# 논문의 모든 시나리오 재현
python examples/paper_reproduction.py --all

# 특정 시나리오만 실행
python examples/paper_reproduction.py --scenario 1
python examples/paper_reproduction.py --scenario 2
```

**재현 가능한 실험:**
- **Scenario I**: PPP vs MHCPP 배치 비교
- **Scenario II**: 사용자 수 변화 영향
- **Scenario III**: SINR 임계값 영향
- **Scenario IV**: 수면 셀 수와 에너지 절약 관계

## 🔍 디버깅 팁

### 1. 상세 로깅 활성화

```python
# 모든 컴포넌트의 verbosity 증가
ric = MiLSF_RIC(sim, verbosity=2)
logger = DetailedLogger(sim, verbosity=2)

# 특정 UE 추적
ue.verbosity = 2
```

### 2. 단계별 디버깅

```python
# 시뮬레이션을 단계적으로 실행
sim = create_simulation()

# 초기 상태 확인
print_network_status(sim)

# 짧은 시간만 실행
sim.run(until=3600)  # 1시간만

# 중간 상태 확인
print_network_status(sim)
```

### 3. 네트워크 상태 시각화

```python
def visualize_network(sim):
    """네트워크 상태 시각화"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 셀 위치 표시
    for cell in sim.cells:
        x, y = cell.xyz[0], cell.xyz[1]
        color = 'red' if cell.bs_type == BSType.MACRO else 'blue'
        marker = 'o' if cell.state == BSState.ACTIVE else 'x'
        ax.scatter(x, y, c=color, marker=marker, s=100)
        ax.text(x+50, y+50, f'{cell.i}', fontsize=8)
    
    # UE 위치 표시
    for ue in sim.UEs:
        x, y = ue.xyz[0], ue.xyz[1]
        ax.scatter(x, y, c='green', marker='^', s=50)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Network Topology')
    ax.legend(['Macro BS', 'Micro BS', 'UE'])
    plt.show()

# 사용법
visualize_network(sim)
```

## 📖 추가 리소스

### 문서
- [API Reference](docs/api_reference.md)
- [Algorithm Details](docs/algorithm_details.md) 
- [Configuration Guide](docs/configuration.md)

### 예제 코드
- [Basic Demo](examples/basic_milsf_demo.py) - 기본 사용법
- [Paper Reproduction](examples/paper_reproduction.py) - 논문 재현
- [Custom Scenarios](examples/custom_scenarios.py) - 커스텀 시나리오
- [Batch Experiments](examples/batch_experiments.py) - 배치 실험

### 관련 링크
- [원본 논문](https://ieeexplore.ieee.org/document/10285284)
- [AIMM Simulator](https://github.com/keithbriggs/AIMM-simulator)
- [이슈 리포트](https://github.com/your-repo/milsf_hetnet/issues)

## 💬 지원 및 기여

### 질문이나 문제가 있으신가요?

1. **문서 확인**: 먼저 docs/ 디렉토리의 문서들을 확인해보세요
2. **예제 참조**: examples/ 디렉토리의 예제 코드들을 참조하세요
3. **이슈 등록**: [GitHub Issues](https://github.com/your-repo/milsf_hetnet/issues)에 문제를 등록하세요
4. **토론 참여**: [GitHub Discussions](https://github.com/your-repo/milsf_hetnet/discussions)에서 토론하세요

### 기여하기

1. 저장소를 포크하세요
2. 새로운 기능 브랜치를 만드세요 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋하세요 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시하세요 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성하세요

### 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

**축하합니다! 🎉** 

이제 MiLSF HetNet 시뮬레이터를 사용할 준비가 되었습니다. 기본 데모부터 시작해서 점진적으로 더 복잡한 시나리오를 탐색해보세요. 

연구나 개발 과정에서 궁금한 점이 있으시면 언제든지 문의해주세요!