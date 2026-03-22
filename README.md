# 🚨 AI 기반 실시간 SOS 행동 인식 및 MLOps 자가 학습 시스템
**"CCTV 영상을 통한 위험 상황(낙상, 도움 요청) 감지 및 데이터 피드백 루프 기반 모델 고도화 서비스"**

---

## 🔗 프로젝트 리소스 (Resources)
- GitHub Repository: 👉 [깃허브 리포지토리 바로가기](#)
- Demo Video: 👉 [서비스 시현 영상 (준비 중)](#)

---

## 📅 프로젝트 개요
- **프로젝트 기간**: 2026.03.20 ~ 2026.03.22  
- **개발자**: 심효진 (@hyojin-shj)  
- **한 줄 소개**:  
  > 멈춰있는 모델이 아닌, 수집된 데이터로 매주 스스로 성장하는 지능형 SOS CCTV 시스템

---

## 💡 프로젝트 배경
- **실시간성**: 위험 상황 발생 시 즉각적인 감지 및 알림(Discord) 필요  
- **데이터 부족 문제**: 초기 데이터만으로 다양한 환경 대응 어려움  
- **지속적 개선 (MLOps)**:  
  사용자가 `S` 키로 수집한 데이터를 Airflow가 자동 수거 → 재학습 → 성능 개선

---

## 📊 데이터 파이프라인 및 MLOps

### 1. 데이터 수집 및 관리 전략
- **Raw Data (`data/new`)**
  - 실시간 추론 중 캡쳐된 신규 행동 데이터 (.npy)

- **Gold Data (`data/processed`)**
  - 날짜별 (`YYYY-MM-DD`)로 정제된 학습 데이터셋

- **Data Flywheel**
  - 초기 90개 데이터 → 지속 수집 → 데이터 증가 → Loss 감소 → 성능 향상

---

### 2. MLOps 인사이트
- **학습 결과**
  - 300 Epoch → 최종 Loss **0.0005 (MLflow 기준)**

- **클래스 가중치**
  - SOS 상황(Label 0)에 **가중치 5.0 부여**
  - → 위험 감지 민감도 강화

---

## 🛠️ 기술 스택 (Tech Stack)

| 구분 | 기술 | 활용 |
|------|------|------|
| AI/DL | PyTorch, MediaPipe | ActionTransformer 기반 행동 인식 |
| MLOps | Apache Airflow | 데이터 이동, 검증, 재학습 자동화 |
| Tracking | MLflow | 실험 및 Loss 추적 |
| Backend | Python 3.12 | 전체 시스템 로직 |
| Infra | macOS (M3) | 로컬 고성능 환경 |
| Alert | Discord Webhook | 위험 상황 알림 |

---

## 💡 주요 기능 (Key Features)

### 1️⃣ 실시간 행동 추론 및 데이터 수집 (`main_inference.py`)
- MediaPipe Pose 기반 **17개 관절 추출**
- 60프레임 단위 행동 분석
- `S` 키 입력 시 데이터 저장 (`data/new`)
- 위험 상황 발생 시 **Discord 알림 전송**

---

### 2️⃣ Airflow 자동화 파이프라인 (`soscctv_workflow.py`)
- `data/new` → `data/processed/YYYY-MM-DD/` 자동 정리
- 주기적 재학습 스케줄링
- 데이터 기반 모델 지속 개선

---

### 3️⃣ MLflow 실험 관리
- Loss 변화 실시간 추적
- 하이퍼파라미터 기록 (LR, Batch Size 등)
- 최적 모델 자동 저장 (`action_model.pth`)

---

## 📂 프로젝트 구조 (Directory Structure)

📂 SOScctv
 ┣ 📂 airflow/              # Airflow 환경 및 DAG 설정
 ┃ ┗ 📂 dags/
 ┃   ┗ 📜 soscctv_workflow.py
 ┣ 📂 data/                 # 데이터 저장소
 ┃ ┣ 📂 new/                # 캡쳐된 신규 데이터 (Raw)
 ┃ ┗ 📂 processed/          # 날짜별 정제 데이터 (Gold)
 ┣ 📂 models/               # 모델 정의 및 가중치 파일
 ┃ ┣ 📜 model.py            # ActionTransformer 구조
 ┃ ┗ 📜 action_model.pth    # 최신 학습 가중치
 ┣ 📂 scripts/              
 ┃ ┗ 📜 train.py            # MLflow 연동 재학습 스크립트
 ┣ 📜 main_inference.py     # 실시간 추론 및 수집 메인 프로그램
 ┣ 📜 mlflow.db             # MLflow 메타데이터 DB
 ┗ 📜 requirements.txt      # 의존성 패키지 목록

 ##🚀 트러블 슈팅 (Troubleshooting)
### 🔧 이슈 1: macOS(M3) 환경의 Airflow SIGSEGV 에러
문제: BashOperator 실행 시 macOS 보안 정책으로 인해 프로세스 포크(Fork) 중 세그멘테이션 폴트 발생.

해결: * OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES 환경 변수 설정.

airflow.cfg 내 mp_start_method를 spawn으로 변경하여 해결.

### 🔧 이슈 2: Python 3.12와 MediaPipe/Protobuf 호환성
문제: AttributeError: 'SymbolDatabase' object has no attribute 'GetPrototype' 발생.

원인: Protobuf 5.x 버전과의 인터페이스 불일치.

해결: pip install "protobuf<5.0.0"으로 버전을 고정하여 라이브러리 간 통신 정상화.

### 🔧 이슈 3: Airflow 실행 시 상대 경로 인식 불가
문제: 터미널과 달리 Airflow 스케줄러 환경에서 data/ 경로를 찾지 못해 학습 실패.

해결: os.path.abspath(__file__)를 활용한 프로젝트 루트 기준 절대 경로 시스템 도입으로 실행 환경 독립성 확보.
