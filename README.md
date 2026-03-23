# 🚨 AI 기반 실시간 SOS 행동 인식 및 MLOps 자가 학습 시스템
**"CCTV 영상을 통한 위험 상황(낙상, 도움 요청) 감지 및 데이터 피드백 루프 기반 모델 고도화 서비스"**

---

## 📅 프로젝트 개요
- **프로젝트 기간**: 2026.03.20 ~ 2026.03.22  
- **개발자**: 심효진 (@hyojin-shj)  
- **한 줄 소개**:  
  > 멈춰있는 모델이 아닌, 수집된 데이터로 매주 스스로 성장하는 지능형 SOS CCTV 시스템
  
### 📊 실시간 행동 인식 결과 (Real-time Inference)

| **Normal (상시 모니터링)** | **Fall (낙상 감지)** | **Help (도움 요청)** |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/aafd52a5-25b4-4ea6-96b9-96c3f0449387" width="300px" height="300px" style="object-fit: cover; border-radius: 8px;" alt="normal" /> | <img src="https://github.com/user-attachments/assets/6c93ce44-c673-45a7-ba29-dbca18e5c561" width="300px" height="300px" style="object-fit: cover; border-radius: 8px;" alt="fall" /> | <img src="https://github.com/user-attachments/assets/eb6e4f80-3bb2-4aa6-b0ba-d23367025d26" width="300px" height="300px" style="object-fit: cover; border-radius: 8px;" alt="help" /> |
| 녹색 텍스트로 안정 상태 표시 | 빨간색 텍스트로 위험 알림 전송 | 디스코드 웹훅으로 스크린샷 전송 |

## 💡 프로젝트 배경
- **실시간성**: 위험 상황 발생 시 즉각적인 감지 및 알림(Discord) 필요  
- **데이터 부족 문제**: 초기 데이터만으로 다양한 환경 대응 어려움  
- **지속적 개선 (MLOps)**:  
  사용자가 `S` 키로 수집한 데이터를 Airflow가 자동 수거 → 재학습 → 성능 개선

---

## 📊 데이터 파이프라인 및 MLOps

### 1. 데이터 수집 및 관리 전략
- **초기 학습 데이터 (data/processed)**
 	•	직접 수집 및 라벨링한 행동 데이터
	•	클래스 구성:
	•	0: Normal
	•	1: Fall
	•	2: Help
	•	각 클래스별 30개 샘플로 구성 (총 90개)
	•	모델의 초기 학습 및 기준 성능 확보에 사용

- **실시간 수집 데이터 (data/new)**
  •	실시간 추론 중 사용자가 S 키 입력 시 캡쳐되는 신규 행동 데이터 (.npy)
	•	아직 정제되지 않은 Raw Data
	•	추가 학습을 위한 후보 데이터로 활용

- **데이터 처리 및 학습 파이프라인**
  • 실시간 추론 중 데이터 수집 → data/new 저장
	• 일정 주기 또는 트리거 발생 시 Airflow 파이프라인 실행
	• data/new 데이터를 검수/정제 후
  • → 날짜 기반 디렉토리 (YYYY-MM-DD)로 data/processed에 이동
	• 모델 재학습 수행
	• 학습 완료 후 data/new 디렉토리는 초기화 (비워짐)

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

## 📂 Project Structure

```
SOScctv/
├── airflow/              # Airflow 환경 및 DAG 설정
│   └── dags/
│       └── soscctv_workflow.py
├── data/                 # 데이터 저장소
│   ├── new/              # 캡쳐된 신규 데이터 (Raw)
│   └── processed/        # 날짜별 정제 데이터 (Seed)
├── models/               # 모델 정의 및 가중치 파일
│   ├── model.py          # ActionTransformer 구조
│   └── action_model.pth  # 최신 학습 가중치
├── scripts/
│   └── train.py          # MLflow 연동 재학습 스크립트
├── main_inference.py     # 실시간 추론 및 수집 메인 프로그램
├── mlflow.db             # MLflow 메타데이터 DB
└── requirements.txt      # 의존성 패키지 목록
```

## 🚀 트러블 슈팅 (Troubleshooting)

### 🔧 이슈 1: 데이터셋 수집 시 MediaPipe 관절 인식 한계
문제: 초기 데이터 수집 과정에서 작은 동작이나 미세한 변화가 제대로 인식되지 않는 문제가 발생.

원인: 촬영 거리 및 관절 인식 영역이 작아 MediaPipe가 일부 동작을 안정적으로 추출하지 못함.

해결:
- 다양한 거리, 각도, 조명 환경에서 반복적으로 데이터 수집
- 약 200회 이상의 모션 캡쳐를 통해 데이터 다양성 확보
- 인식 안정성이 높은 샘플만 선별하여 초기 Seed Dataset 구성

---

### 🔧 이슈 2: Python 3.12와 MediaPipe / Protobuf 호환성 문제
문제: 실행 중 아래와 같은 에러 발생
AttributeError: 'SymbolDatabase' object has no attribute 'GetPrototype'

원인: Protobuf 5.x 버전과 MediaPipe 간 인터페이스 호환성 문제

해결:
- Protobuf 버전을 5.x 미만으로 고정하여 해결
- 명령어:
  pip install "protobuf<5.0.0"

---

### 🔧 이슈 3: macOS (M3) 환경에서 Airflow SIGSEGV 에러
문제: BashOperator 실행 시 프로세스 Fork 과정에서 세그멘테이션 폴트 발생

원인: macOS의 보안 정책과 Python multiprocessing 방식 충돌

해결:
- 환경 변수 설정:
  export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
- airflow.cfg 설정 변경:
  mp_start_method = spawn

---

### 🔧 이슈 4: Airflow 실행 시 상대 경로 인식 불가
문제: Airflow 스케줄러 환경에서 data/ 경로를 찾지 못해 학습 실패

원인: 실행 환경이 달라지면서 상대 경로 기준이 변경됨

해결:
- 프로젝트 루트 기준 절대 경로 방식 도입
- 예시 코드:
  import os
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
```
