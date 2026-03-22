import sys
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# MLflow 관련 추가
import mlflow
import mlflow.pytorch

# 1. 경로 문제 해결: 현재 파일(train.py) 위치 기준으로 프로젝트 루트(SOScctv)를 찾음
# scripts/train.py 기준이므로 부모의 부모 폴더가 루트입니다.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.model import ActionTransformer

class MotionDataset(Dataset):
    def __init__(self, data_dir, max_len=60):
        # glob에도 절대 경로를 전달하여 어디서든 파일을 찾게 합니다.
        self.file_list = glob.glob(os.path.join(data_dir, "**", "*.npy"))
        self.max_len = max_len

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path) 
        # 파일명에서 라벨 추출 (예: action_label_1.npy -> 1)
        label = int(file_path.split('_label_')[-1].split('.')[0])
        
        if len(data) < self.max_len:
            pad = np.zeros((self.max_len - len(data), data.shape[1]))
            data = np.vstack([data, pad])
        else:
            data = data[:self.max_len]
        return torch.FloatTensor(data), torch.tensor(label, dtype=torch.long)

def train():
    # 2. 절대 경로 설정
    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "action_model.pth")

    # 3. MLflow 실험(Experiment) 이름 설정
    mlflow.set_tracking_uri("sqlite:////Users/hyojin/Desktop/soscctv/mlflow.db")
    mlflow.set_experiment("SOS_CCTV_Project")

    # 하이퍼파라미터 설정
    learning_rate = 0.0005
    batch_size = 16
    epochs = 300 
    input_dim = 34
    num_classes = 3

    # 데이터셋 로드 (절대 경로 사용)
    dataset = MotionDataset(DATA_DIR)
    
    # 데이터가 0개일 경우 에러를 발생시켜 파이프라인 중단 (안전장치)
    if len(dataset) == 0:
        print(f"❌ 에러: 데이터를 찾을 수 없습니다. 경로 확인: {DATA_DIR}")
        sys.exit(1)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. MLflow 실행(Run) 시작
    with mlflow.start_run():
        mlflow.log_param("lr", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        model = ActionTransformer(input_dim=input_dim, num_classes=num_classes) 
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # SOS 상황(라벨 0)에 더 높은 가중치를 주어 민감하게 감지하도록 설정
        weights = torch.tensor([4.0, 1.5, 1.5]).to('cpu') 
        criterion = nn.CrossEntropyLoss(weight=weights)

        print(f"🚀 학습 시작! 총 데이터: {len(dataset)}개")
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            # 매 에폭마다 메트릭 기록
            mlflow.log_metric("loss", avg_loss, step=epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f"🔥 Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # 5. 모델 저장
        mlflow.pytorch.log_model(model, "action_model")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✅ 학습 완료! 모델 저장 위치: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()

# 1. SIGSEGV (Segmentation Fault)의 정체
# Airflow는 작업을 실행할 때 현재 프로세스를 복제(Fork)해서 자식 프로세스를 만듭니다.
# 문제: macOS는 멀티스레드 환경에서 fork()가 일어나는 것을 보안상 매우 위험하다고 판단합니다.
# 결과: 특히 libarrow나 curl 같은 라이브러리가 초기화될 때 macOS 시스템 설정(프록시 등)에 접근하려고 하면, OS가 즉시 해당 프로세스를 죽여버리고 SIGSEGV 에러를 띄운 것입니다.

# 2. 경로 유실 (num_samples=0)
# 문제: Airflow는 DAG를 실행할 때 우리가 터미널에서 작업하던 위치(SOScctv 폴더)가 아닌 다른 임시 폴더에서 실행할 때가 많습니다.
# 결과: 코드에 data/processed라고 상대 경로로 적어두면, Airflow 입장에서는 그 폴더가 보이지 않아 데이터를 하나도 못 읽어온 것입니다.

# 시스템 환경 변수 (.zshrc): * OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES: macOS의 보안 검사를 우회했습니다.
# no_proxy="*": 네트워크 라이브러리가 시스템 프록시를 건드려 터지는 것을 방지했습니다.
# Airflow 설정 (airflow.cfg):
# mp_start_method = spawn: 프로세스 실행 방식을 fork에서 spawn으로 강제 전환하여 macOS 호환성을 확보했습니다.
# 코드 로직 (train.py):
# 절대 경로 도입: Airflow가 어디서 실행하든 soscctv/data/processed를 정확히 찾도록 os.path.abspath를 사용하여 경로 문제를 해결했습니
