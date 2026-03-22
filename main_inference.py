import cv2
import torch
import numpy as np
import os
import sys
import time
import requests
from datetime import datetime

# [1] 경로 및 모델 설정 (심요님 환경에 맞춤)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from models.model import ActionTransformer

# MediaPipe 임포트 (가장 안전한 방식)
import mediapipe as mp
mp_pose = mp.solutions.pose

# --- [설정부] ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "action_model.pth")
SAVE_DIR = os.path.join(BASE_DIR, "data", "new") # 에어플로우가 감시하는 곳
os.makedirs(SAVE_DIR, exist_ok=True)

DISCORD_URL = 'https://discord.com/api/webhooks/1484916960021184624/Ba6K9J7pwtZTJ9ysnaZMBFZEBORY6ENO28vWR5JgOBweHU3ZU_UsiSb9kWRx48l4Pyul'
MAX_SEQ_LEN = 60
CLASSES = {0: "Normal", 1: "Fall Detected!", 2: "Help Signal!"}
COCO_INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# 1. 모델 로드
device = torch.device("cpu")
model = ActionTransformer(input_dim=34, num_classes=3).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ 모델 로드 성공: {MODEL_PATH}")
model.eval()

# 2. MediaPipe 및 카메라 초기화
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
sequence_buffer = []

print("🚀 모니터링 중... [S]키: 학습 데이터 저장 (data/new) | [ESC]: 종료")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # 골반 중심 정규화 (아까 잘 작동하던 그 로직!)
        base_x = (lm[23].x + lm[24].x) / 2
        base_y = (lm[23].y + lm[24].y) / 2
        
        current_kp = []
        for idx in COCO_INDICES:
            current_kp.extend([lm[idx].x - base_x, lm[idx].y - base_y])
        
        sequence_buffer.append(current_kp)

        if len(sequence_buffer) > MAX_SEQ_LEN:
            sequence_buffer.pop(0)

        if len(sequence_buffer) == MAX_SEQ_LEN:
            # --- [추론부] ---
            input_tensor = torch.FloatTensor([sequence_buffer]).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)[0]
                prediction = torch.argmax(output, dim=1).item()
            
            # 화면 표시
            status_text = f"{CLASSES[prediction]} ({probs[prediction]*100:.1f}%)"
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow('SOS CCTV Monitoring', frame)
    
    # --- [핵심: S키 저장 로직] ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') or key == ord('S'):
        if len(sequence_buffer) == MAX_SEQ_LEN:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 에어플로우가 인식할 수 있는 파일명 규칙 적용
            file_name = f"captured_{timestamp}_label_{prediction}.npy"
            save_path = os.path.join(SAVE_DIR, file_name)
            
            np.save(save_path, np.array(sequence_buffer))
            print(f"📸 [저장 완료] {file_name} -> data/new 폴더로 이동됨")
        else:
            print("⏳ 60프레임이 쌓일 때까지 조금만 기다려주세요!")

    elif key == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()