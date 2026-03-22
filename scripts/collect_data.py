import cv2
import torch
import numpy as np
import os
import datetime # 시간 정밀도를 높이기 위해 추가

# 저장 폴더 강제 생성
SAVE_DIR = "data/processed"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# MediaPipe 임포트 (안전한 방식)
try:
    from mediapipe.python.solutions import pose as mp_pose
except ImportError:
    import mediapipe.solutions.pose as mp_pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

MAX_FRAMES = 60 
COCO_INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

def collect(label_name, label_id):
    cap = cv2.VideoCapture(0)
    # 카메라 설정 최적화
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    saved_count = len([f for f in os.listdir(SAVE_DIR) if f"label_{label_id}" in f])
    
    print(f"--- [{label_name}] 수집 모드 ---")
    print(f"현재 저장된 {label_name} 데이터: {saved_count}개")
    print("'s' 누르면 2초 녹화, 'q' 누르면 종료")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Mode: {label_name} | Count: {saved_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("SOS Collector", display_frame)
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            print("녹화 중...")
            data_buffer = []
            
            for i in range(MAX_FRAMES):
                ret, frame = cap.read()
                if not ret: break
                
                # 시각적인 피드백 (화면에 카운트 표시)
                cv2.putText(frame, f"REC: {i+1}/{MAX_FRAMES}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("SOS Collector", frame)
                cv2.waitKey(1)
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    base_x = (lm[23].x + lm[24].x) / 2
                    base_y = (lm[23].y + lm[24].y) / 2
                    
                    frame_kp = []
                    for idx in COCO_INDICES:
                        frame_kp.extend([lm[idx].x - base_x, lm[idx].y - base_y])
                    data_buffer.append(frame_kp)

            if len(data_buffer) == MAX_FRAMES:
                # 파일명 중복 방지를 위해 상세 시간 사용
                now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                file_name = f"{label_name}_{now}_label_{label_id}.npy"
                np.save(os.path.join(SAVE_DIR, file_name), np.array(data_buffer))
                saved_count += 1
                print(f"저장 성공: {file_name} (총 {saved_count}개)")
            else:
                print("녹화 실패: 관절 인식 프레임 부족")

        elif key == ord('q'): # q 누르면 종료
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 0: normal, 1: fall, 2: help 순서대로 수집하세요.
    collect("normal", 0)