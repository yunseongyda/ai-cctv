import cv2
import time
import csv
from ultralytics import YOLO
from playsound import playsound
import threading

# YOLO 모델 로드 (COCO 학습된 기본 모델 사용)
model = YOLO("yolov8n.pt")

# 경고음 재생 함수 (스레드로 실행)
def play_warning():
    # threading.Thread(target=playsound, args=("warning.mp3",)).start()
    return 0

# 로그 저장 함수
def log_event():
    with open("log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        now = time.localtime()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", now)
        writer.writerow([timestamp, "Illegal dumping detected"])

# 웹캠 설정
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not detected")
    exit()

# 영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640, 480))

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    frame = results[0].plot()  # 탐지 결과 그리기

    # 사람(person)과 쓰레기 유사 객체(backpack, handbag 등)를 동시에 탐지했는지 확인
    labels = [x.name for x in results[0].names.values()]
    detected = [labels[int(cls)] for cls in results[0].boxes.cls]

    if "person" in detected and ("backpack" in detected or "handbag" in detected):
        cv2.putText(frame, "Warning: Dumping detected!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # play_warning()
        log_event()
        out.write(frame)

    cv2.imshow("Illegal Dumping Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
