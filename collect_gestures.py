"""커스텀 제스처 데이터 수집 - 웹캠 손 랜드마크를 라벨과 함께 CSV로 저장

실행: python collect_gestures.py heart ok call none
  - 1~9 : 수집할 라벨 선택 (명령행에 적은 순서)
  - SPACE: 녹화 시작/정지 (녹화 중에는 손이 보이는 프레임마다 1개씩 저장)
  - q/ESC: 종료
저장 위치: dataset/custom_gestures.csv (실행할 때마다 이어서 추가)
"""
import csv
import os
import sys
import time
from collections import Counter

import cv2
import mediapipe as mp

from gesture_features import (CSV_HEADER, DATASET_PATH, create_hand_landmarker,
                              draw_hand, landmarks_to_features)

CAMERA_INDEX = 0


def load_counts():
    if not os.path.exists(DATASET_PATH):
        return Counter()
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        return Counter(row["label"] for row in csv.DictReader(f))


def draw_status(frame, labels, current, recording, counts):
    status = "REC" if recording else "PAUSE"
    color = (0, 0, 255) if recording else (200, 200, 200)
    cv2.putText(frame, f"[{status}] {labels[current]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    for i, label in enumerate(labels):
        marker = ">" if i == current else " "
        cv2.putText(frame, f"{marker}{i + 1}: {label} ({counts[label]})", (10, 65 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    if recording:
        cv2.circle(frame, (frame.shape[1] - 25, 25), 10, (0, 0, 255), -1)


def main():
    labels = sys.argv[1:]
    if not labels or len(labels) > 9:
        print("사용법: python collect_gestures.py <라벨1> <라벨2> ... (최대 9개)")
        print("예시:   python collect_gestures.py heart ok call none")
        sys.exit(1)

    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    is_new = not os.path.exists(DATASET_PATH)
    counts = load_counts()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    current = 0
    recording = False
    start = time.monotonic()
    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f, \
            create_hand_landmarker(num_hands=1) as landmarker:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(CSV_HEADER)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # 거울 모드
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                handedness = result.handedness[0][0].category_name
                draw_hand(frame, landmarks)
                if recording:
                    features = landmarks_to_features(landmarks, handedness)
                    writer.writerow([labels[current]] + [f"{v:.5f}" for v in features])
                    counts[labels[current]] += 1

            draw_status(frame, labels, current, recording, counts)
            cv2.imshow("Collect Custom Gestures", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = not recording
                f.flush()
            if ord("1") <= key < ord("1") + len(labels):
                current = key - ord("1")
                recording = False

    cap.release()
    cv2.destroyAllWindows()
    print("수집된 샘플 수:", dict(counts))


if __name__ == "__main__":
    main()
