"""커스텀 제스처 실시간 추론 - train_gestures.py로 만든 모델을 웹캠에 적용

실행: python custom_gesture_webcam.py   (종료: q 또는 ESC)
"""
import os
import time

import cv2
import joblib
import mediapipe as mp

from gesture_features import (CLASSIFIER_PATH, create_hand_landmarker, draw_hand,
                              landmarks_to_features)

CAMERA_INDEX = 0
NUM_HANDS = 2
MIN_CONFIDENCE = 0.7  # 이보다 낮으면 "?"로 표시


def main():
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"모델이 없습니다. 먼저 train_gestures.py를 실행하세요: {CLASSIFIER_PATH}")
    classifier = joblib.load(CLASSIFIER_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    start = time.monotonic()
    prev = start
    with create_hand_landmarker(num_hands=NUM_HANDS) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # 거울 모드
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                hand = handedness[0].category_name
                features = landmarks_to_features(landmarks, hand)
                probs = classifier.predict_proba([features])[0]
                best = probs.argmax()
                label = classifier.classes_[best] if probs[best] >= MIN_CONFIDENCE else "?"

                points = draw_hand(frame, landmarks)
                x0 = min(p[0] for p in points)
                y0 = min(p[1] for p in points) - 10
                cv2.putText(frame, f"{label} {probs[best]:.2f} ({hand})", (x0, max(y0, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}  Hands: {len(result.hand_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Custom Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
