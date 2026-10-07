"""MediaPipe Gesture Recognizer - 웹캠 실시간 손 제스처 인식

인식 가능한 제스처: None, Closed_Fist, Open_Palm, Pointing_Up,
                   Thumb_Down, Thumb_Up, Victory, ILoveYou

실행: python gesture_webcam.py   (종료: q 또는 ESC)
"""
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_recognizer.task")
CAMERA_INDEX = 0
NUM_HANDS = 2

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def draw_result(frame, result):
    h, w = frame.shape[:2]
    for i, landmarks in enumerate(result.hand_landmarks):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for conn in HAND_CONNECTIONS:
            cv2.line(frame, points[conn.start], points[conn.end], (0, 255, 0), 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        x0 = min(p[0] for p in points)
        y0 = min(p[1] for p in points) - 10

        # 제스처 + 좌/우 손 라벨
        lines = []
        if result.gestures and i < len(result.gestures):
            gesture = result.gestures[i][0]
            lines.append(f"{gesture.category_name} {gesture.score:.2f}")
        if result.handedness and i < len(result.handedness):
            lines.append(result.handedness[i][0].category_name)

        for j, text in enumerate(reversed(lines)):
            y = max(y0 - j * 28, 20 + (len(lines) - 1 - j) * 28)
            cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2, cv2.LINE_AA)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    # 경로에 한글이 있으면 MediaPipe가 파일을 못 열기 때문에 바이트로 읽어서 전달
    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()

    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    start = time.monotonic()
    prev = start
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # 거울 모드
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            draw_result(frame, result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}  Hands: {len(result.hand_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("MediaPipe Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
