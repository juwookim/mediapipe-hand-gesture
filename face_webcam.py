"""MediaPipe Face Landmarker - 웹캠 실시간 얼굴 랜드마크(478점) + 표정(blendshape) 검출

실행: python face_webcam.py   (종료: q 또는 ESC, 메시 표시 전환: m)
"""
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
CAMERA_INDEX = 0
NUM_FACES = 1
TOP_BLENDSHAPES = 8

FACE = vision.FaceLandmarksConnections
CONTOUR_STYLES = [
    (FACE.FACE_LANDMARKS_FACE_OVAL, (220, 220, 220)),
    (FACE.FACE_LANDMARKS_LIPS, (80, 80, 255)),
    (FACE.FACE_LANDMARKS_LEFT_EYE, (80, 255, 80)),
    (FACE.FACE_LANDMARKS_RIGHT_EYE, (80, 255, 80)),
    (FACE.FACE_LANDMARKS_LEFT_EYEBROW, (80, 200, 255)),
    (FACE.FACE_LANDMARKS_RIGHT_EYEBROW, (80, 200, 255)),
    (FACE.FACE_LANDMARKS_LEFT_IRIS, (255, 255, 0)),
    (FACE.FACE_LANDMARKS_RIGHT_IRIS, (255, 255, 0)),
]


def draw_landmarks(frame, result, show_mesh):
    h, w = frame.shape[:2]
    for landmarks in result.face_landmarks:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        if show_mesh:
            for conn in FACE.FACE_LANDMARKS_TESSELATION:
                cv2.line(frame, points[conn.start], points[conn.end], (110, 110, 110), 1, cv2.LINE_AA)

        for connections, color in CONTOUR_STYLES:
            for conn in connections:
                cv2.line(frame, points[conn.start], points[conn.end], color, 2, cv2.LINE_AA)


def draw_blendshapes(frame, result):
    # 첫 번째 얼굴의 점수가 높은 표정 계수를 막대그래프로 표시
    if not result.face_blendshapes:
        return
    categories = sorted(result.face_blendshapes[0], key=lambda c: c.score, reverse=True)
    x0, y0, bar_w = 10, 60, 150
    for j, c in enumerate(categories[:TOP_BLENDSHAPES]):
        y = y0 + j * 22
        cv2.rectangle(frame, (x0, y), (x0 + int(bar_w * c.score), y + 14), (0, 200, 255), -1)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + 14), (200, 200, 200), 1)
        cv2.putText(frame, f"{c.category_name} {c.score:.2f}", (x0 + bar_w + 8, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    # 경로에 한글이 있으면 MediaPipe가 파일을 못 열기 때문에 바이트로 읽어서 전달
    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=NUM_FACES,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    show_mesh = True
    start = time.monotonic()
    prev = start
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # 거울 모드
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_landmarks(frame, result, show_mesh)
            draw_blendshapes(frame, result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}  Faces: {len(result.face_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("MediaPipe Face Landmarker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_mesh = not show_mesh

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
