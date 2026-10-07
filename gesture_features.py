"""커스텀 제스처 수집/훈련/추론에서 공통으로 쓰는 랜드마크 → 특징 벡터 변환"""
import os

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "custom_gestures.csv")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "custom_gesture_model.joblib")

NUM_FEATURES = 21 * 3
CSV_HEADER = ["label"] + [f"{axis}{i}" for i in range(21) for axis in "xyz"]

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def create_hand_landmarker(num_hands):
    # 경로에 한글이 있으면 MediaPipe가 파일을 못 열기 때문에 바이트로 읽어서 전달
    with open(HAND_MODEL_PATH, "rb") as f:
        model_bytes = f.read()

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmarks_to_features(landmarks, handedness_name):
    """손 21점을 위치·크기·좌우손에 무관한 63차원 벡터로 변환

    - 손목(0번)을 원점으로 이동
    - 왼손은 x를 뒤집어 오른손 모양으로 통일 (한 손으로만 모아도 양손 인식)
    - 손목에서 가장 먼 점까지의 거리로 나눠 크기 정규화
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    if handedness_name == "Left":
        pts[:, 0] *= -1
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()


def draw_hand(frame, landmarks, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in HAND_CONNECTIONS:
        cv2.line(frame, points[conn.start], points[conn.end], color, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    return points
