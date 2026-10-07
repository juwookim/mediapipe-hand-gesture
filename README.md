# MediaPipe Hand, Gesture & Face Webcam

MediaPipe Tasks API로 웹캠 영상에서 손 랜드마크·손 제스처·얼굴 랜드마크를 실시간으로 인식하는 파이썬 예제입니다.

## 작업 내역 (2026-10-07)

1. **Hand Landmarker** — `hand_landmarker.task` 모델을 받고, 웹캠에서 손 21개 관절점과 좌/우 손을 표시하는 `hand_webcam.py`를 작성했습니다.
2. **Gesture Recognizer** — [MediaPipe Gesture Recognizer 공식 문서](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer)에 있는 `gesture_recognizer.task`(float16) 모델을 Claude in Chrome으로 받고, 웹캠에서 제스처를 실시간으로 인식하는 `gesture_webcam.py`를 작성했습니다.
3. **Face Landmarker** — [MediaPipe Face Landmarker 공식 문서](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)에 있는 `face_landmarker.task`(float16) 모델을 Claude in Chrome으로 받고, 웹캠에서 얼굴 478점 메시와 표정 계수(blendshape)를 보여주는 `face_webcam.py`를 작성했습니다.
4. 경로에 한글이 있을 때 모델 파일을 못 여는 문제를 해결했습니다(아래 참고).

## 파일 구성

| 파일 | 설명 |
|---|---|
| `hand_webcam.py` | 손 랜드마크 검출 (관절점·뼈대, Left/Right 라벨, FPS) |
| `gesture_webcam.py` | 제스처 인식 (뼈대 + 제스처 이름·점수, Left/Right 라벨, FPS) |
| `face_webcam.py` | 얼굴 랜드마크 (478점 메시·눈/입/눈썹/홍채 윤곽, 상위 8개 blendshape 막대, FPS) |
| `hand_landmarker.task` | Hand Landmarker 모델 |
| `gesture_recognizer.task` | Gesture Recognizer 모델 |
| `face_landmarker.task` | Face Landmarker 모델 |

## 인식 가능한 제스처

`Closed_Fist` ✊ · `Open_Palm` ✋ · `Pointing_Up` ☝️ · `Thumb_Down` 👎 · `Thumb_Up` 👍 · `Victory` ✌️ · `ILoveYou` 🤟 (해당 없으면 `None`)

## 실행 방법

```bash
pip install mediapipe opencv-python

python hand_webcam.py      # 손 랜드마크
python gesture_webcam.py   # 제스처 인식
python face_webcam.py      # 얼굴 랜드마크 (m: 메시 표시 전환)
```

종료하려면 `q` 또는 `ESC`를 누르세요.

테스트 환경: Windows 10, Python 3.14, mediapipe 1.1.0, opencv-python 5.0

## 참고 사항

- **한글 경로 문제**: 폴더 경로에 한글이 있으면 MediaPipe가 `model_asset_path`로 모델을 열지 못합니다. 그래서 파일을 바이트로 읽어 `model_asset_buffer`로 넘깁니다.
- **실행 모드**: `RunningMode.VIDEO`를 쓰고, 프레임마다 증가하는 타임스탬프(ms)를 넘깁니다.
- **화면**: 거울 모드(좌우 반전)로 표시합니다. Windows에서는 `cv2.CAP_DSHOW` 백엔드를 씁니다.
- 실행할 때 나오는 `Feedback manager…`, `NORM_RECT without IMAGE_DIMENSIONS` 경고는 무시해도 됩니다.
- 웹캠이 여러 대라면 스크립트의 `CAMERA_INDEX`를 바꾸세요.

## 모델 출처

- Face Landmarker: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- Gesture Recognizer: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
- Hand Landmarker: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
