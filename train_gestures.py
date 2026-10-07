"""커스텀 제스처 분류기 훈련 - dataset/custom_gestures.csv → custom_gesture_model.joblib

실행: python train_gestures.py
"""
import csv
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from gesture_features import CLASSIFIER_PATH, DATASET_PATH

MIN_SAMPLES_PER_LABEL = 20


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"데이터가 없습니다. 먼저 collect_gestures.py로 수집하세요: {DATASET_PATH}")
    labels, features = [], []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 헤더
        for row in reader:
            labels.append(row[0])
            features.append([float(v) for v in row[1:]])
    return np.array(features, dtype=np.float32), np.array(labels)


def build_model():
    return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000,
                         early_stopping=True, random_state=0)


def train(log=print):
    """데이터를 읽어 훈련·평가하고 모델을 저장. 진행 상황은 log 함수로 출력"""
    X, y = load_dataset()
    counts = Counter(y)
    log(f"라벨별 샘플 수: {dict(counts)}")

    too_few = [label for label, n in counts.items() if n < MIN_SAMPLES_PER_LABEL]
    if too_few:
        raise ValueError(f"샘플이 {MIN_SAMPLES_PER_LABEL}개 미만인 라벨이 있습니다: {too_few}")
    if len(counts) < 2:
        raise ValueError("라벨이 2개 이상 있어야 합니다. (예: 원하는 제스처 + none)")

    # 1) 80%로 훈련, 20%로 성능 확인
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0)
    model = build_model().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    log("\n=== 테스트 결과 ===")
    log(classification_report(y_test, y_pred, digits=3))
    log("혼동 행렬 (행: 정답, 열: 예측)")
    log(f"labels: {[str(c) for c in model.classes_]}")
    log(str(confusion_matrix(y_test, y_pred, labels=model.classes_)))

    # 2) 전체 데이터로 다시 훈련해서 저장
    final_model = build_model().fit(X, y)
    joblib.dump(final_model, CLASSIFIER_PATH)
    log(f"\n모델 저장 완료: {CLASSIFIER_PATH}")
    return accuracy_score(y_test, y_pred)


def main():
    train()


if __name__ == "__main__":
    main()
