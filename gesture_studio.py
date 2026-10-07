"""커스텀 제스처 스튜디오 - 수집 · 훈련 · 실시간 추론을 하나의 창(UI)에서

실행: python gesture_studio.py
  [수집] 라벨 추가 → 목록에서 선택 → 녹화 버튼(또는 Space)
  [훈련] 훈련 시작 버튼 → 결과 확인 (모델 자동 저장)
  [추론] 웹캠에 손을 보이면 예측 결과 표시
"""
import csv
import os
import queue
import threading
import time
import tkinter as tk
from collections import Counter
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2
import joblib
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk

from gesture_features import (CLASSIFIER_PATH, CSV_HEADER, DATASET_PATH,
                              create_hand_landmarker, draw_hand, landmarks_to_features,
                              load_emojis, save_emojis)
from train_gestures import train

CAMERA_INDEX = 0
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"  # 영상 위에 한글 라벨을 그리기 위한 폰트
EMOJI_FONT_PATH = r"C:\Windows\Fonts\seguiemj.ttf"  # 영상 위에 컬러 이모지를 그리기 위한 폰트
DEFAULT_TARGET = 300
DEFAULT_EMOJI_SIZE = 48  # 영상 위 이모지 크기(px), [추론] 탭 슬라이더로 조절
# 이모지 색: (어두운 부분, 중간, 밝은 부분) 3색으로 물들임. 원래 색으로 보려면 None
EMOJI_TINT = ((120, 10, 70), (255, 80, 170), (255, 205, 230))  # 핑크
EMOJI_PALETTE = "👍👎✌👌✊✋🖐☝🤙🤘🤟👋🙏❤"

TAB_COLLECT, TAB_TRAIN, TAB_INFER = 0, 1, 2

_fonts = {}


def font(size, path=FONT_PATH):
    if (path, size) not in _fonts:
        try:
            _fonts[path, size] = ImageFont.truetype(path, size)
        except OSError:
            _fonts[path, size] = ImageFont.load_default()
    return _fonts[path, size]


def clean_emoji(text):
    # 이모지 변형 선택자(U+FE0F)는 PIL에서 빈칸으로 그려지므로 제거
    return text.replace("\ufe0f", "").strip()


_emoji_images = {}


def render_emoji(emoji, size):
    """이모지를 투명 배경 RGBA 이미지로 그림 (EMOJI_TINT 색으로 물들임)"""
    key = (emoji, size)
    if key not in _emoji_images:
        emoji_font = font(size, EMOJI_FONT_PATH)
        left, top, right, bottom = emoji_font.getbbox(emoji)
        image = Image.new("RGBA", (max(right - left, 1), max(bottom - top, 1)), (0, 0, 0, 0))
        ImageDraw.Draw(image).text((-left, -top), emoji, font=emoji_font, embedded_color=True)
        if EMOJI_TINT:
            dark, mid, light = EMOJI_TINT
            tinted = ImageOps.colorize(image.convert("L"), black=dark, mid=mid, white=light).convert("RGBA")
            tinted.putalpha(image.getchannel("A"))
            image = tinted
        _emoji_images[key] = image
    return _emoji_images[key]


def draw_overlay(image, pos, text, color, emoji="", size=28, emoji_size=DEFAULT_EMOJI_SIZE):
    """[이모지] + 한글 텍스트를 한 줄로 그림 (텍스트는 이모지 높이의 가운데에 맞춤)"""
    x, y = pos
    if emoji:
        emoji_image = render_emoji(emoji, emoji_size)
        image.alpha_composite(emoji_image, (x, y))
        x += emoji_image.width + 6
        y += max((emoji_image.height - size) // 2, 0)
    ImageDraw.Draw(image).text((x, y), text, font=font(size), fill=color,
                               stroke_width=2, stroke_fill=(0, 0, 0))


def read_dataset_rows():
    if not os.path.exists(DATASET_PATH):
        return []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # 헤더
        return list(reader)


def append_dataset_rows(rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    is_new = not os.path.exists(DATASET_PATH)
    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def delete_label_rows(label):
    rows = [row for row in read_dataset_rows() if row[0] != label]
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


class GestureStudio:
    def __init__(self, root):
        self.root = root
        root.title("Custom Gesture Studio")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("오류", "웹캠을 열 수 없습니다.")
            root.after(0, root.destroy)
            return
        self.landmarker = create_hand_landmarker(num_hands=2)
        self.start = time.monotonic()
        self.last_ts = -1
        self.prev_time = self.start

        # 수집 상태
        self.counts = Counter(row[0] for row in read_dataset_rows())
        self.labels = list(self.counts)
        self.emojis = load_emojis()
        self.recording = False
        self.session_count = 0
        self.pending_rows = []

        # 훈련/추론 상태
        self.classifier = None
        self.train_log = queue.Queue()
        self.training = False

        self.build_ui()
        self.refresh_label_list()
        self.load_classifier()
        self.update_frame()

    # ------------------------------------------------------------------ UI
    def build_ui(self):
        style = ttk.Style()
        style.configure("Big.TLabel", font=("Malgun Gothic", 18, "bold"))
        style.configure("Rec.TButton", font=("Malgun Gothic", 12, "bold"))

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.video = ttk.Label(left)
        self.video.pack()
        self.status = ttk.Label(left, text="", foreground="gray")
        self.status.pack(anchor="w", pady=(4, 0))

        self.notebook = ttk.Notebook(main, width=340)
        self.notebook.pack(side="right", fill="both", padx=(8, 0))
        self.notebook.add(self.build_collect_tab(), text="  1. 수집  ")
        self.notebook.add(self.build_train_tab(), text="  2. 훈련  ")
        self.notebook.add(self.build_infer_tab(), text="  3. 추론  ")
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        self.bind_space(self.root)

    def bind_space(self, widget):
        """Space를 녹화 전용 키로 사용

        버튼·리스트에 포커스가 있으면 Space가 버튼 클릭/항목 선택으로도 처리돼
        녹화가 켜졌다 바로 꺼지므로, 각 위젯에 직접 바인딩하고 "break"로 기본 동작을 막음
        """
        if not isinstance(widget, (tk.Entry, ttk.Entry, ttk.Spinbox)):
            widget.bind("<space>", self.on_space)
        for child in widget.winfo_children():
            self.bind_space(child)

    def build_collect_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)

        ttk.Label(tab, text="제스처 라벨").pack(anchor="w")
        add_row = ttk.Frame(tab)
        add_row.pack(fill="x", pady=(2, 6))
        self.label_entry = ttk.Entry(add_row)
        self.label_entry.pack(side="left", fill="x", expand=True)
        self.label_entry.bind("<Return>", lambda e: self.add_label())
        ttk.Button(add_row, text="추가", width=6, command=self.add_label).pack(side="left", padx=(4, 0))

        self.label_list = tk.Listbox(tab, height=10, font=("Malgun Gothic", 11),
                                     exportselection=False, activestyle="none")
        self.label_list.pack(fill="both", expand=True)
        self.label_list.bind("<<ListboxSelect>>", lambda e: self.stop_recording())

        emoji_row = ttk.Frame(tab)
        emoji_row.pack(fill="x", pady=(6, 2))
        ttk.Label(emoji_row, text="선택한 라벨의 이모지").pack(side="left")
        ttk.Button(emoji_row, text="지우기", width=6,
                   command=lambda: self.set_emoji("")).pack(side="right")
        ttk.Button(emoji_row, text="적용", width=5,
                   command=lambda: self.set_emoji(self.emoji_entry.get())).pack(side="right", padx=(4, 4))
        self.emoji_entry = ttk.Entry(emoji_row, width=5, font=("Segoe UI Emoji", 11))
        self.emoji_entry.pack(side="right")
        self.emoji_entry.bind("<Return>", lambda e: self.set_emoji(self.emoji_entry.get()))

        palette = ttk.Frame(tab)
        palette.pack(fill="x")
        for i, emoji in enumerate(EMOJI_PALETTE):
            tk.Button(palette, text=emoji, font=("Segoe UI Emoji", 12), relief="flat", width=2,
                      command=lambda e=emoji: self.set_emoji(e)).grid(row=i // 7, column=i % 7)
        ttk.Label(tab, foreground="gray", text="다른 이모지는 입력칸에서 Win + . 으로 고르세요").pack(anchor="w")

        ttk.Button(tab, text="선택한 라벨과 데이터 삭제", command=self.delete_label).pack(fill="x", pady=(4, 10))

        target_row = ttk.Frame(tab)
        target_row.pack(fill="x")
        ttk.Label(target_row, text="녹화 1회당 자동 정지 개수").pack(side="left")
        self.target_var = tk.IntVar(value=DEFAULT_TARGET)
        ttk.Spinbox(target_row, from_=10, to=5000, increment=50, width=7,
                    textvariable=self.target_var).pack(side="right")

        self.rec_button = ttk.Button(tab, text="● 녹화 시작 (Space)", style="Rec.TButton",
                                     command=self.toggle_recording)
        self.rec_button.pack(fill="x", pady=(8, 4), ipady=6)
        self.rec_progress = ttk.Progressbar(tab, maximum=DEFAULT_TARGET)
        self.rec_progress.pack(fill="x")

        ttk.Label(tab, foreground="gray", wraplength=310, justify="left", text=(
            "팁\n"
            "· 'none'(아무 제스처 아닌 손) 라벨을 꼭 함께 모으세요.\n"
            "· 라벨당 200~500개, 녹화 중 손 각도·거리를 조금씩 바꾸세요.\n"
            "· 왼손/오른손은 자동으로 통일되니 한 손으로만 모아도 됩니다."
        )).pack(anchor="w", pady=(10, 0))
        return tab

    def build_train_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.train_button = ttk.Button(tab, text="훈련 시작", style="Rec.TButton", command=self.start_training)
        self.train_button.pack(fill="x", ipady=6)
        self.train_progress = ttk.Progressbar(tab, mode="indeterminate")
        self.train_progress.pack(fill="x", pady=6)
        self.train_result = ttk.Label(tab, text="", style="Big.TLabel")
        self.train_result.pack(anchor="w")
        self.log_text = ScrolledText(tab, height=20, width=40, font=("Consolas", 9), state="disabled")
        self.log_text.pack(fill="both", expand=True, pady=(6, 0))
        return tab

    def build_infer_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.model_status = ttk.Label(tab, text="", wraplength=310)
        self.model_status.pack(anchor="w")
        ttk.Button(tab, text="모델 다시 불러오기", command=self.load_classifier).pack(fill="x", pady=(4, 10))

        conf_row = ttk.Frame(tab)
        conf_row.pack(fill="x")
        ttk.Label(conf_row, text="최소 확신도").pack(side="left")
        self.conf_label = ttk.Label(conf_row, text="0.70")
        self.conf_label.pack(side="right")
        self.conf_var = tk.DoubleVar(value=0.7)
        ttk.Scale(tab, from_=0.0, to=1.0, variable=self.conf_var,
                  command=lambda v: self.conf_label.config(text=f"{float(v):.2f}")).pack(fill="x")

        size_row = ttk.Frame(tab)
        size_row.pack(fill="x", pady=(10, 0))
        ttk.Label(size_row, text="이모지 크기").pack(side="left")
        self.emoji_size_label = ttk.Label(size_row, text=f"{DEFAULT_EMOJI_SIZE}px")
        self.emoji_size_label.pack(side="right")
        self.emoji_size_var = tk.IntVar(value=DEFAULT_EMOJI_SIZE)
        ttk.Scale(tab, from_=16, to=200, variable=self.emoji_size_var,
                  command=self.on_emoji_size).pack(fill="x")

        ttk.Separator(tab).pack(fill="x", pady=12)
        self.infer_emoji = ttk.Label(tab)  # 이모지를 핑크로 칠한 이미지로 표시
        self.infer_emoji_key = None
        self.infer_emoji.pack()
        self.infer_result = ttk.Label(tab, text="손을 보여주세요", style="Big.TLabel", justify="left")
        self.infer_result.pack(anchor="w")
        return tab

    # -------------------------------------------------------------- 수집
    def refresh_label_list(self):
        selected = self.selected_label()
        self.label_list.delete(0, "end")
        for label in self.labels:
            emoji = self.emojis.get(label, "")
            self.label_list.insert("end", f"{emoji + ' ' if emoji else ''}{label}   ({self.counts[label]})")
        if selected in self.labels:
            self.label_list.selection_set(self.labels.index(selected))
        elif self.labels:
            self.label_list.selection_set(0)

    def selected_label(self):
        sel = self.label_list.curselection()
        return self.labels[sel[0]] if sel else None

    def add_label(self):
        label = self.label_entry.get().strip().replace(",", "_")
        if not label:
            return
        if label not in self.labels:
            self.labels.append(label)
        self.label_entry.delete(0, "end")
        self.refresh_label_list()
        self.label_list.selection_clear(0, "end")
        self.label_list.selection_set(self.labels.index(label))
        self.root.focus_set()  # Space가 입력창이 아닌 녹화로 가도록

    def delete_label(self):
        label = self.selected_label()
        if label is None:
            return
        if not messagebox.askyesno("삭제 확인", f"'{label}' 라벨과 수집된 {self.counts[label]}개 샘플을 삭제할까요?"):
            return
        self.stop_recording()
        if self.counts[label]:
            delete_label_rows(label)
        self.labels.remove(label)
        del self.counts[label]
        if self.emojis.pop(label, None) is not None:
            save_emojis(self.emojis)
        self.refresh_label_list()

    def set_emoji(self, emoji):
        label = self.selected_label()
        if label is None:
            messagebox.showinfo("안내", "먼저 라벨을 선택하세요.")
            return
        emoji = clean_emoji(emoji)
        if emoji:
            self.emojis[label] = emoji
        else:
            self.emojis.pop(label, None)
        save_emojis(self.emojis)
        self.emoji_entry.delete(0, "end")
        self.refresh_label_list()
        self.root.focus_set()  # Space가 입력창이 아닌 녹화로 가도록

    def on_space(self, event):
        if self.notebook.index("current") == TAB_COLLECT:
            self.toggle_recording()
        return "break"

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
            return
        if self.selected_label() is None:
            messagebox.showinfo("안내", "먼저 라벨을 추가하고 선택하세요.")
            return
        try:
            target = max(1, int(self.target_var.get()))
        except (tk.TclError, ValueError):
            target = DEFAULT_TARGET
        self.recording = True
        self.session_count = 0
        self.rec_progress.config(maximum=target, value=0)
        self.rec_button.config(text="■ 녹화 정지 (Space)")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.rec_button.config(text="● 녹화 시작 (Space)")
            self.refresh_label_list()
        append_dataset_rows(self.pending_rows)
        self.pending_rows = []

    def on_tab_changed(self, event):
        self.stop_recording()

    # -------------------------------------------------------------- 훈련
    def start_training(self):
        if self.training:
            return
        self.stop_recording()
        self.training = True
        self.train_button.config(state="disabled")
        self.train_result.config(text="훈련 중...", foreground="")
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self.train_progress.start(10)
        threading.Thread(target=self.train_worker, daemon=True).start()
        self.poll_train_log()

    def train_worker(self):
        try:
            acc = train(log=lambda msg: self.train_log.put(("log", msg)))
            self.train_log.put(("done", acc))
        except Exception as e:  # 데이터 부족 등은 화면에 보여주기
            self.train_log.put(("error", str(e)))

    def poll_train_log(self):
        while not self.train_log.empty():
            kind, value = self.train_log.get()
            if kind == "log":
                self.log_text.config(state="normal")
                self.log_text.insert("end", value + "\n")
                self.log_text.see("end")
                self.log_text.config(state="disabled")
                continue
            self.training = False
            self.train_progress.stop()
            self.train_button.config(state="normal")
            if kind == "done":
                self.train_result.config(text=f"테스트 정확도 {value * 100:.1f}%", foreground="green")
                self.load_classifier()
            else:
                self.train_result.config(text="훈련 실패", foreground="red")
                messagebox.showerror("훈련 실패", value)
            return
        self.root.after(100, self.poll_train_log)

    # -------------------------------------------------------------- 추론
    def load_classifier(self):
        if not os.path.exists(CLASSIFIER_PATH):
            self.classifier = None
            self.model_status.config(text="모델이 없습니다. 먼저 [훈련] 탭에서 훈련하세요.", foreground="red")
            return
        self.classifier = joblib.load(CLASSIFIER_PATH)
        classes = ", ".join(str(c) for c in self.classifier.classes_)
        self.model_status.config(text=f"모델 로드됨: {classes}", foreground="green")

    def predict(self, landmarks, hand):
        probs = self.classifier.predict_proba([landmarks_to_features(landmarks, hand)])[0]
        best = probs.argmax()
        label = str(self.classifier.classes_[best]) if probs[best] >= self.conf_var.get() else "?"
        return label, probs[best]

    def on_emoji_size(self, value):
        size = int(float(value))
        self.emoji_size_var.set(size)
        self.emoji_size_label.config(text=f"{size}px")

    def show_infer_emojis(self, emojis):
        """[추론] 탭에 큰 이모지 표시 (바뀔 때만 다시 그림)"""
        key = (tuple(emojis), self.emoji_size_var.get())
        if key == self.infer_emoji_key:
            return
        self.infer_emoji_key = key
        if not emojis:
            self.infer_emoji_photo = None
            self.infer_emoji.config(image="")
            return
        images = [render_emoji(e, key[1]) for e in emojis]
        gap = 10
        canvas = Image.new("RGBA", (sum(i.width for i in images) + gap * (len(images) - 1),
                                    max(i.height for i in images)), (0, 0, 0, 0))
        x = 0
        for i in images:
            canvas.alpha_composite(i, (x, 0))
            x += i.width + gap
        self.infer_emoji_photo = ImageTk.PhotoImage(canvas)
        self.infer_emoji.config(image=self.infer_emoji_photo)

    # -------------------------------------------------------------- 메인 루프
    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.status.config(text="카메라 영상을 읽을 수 없습니다. 다른 프로그램(Zoom, 다른 웹캠 창 등)이 카메라를 쓰고 있는지 확인하세요.",
                               foreground="red")
            self.after_id = self.root.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)  # 거울 모드
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int((time.monotonic() - self.start) * 1000), self.last_ts + 1)
        self.last_ts = ts
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        tab = self.notebook.index("current")
        overlays = []  # (위치, 글자, 색, 이모지) - 한글·이모지 표시를 위해 PIL로 나중에 그림
        hands = list(zip(result.hand_landmarks, result.handedness))
        if tab == TAB_COLLECT:
            hands = hands[:1]  # 수집은 한 손만

        infer_lines = []
        infer_emojis = []
        for landmarks, handedness in hands:
            hand = handedness[0].category_name
            points = draw_hand(frame, landmarks)
            if tab == TAB_COLLECT and self.recording:
                features = landmarks_to_features(landmarks, hand)
                label = self.selected_label()
                self.pending_rows.append([label] + [f"{v:.5f}" for v in features])
                self.counts[label] += 1
                self.session_count += 1
            elif tab == TAB_INFER and self.classifier is not None:
                label, prob = self.predict(landmarks, hand)
                x0 = min(p[0] for p in points)
                emoji = self.emojis.get(label, "")
                y0 = max(min(p[1] for p in points) - (self.emoji_size_var.get() if emoji else 28) - 12, 0)
                overlays.append(((x0, y0), f"{label} {prob:.2f}", (255, 255, 0), emoji))
                infer_lines.append(f"{hand}: {emoji + ' ' if emoji else ''}{label} ({prob:.2f})")
                infer_emojis.append(emoji)

        if tab == TAB_COLLECT:
            self.update_collect_overlay(overlays, bool(hands))
        elif tab == TAB_INFER:
            self.infer_result.config(text="\n".join(infer_lines) or "손을 보여주세요")
            self.show_infer_emojis([e for e in infer_emojis if e])

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        for pos, text, color, emoji in overlays:
            draw_overlay(image, pos, text, color, emoji, emoji_size=self.emoji_size_var.get())
        self.photo = ImageTk.PhotoImage(image)  # 참조를 유지해야 화면에 남음
        self.video.config(image=self.photo)

        now = time.monotonic()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        self.status.config(foreground="gray", text=f"FPS {fps:.1f}   손 {len(result.hand_landmarks)}개   "
                                f"데이터: {os.path.relpath(DATASET_PATH)}")
        self.after_id = self.root.after(1, self.update_frame)

    def update_collect_overlay(self, overlays, hand_visible):
        label = self.selected_label()
        if self.recording:
            target = int(self.rec_progress.cget("maximum"))
            self.rec_progress.config(value=self.session_count)
            if self.session_count % 30 == 0:
                self.refresh_label_list()
            overlays.append(((10, 8), f"● REC  {label}  {self.session_count}/{target}", (255, 60, 60),
                             self.emojis.get(label, "")))
            if not hand_visible:
                overlays.append(((10, 48), "손이 보이지 않음", (255, 200, 0), ""))
            if self.session_count >= target:
                self.stop_recording()
        elif label is not None:
            overlays.append(((10, 8), f"대기 중  {label}  (Space로 녹화)", (220, 220, 220),
                             self.emojis.get(label, "")))

    def on_close(self):
        self.root.after_cancel(self.after_id)
        self.stop_recording()
        self.cap.release()
        self.landmarker.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    GestureStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
