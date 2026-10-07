// Gesture Studio (웹 버전) - gesture_studio.py를 브라우저로 옮긴 것
// 영상 처리·수집·훈련·추론이 모두 브라우저 안에서 일어나고, 데이터는 localStorage에만 저장됨
import { FilesetResolver, GestureRecognizer } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

const MIN_SAMPLES_PER_LABEL = 20;
const HIDDEN_LAYERS = [64, 32];
const EPOCHS = 60;
const EMOJI_PALETTE = ["👍", "👎", "✌", "👌", "✊", "✋", "🖐", "☝", "🤙", "🤘", "🤟", "👋", "🙏", "❤"];
const BUILTIN_EMOJIS = {
  Closed_Fist: "✊", Open_Palm: "✋", Pointing_Up: "☝", Thumb_Down: "👎",
  Thumb_Up: "👍", Victory: "✌", ILoveYou: "🤟",
};
// 이모지 색: (어두운 부분, 중간, 밝은 부분) 3색으로 물들임 - gesture_studio.py의 EMOJI_TINT와 같음
const TINTS = { pink: [[120, 10, 70], [255, 80, 170], [255, 205, 230]], none: null };
const EMOJI_FONT = '"Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif';
const TEXT_FONT = '"Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif';
const STORE = { labels: "gs.labels", rows: "gs.rows", emojis: "gs.emojis", model: "gs.model", settings: "gs.settings" };

const $ = (id) => document.getElementById(id);
const canvas = $("stage");
const ctx = canvas.getContext("2d", { willReadFrequently: false });
const video = document.createElement("video");
video.playsInline = true;
video.muted = true;

// ------------------------------------------------------------------ 저장소
function load(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function save(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    setStatus("브라우저 저장 공간이 부족해 저장하지 못했습니다. [JSON 내보내기]로 백업하세요.", true);
    return false;
  }
}

const state = {
  tab: "builtin",
  labels: load(STORE.labels, []),
  rows: load(STORE.rows, []),          // [label, x0, y0, z0, ...] (63개 특징)
  emojis: load(STORE.emojis, {}),
  model: load(STORE.model, null),
  selected: null,
  recording: false,
  sessionCount: 0,
  training: false,
  recognizer: null,
  running: false,
  lastTs: -1,
  prevFrame: performance.now(),
  settings: { emojiSize: 48, tint: "pink", ...load(STORE.settings, {}) },
};
for (const row of state.rows) if (!state.labels.includes(row[0])) state.labels.push(row[0]);
state.selected = state.labels[0] ?? null;

function counts() {
  const c = Object.fromEntries(state.labels.map((l) => [l, 0]));
  for (const row of state.rows) c[row[0]] = (c[row[0]] ?? 0) + 1;
  return c;
}

function saveDataset() {
  save(STORE.labels, state.labels);
  save(STORE.rows, state.rows);
}

// ------------------------------------------------------------------ 특징 변환
/** 손 21점을 위치·크기·좌우손에 무관한 63차원 벡터로 변환 (gesture_features.py와 동일) */
function landmarksToFeatures(landmarks, handedness) {
  const ox = landmarks[0].x, oy = landmarks[0].y, oz = landmarks[0].z;
  const pts = landmarks.map((p) => [p.x - ox, p.y - oy, p.z - oz]);
  if (handedness === "Left") for (const p of pts) p[0] *= -1;
  const scale = Math.max(...pts.map(([x, y, z]) => Math.hypot(x, y, z)));
  const out = new Array(63);
  pts.forEach((p, i) => {
    for (let k = 0; k < 3; k++) out[i * 3 + k] = scale > 0 ? p[k] / scale : p[k];
  });
  return out;
}

// ------------------------------------------------------------------ 이모지 (핑크로 물들이기)
const emojiCache = new Map();

function cleanEmoji(text) {
  return text.replace(/️/g, "").trim();
}

function renderEmoji(emoji, size) {
  const key = `${emoji}|${size}|${state.settings.tint}`;
  if (emojiCache.has(key)) return emojiCache.get(key);

  const c = document.createElement("canvas");
  const g = c.getContext("2d", { willReadFrequently: true });
  g.font = `${size}px ${EMOJI_FONT}`;
  const m = g.measureText(emoji);
  const left = m.actualBoundingBoxLeft || 0;
  const width = Math.ceil(left + (m.actualBoundingBoxRight || m.width)) || size;
  const ascent = m.actualBoundingBoxAscent || size * 0.9;
  const height = Math.ceil(ascent + (m.actualBoundingBoxDescent || size * 0.1)) || size;
  const pad = 2;
  c.width = width + pad * 2;
  c.height = height + pad * 2;
  g.font = `${size}px ${EMOJI_FONT}`;
  g.fillText(emoji, pad + left, pad + ascent);

  const tint = TINTS[state.settings.tint];
  if (tint) {
    const [dark, mid, light] = tint;
    const img = g.getImageData(0, 0, c.width, c.height);
    const d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] === 0) continue;
      const lum = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255;
      const [a, b, t] = lum < 0.5 ? [dark, mid, lum * 2] : [mid, light, (lum - 0.5) * 2];
      for (let k = 0; k < 3; k++) d[i + k] = a[k] + (b[k] - a[k]) * t;
    }
    g.putImageData(img, 0, 0);
  }
  emojiCache.set(key, c);
  return c;
}

function emojiImg(emoji, px = 22) {
  const img = document.createElement("img");
  img.alt = emoji;
  img.src = renderEmoji(emoji, px * 2).toDataURL();
  return img;
}

let shownEmojiKey = { builtin: null, infer: null };
function showBigEmojis(target, emojis) {
  const size = state.settings.emojiSize;
  const key = `${emojis.join(",")}|${size}|${state.settings.tint}`;
  if (shownEmojiKey[target] === key) return;
  shownEmojiKey[target] = key;
  const box = $(target === "builtin" ? "builtinEmoji" : "inferEmoji");
  box.replaceChildren(...emojis.map((e) => {
    const src = renderEmoji(e, size);
    const c = document.createElement("canvas");
    c.width = src.width;
    c.height = src.height;
    c.getContext("2d").drawImage(src, 0, 0);
    return c;
  }));
}

// ------------------------------------------------------------------ 그리기
function drawHand(landmarks) {
  const w = canvas.width, h = canvas.height;
  ctx.strokeStyle = "rgb(0,255,0)";
  ctx.lineWidth = 2;
  for (const { start, end } of GestureRecognizer.HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[start].x * w, landmarks[start].y * h);
    ctx.lineTo(landmarks[end].x * w, landmarks[end].y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "rgb(255,0,0)";
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  const xs = landmarks.map((p) => p.x * w), ys = landmarks.map((p) => p.y * h);
  return { x0: Math.min(...xs), y0: Math.min(...ys) };
}

/** [이모지] + 텍스트를 한 줄로 그림 (텍스트는 이모지 높이의 가운데에 맞춤) */
function drawOverlay(x, y, text, color, emoji = "", size = 28) {
  if (emoji) {
    const img = renderEmoji(emoji, state.settings.emojiSize);
    ctx.drawImage(img, x, y);
    x += img.width + 6;
    y += Math.max((img.height - size) / 2, 0);
  }
  ctx.font = `bold ${size}px ${TEXT_FONT}`;
  ctx.textBaseline = "top";
  ctx.lineWidth = 4;
  ctx.strokeStyle = "black";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function labelTop(y0, emoji) {
  return Math.max(y0 - (emoji ? state.settings.emojiSize : 28) - 12, 0);
}

// ------------------------------------------------------------------ 카메라 + 메인 루프
async function start() {
  const btn = $("startBtn");
  btn.disabled = true;
  try {
    $("startMsg").textContent = "카메라 권한을 요청하는 중…";
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    $("startMsg").textContent = "MediaPipe 모델을 불러오는 중… (처음엔 몇 초 걸려요)";
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    try {
      state.recognizer = await GestureRecognizer.createFromOptions(vision, options("GPU"));
    } catch {
      state.recognizer = await GestureRecognizer.createFromOptions(vision, options("CPU"));
    }
    $("startOverlay").classList.add("hidden");
    state.running = true;
    requestAnimationFrame(loop);
  } catch (err) {
    btn.disabled = false;
    $("startMsg").textContent = err?.name === "NotAllowedError"
      ? "카메라 권한이 거부됐습니다. 주소창의 카메라 아이콘에서 허용한 뒤 다시 눌러주세요."
      : `시작하지 못했습니다: ${err?.message ?? err}`;
  }
}

function loop() {
  if (!state.running) return;
  if (video.readyState >= 2) processFrame();
  requestAnimationFrame(loop);
}

function processFrame() {
  const w = canvas.width, h = canvas.height;
  // 거울 모드로 그린 화면을 그대로 인식 (파이썬 버전의 cv2.flip과 같은 좌표계)
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();

  let ts = performance.now();
  if (ts <= state.lastTs) ts = state.lastTs + 1;
  state.lastTs = ts;
  const result = state.recognizer.recognizeForVideo(canvas, ts);
  const handed = result.handedness ?? result.handednesses ?? [];

  let hands = result.landmarks.map((lm, i) => ({ lm, hand: handed[i]?.[0]?.categoryName ?? "Right", i }));
  if (state.tab === "collect") hands = hands.slice(0, 1); // 수집은 한 손만

  const overlays = [];
  const lines = [];
  const bigEmojis = [];
  for (const { lm, hand, i } of hands) {
    const { x0, y0 } = drawHand(lm);
    if (state.tab === "builtin") {
      const g = result.gestures?.[i]?.[0];
      if (!g) continue;
      const name = g.categoryName || "None";
      const emoji = BUILTIN_EMOJIS[name] ?? "";
      overlays.push([x0, labelTop(y0, emoji), `${name} ${g.score.toFixed(2)}`, "rgb(255,255,0)", emoji]);
      lines.push(`${hand}: ${name} (${g.score.toFixed(2)})`);
      if (emoji) bigEmojis.push(emoji);
    } else if (state.tab === "collect" && state.recording) {
      const f = landmarksToFeatures(lm, hand);
      state.rows.push([state.selected, ...f.map((v) => Math.round(v * 1e5) / 1e5)]);
      state.sessionCount++;
    } else if (state.tab === "infer" && state.model) {
      const { label, prob } = predict(landmarksToFeatures(lm, hand));
      const emoji = state.emojis[label] ?? "";
      overlays.push([x0, labelTop(y0, emoji), `${label} ${prob.toFixed(2)}`, "rgb(255,255,0)", emoji]);
      lines.push(`${hand}: ${label} (${prob.toFixed(2)})`);
      if (emoji) bigEmojis.push(emoji);
    }
  }

  if (state.tab === "collect") collectOverlay(overlays, hands.length > 0);
  if (state.tab === "builtin" || state.tab === "infer") {
    $(state.tab === "builtin" ? "builtinResult" : "inferResult").textContent = lines.join("\n") || "손을 보여주세요";
    showBigEmojis(state.tab, bigEmojis);
  }
  for (const args of overlays) drawOverlay(...args);

  const now = performance.now();
  const fps = 1000 / Math.max(now - state.prevFrame, 1);
  state.prevFrame = now;
  setStatus(`FPS ${fps.toFixed(1)}   손 ${result.landmarks.length}개   수집 데이터 ${state.rows.length}개`);
}

function collectOverlay(overlays, handVisible) {
  const label = state.selected;
  const emoji = label ? state.emojis[label] ?? "" : "";
  if (state.recording) {
    const target = Number($("recProgress").max);
    $("recProgress").value = state.sessionCount;
    if (state.sessionCount % 10 === 0) renderLabelList();
    overlays.push([10, 8, `● REC  ${label}  ${state.sessionCount}/${target}`, "rgb(255,60,60)", emoji]);
    if (!handVisible) overlays.push([10, 8 + (emoji ? state.settings.emojiSize : 28) + 12, "손이 보이지 않음", "rgb(255,200,0)"]);
    if (state.sessionCount >= target) stopRecording();
  } else if (label) {
    overlays.push([10, 8, `대기 중  ${label}  (Space로 녹화)`, "rgb(220,220,220)", emoji]);
  }
}

function setStatus(text, warn = false) {
  const el = $("status");
  el.textContent = text;
  el.style.color = warn ? "var(--danger)" : "";
}

// ------------------------------------------------------------------ 탭
function selectTab(tab) {
  stopRecording();
  state.tab = tab;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".tab-body").forEach((s) => s.classList.toggle("hidden", s.dataset.body !== tab));
  if (tab === "infer") renderModelStatus();
}

// ------------------------------------------------------------------ 수집
function renderLabelList() {
  const c = counts();
  const list = $("labelList");
  if (!state.labels.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "위에서 라벨을 추가하세요";
    list.replaceChildren(li);
    return;
  }
  list.replaceChildren(...state.labels.map((label) => {
    const li = document.createElement("li");
    li.classList.toggle("selected", label === state.selected);
    if (state.emojis[label]) li.append(emojiImg(state.emojis[label]));
    const name = document.createElement("span");
    name.textContent = label;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = c[label] ?? 0;
    li.append(name, count);
    li.addEventListener("click", () => {
      stopRecording();
      state.selected = label;
      renderLabelList();
    });
    return li;
  }));
}

function addLabel() {
  const label = $("labelInput").value.trim().replace(/,/g, "_");
  if (!label) return;
  if (!state.labels.includes(label)) state.labels.push(label);
  state.selected = label;
  $("labelInput").value = "";
  $("labelInput").blur(); // Space가 입력창이 아닌 녹화로 가도록
  saveDataset();
  renderLabelList();
}

function deleteLabel() {
  const label = state.selected;
  if (!label) return;
  const n = counts()[label] ?? 0;
  if (!confirm(`'${label}' 라벨과 수집된 ${n}개 샘플을 삭제할까요?`)) return;
  stopRecording();
  state.rows = state.rows.filter((r) => r[0] !== label);
  state.labels = state.labels.filter((l) => l !== label);
  delete state.emojis[label];
  state.selected = state.labels[0] ?? null;
  saveDataset();
  save(STORE.emojis, state.emojis);
  renderLabelList();
}

function setEmoji(emoji) {
  if (!state.selected) {
    setStatus("먼저 라벨을 선택하세요.", true);
    return;
  }
  emoji = cleanEmoji(emoji);
  if (emoji) state.emojis[state.selected] = emoji;
  else delete state.emojis[state.selected];
  save(STORE.emojis, state.emojis);
  $("emojiInput").value = "";
  $("emojiInput").blur();
  renderLabelList();
}

function toggleRecording() {
  if (state.recording) return stopRecording();
  if (!state.running) return setStatus("먼저 [카메라 켜기]를 누르세요.", true);
  if (!state.selected) return setStatus("먼저 라벨을 추가하고 선택하세요.", true);
  const target = Math.max(1, parseInt($("targetInput").value, 10) || 300);
  state.recording = true;
  state.sessionCount = 0;
  $("recProgress").max = target;
  $("recProgress").value = 0;
  $("recBtn").textContent = "■ 녹화 정지 (Space)";
  $("recBtn").classList.add("recording");
}

function stopRecording() {
  if (!state.recording) return;
  state.recording = false;
  $("recBtn").textContent = "● 녹화 시작 (Space)";
  $("recBtn").classList.remove("recording");
  saveDataset();
  renderLabelList();
}

function exportData() {
  const blob = new Blob([JSON.stringify({ labels: state.labels, emojis: state.emojis, rows: state.rows })],
    { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "custom_gestures.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function importData(file) {
  const text = await file.text();
  let rows = [], emojis = {}, labels = [];
  try {
    if (file.name.toLowerCase().endsWith(".csv")) {
      // gesture_studio.py의 dataset/custom_gestures.csv (label,x0,y0,z0,...)
      const lines = text.split(/\r?\n/).filter(Boolean);
      for (const line of lines.slice(1)) {
        const [label, ...vals] = line.split(",");
        if (vals.length === 63) rows.push([label, ...vals.map(Number)]);
      }
    } else {
      ({ rows = [], emojis = {}, labels = [] } = JSON.parse(text));
    }
  } catch {
    return setStatus("파일을 읽지 못했습니다. 이 페이지나 gesture_studio.py에서 만든 파일인지 확인하세요.", true);
  }
  rows = rows.filter((r) => Array.isArray(r) && r.length === 64);
  if (!rows.length && !labels.length) return setStatus("가져올 데이터가 없습니다.", true);
  state.rows.push(...rows);
  for (const l of [...labels, ...rows.map((r) => r[0])]) if (!state.labels.includes(l)) state.labels.push(l);
  Object.assign(state.emojis, emojis);
  state.selected ??= state.labels[0] ?? null;
  saveDataset();
  save(STORE.emojis, state.emojis);
  renderLabelList();
  setStatus(`${rows.length}개 샘플을 가져왔습니다.`);
}

function clearAll() {
  if (!confirm("수집한 데이터, 라벨, 이모지, 훈련한 모델을 모두 삭제할까요?")) return;
  stopRecording();
  Object.assign(state, { labels: [], rows: [], emojis: {}, model: null, selected: null });
  for (const key of [STORE.labels, STORE.rows, STORE.emojis, STORE.model]) {
    try { localStorage.removeItem(key); } catch { /* 저장소를 못 쓰는 환경 */ }
  }
  renderLabelList();
}

// ------------------------------------------------------------------ 신경망 (MLP)
function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createMLP(sizes, rand) {
  return sizes.slice(1).map((out, i) => {
    const inp = sizes[i];
    const limit = Math.sqrt(6 / inp);
    const W = new Float32Array(inp * out).map(() => (rand() * 2 - 1) * limit);
    return { inp, out, W, b: new Float32Array(out) };
  });
}

function forward(layers, x) {
  const acts = [Float32Array.from(x)];
  layers.forEach((L, li) => {
    const a = acts[acts.length - 1];
    const z = new Float32Array(L.out);
    for (let o = 0; o < L.out; o++) {
      let s = L.b[o];
      const off = o * L.inp;
      for (let i = 0; i < L.inp; i++) s += L.W[off + i] * a[i];
      z[o] = s;
    }
    if (li < layers.length - 1) {
      for (let o = 0; o < L.out; o++) if (z[o] < 0) z[o] = 0;
    } else {
      const m = Math.max(...z);
      let sum = 0;
      for (let o = 0; o < L.out; o++) sum += (z[o] = Math.exp(z[o] - m));
      for (let o = 0; o < L.out; o++) z[o] /= sum;
    }
    acts.push(z);
  });
  return acts;
}

async function fitMLP(X, y, numClasses, onEpoch, seed = 0) {
  const rand = mulberry32(seed + 1);
  const layers = createMLP([63, ...HIDDEN_LAYERS, numClasses], rand);
  const lr = 0.003, b1 = 0.9, b2 = 0.999, eps = 1e-8, alpha = 1e-4, batch = 32;
  const opt = layers.map((L) => ({
    mW: new Float32Array(L.W.length), vW: new Float32Array(L.W.length),
    mb: new Float32Array(L.out), vb: new Float32Array(L.out),
    gW: new Float32Array(L.W.length), gb: new Float32Array(L.out),
  }));
  const idx = [...X.keys()];
  let step = 0;

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    for (let s = 0; s < idx.length; s += batch) {
      const chunk = idx.slice(s, s + batch);
      for (const o of opt) { o.gW.fill(0); o.gb.fill(0); }
      for (const n of chunk) {
        const acts = forward(layers, X[n]);
        let delta = Float32Array.from(acts[acts.length - 1]);
        delta[y[n]] -= 1;
        for (let li = layers.length - 1; li >= 0; li--) {
          const L = layers[li], a = acts[li], g = opt[li];
          const prev = li > 0 ? new Float32Array(L.inp) : null;
          for (let o = 0; o < L.out; o++) {
            const d = delta[o];
            if (d === 0) continue;
            g.gb[o] += d;
            const off = o * L.inp;
            for (let i = 0; i < L.inp; i++) {
              g.gW[off + i] += d * a[i];
              if (prev) prev[i] += L.W[off + i] * d;
            }
          }
          if (prev) for (let i = 0; i < L.inp; i++) if (a[i] <= 0) prev[i] = 0;
          delta = prev;
        }
      }
      step++;
      const c1 = 1 - b1 ** step, c2 = 1 - b2 ** step, n = chunk.length;
      layers.forEach((L, li) => {
        const g = opt[li];
        for (let k = 0; k < L.W.length; k++) {
          const grad = g.gW[k] / n + alpha * L.W[k];
          g.mW[k] = b1 * g.mW[k] + (1 - b1) * grad;
          g.vW[k] = b2 * g.vW[k] + (1 - b2) * grad * grad;
          L.W[k] -= (lr * g.mW[k] / c1) / (Math.sqrt(g.vW[k] / c2) + eps);
        }
        for (let k = 0; k < L.out; k++) {
          const grad = g.gb[k] / n;
          g.mb[k] = b1 * g.mb[k] + (1 - b1) * grad;
          g.vb[k] = b2 * g.vb[k] + (1 - b2) * grad * grad;
          L.b[k] -= (lr * g.mb[k] / c1) / (Math.sqrt(g.vb[k] / c2) + eps);
        }
      });
    }
    onEpoch(epoch + 1);
    await yieldToUI();
  }
  return layers;
}

let lastYield = 0;
/** 화면이 멈추지 않게 가끔 양보 (setTimeout은 백그라운드 탭에서 1초씩 늦춰지므로 MessageChannel 사용) */
function yieldToUI() {
  if (performance.now() - lastYield < 30) return Promise.resolve();
  return new Promise((resolve) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => { lastYield = performance.now(); resolve(); };
    ch.port2.postMessage(null);
  });
}

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

function predict(features) {
  const probs = forward(state.model.layers, features).at(-1);
  const best = argmax(probs);
  const conf = Number($("confInput").value);
  return { label: probs[best] >= conf ? state.model.labels[best] : "?", prob: probs[best] };
}

function serializeModel(labels, layers) {
  return { labels, layers: layers.map((L) => ({ inp: L.inp, out: L.out, W: Array.from(L.W), b: Array.from(L.b) })) };
}

function restoreModel(m) {
  if (!m?.layers) return null;
  return { labels: m.labels, layers: m.layers.map((L) => ({ ...L, W: Float32Array.from(L.W), b: Float32Array.from(L.b) })) };
}
state.model = restoreModel(state.model);

// ------------------------------------------------------------------ 훈련
async function train() {
  if (state.training) return;
  stopRecording();
  const log = (msg) => { $("trainLog").textContent += msg + "\n"; };
  $("trainLog").textContent = "";
  $("trainResult").textContent = "";
  $("trainResult").style.color = "";

  const c = counts();
  const labels = state.labels.filter((l) => c[l] > 0);
  log(`라벨별 샘플 수: ${labels.map((l) => `${l} ${c[l]}`).join(", ") || "없음"}`);
  const tooFew = labels.filter((l) => c[l] < MIN_SAMPLES_PER_LABEL);
  if (labels.length < 2 || tooFew.length) {
    $("trainResult").textContent = "훈련 실패";
    $("trainResult").style.color = "var(--danger)";
    log(labels.length < 2
      ? "라벨이 2개 이상 있어야 합니다. (예: 원하는 제스처 + none)"
      : `샘플이 ${MIN_SAMPLES_PER_LABEL}개 미만인 라벨이 있습니다: ${tooFew.join(", ")}`);
    return;
  }

  state.training = true;
  $("trainBtn").disabled = true;
  $("trainResult").textContent = "훈련 중...";
  const X = state.rows.map((r) => r.slice(1));
  const y = state.rows.map((r) => labels.indexOf(r[0]));

  // 1) 라벨별로 80%는 훈련, 20%는 성능 확인
  const rand = mulberry32(42);
  const trainIdx = [], testIdx = [];
  labels.forEach((_, k) => {
    const ids = y.map((v, i) => (v === k ? i : -1)).filter((i) => i >= 0);
    for (let i = ids.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [ids[i], ids[j]] = [ids[j], ids[i]];
    }
    const nTest = Math.max(1, Math.round(ids.length * 0.2));
    testIdx.push(...ids.slice(0, nTest));
    trainIdx.push(...ids.slice(nTest));
  });

  const progress = $("trainProgress");
  progress.max = EPOCHS * 2;
  progress.value = 0;
  try {
    const evalLayers = await fitMLP(trainIdx.map((i) => X[i]), trainIdx.map((i) => y[i]), labels.length,
      (e) => { progress.value = e; });

    const K = labels.length;
    const cm = Array.from({ length: K }, () => new Array(K).fill(0));
    for (const i of testIdx) cm[y[i]][argmax(forward(evalLayers, X[i]).at(-1))]++;
    const correct = cm.reduce((s, row, k) => s + row[k], 0);
    const acc = correct / testIdx.length;

    const pad = Math.max(8, ...labels.map((l) => l.length + 1));
    log("\n=== 테스트 결과 ===");
    log(`${"".padEnd(pad)} precision  recall  샘플`);
    labels.forEach((l, k) => {
      const predicted = cm.reduce((s, row) => s + row[k], 0);
      const actual = cm[k].reduce((s, v) => s + v, 0);
      const p = predicted ? cm[k][k] / predicted : 0;
      const r = actual ? cm[k][k] / actual : 0;
      log(`${l.padEnd(pad)} ${p.toFixed(3).padStart(9)}  ${r.toFixed(3).padStart(6)}  ${String(actual).padStart(4)}`);
    });
    log("\n혼동 행렬 (행: 정답, 열: 예측)");
    log(`${"".padEnd(pad)}${labels.map((l) => l.slice(0, 6).padStart(7)).join("")}`);
    cm.forEach((row, k) => log(`${labels[k].padEnd(pad)}${row.map((v) => String(v).padStart(7)).join("")}`));

    // 2) 전체 데이터로 다시 훈련해서 저장
    log("\n전체 데이터로 최종 모델 훈련 중...");
    const finalLayers = await fitMLP(X, y, labels.length, (e) => { progress.value = EPOCHS + e; });
    const serialized = serializeModel(labels, finalLayers);
    state.model = restoreModel(serialized);
    save(STORE.model, serialized);
    log("모델 저장 완료 (이 브라우저에 저장됨)");

    $("trainResult").textContent = `테스트 정확도 ${(acc * 100).toFixed(1)}%`;
    $("trainResult").style.color = "var(--accent)";
  } catch (err) {
    $("trainResult").textContent = "훈련 실패";
    $("trainResult").style.color = "var(--danger)";
    log(String(err));
  } finally {
    state.training = false;
    $("trainBtn").disabled = false;
  }
}

function renderModelStatus() {
  const el = $("modelStatus");
  if (state.model) {
    el.textContent = `모델 로드됨: ${state.model.labels.join(", ")}`;
    el.style.color = "var(--accent)";
  } else {
    el.textContent = "모델이 없습니다. 먼저 [훈련] 탭에서 훈련하세요.";
    el.style.color = "var(--danger)";
  }
}

// ------------------------------------------------------------------ 설정 (이모지 크기·색)
function applySettings() {
  $("emojiSize").value = state.settings.emojiSize;
  $("emojiSizeVal").textContent = `${state.settings.emojiSize}px`;
  $("emojiTint").value = state.settings.tint;
  shownEmojiKey = { builtin: null, infer: null };
  save(STORE.settings, state.settings);
}

function renderBuiltinList() {
  $("builtinList").replaceChildren(...Object.entries(BUILTIN_EMOJIS).map(([name, emoji]) => {
    const li = document.createElement("li");
    li.append(emojiImg(emoji), name);
    return li;
  }));
}

// ------------------------------------------------------------------ 이벤트 연결
$("startBtn").addEventListener("click", start);
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));
$("addLabelBtn").addEventListener("click", addLabel);
$("labelInput").addEventListener("keydown", (e) => { if (e.key === "Enter") addLabel(); });
$("emojiApplyBtn").addEventListener("click", () => setEmoji($("emojiInput").value));
$("emojiClearBtn").addEventListener("click", () => setEmoji(""));
$("emojiInput").addEventListener("keydown", (e) => { if (e.key === "Enter") setEmoji($("emojiInput").value); });
$("palette").replaceChildren(...EMOJI_PALETTE.map((emoji) => {
  const b = document.createElement("button");
  b.textContent = emoji;
  b.title = `${emoji} 지정`;
  b.addEventListener("click", () => setEmoji(emoji));
  return b;
}));
$("deleteLabelBtn").addEventListener("click", deleteLabel);
$("recBtn").addEventListener("click", toggleRecording);
$("exportBtn").addEventListener("click", exportData);
$("importInput").addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file) importData(file);
  e.target.value = "";
});
$("clearAllBtn").addEventListener("click", clearAll);
$("trainBtn").addEventListener("click", train);
$("confInput").addEventListener("input", (e) => { $("confVal").textContent = Number(e.target.value).toFixed(2); });
$("emojiSize").addEventListener("input", (e) => {
  state.settings.emojiSize = Number(e.target.value);
  applySettings();
});
$("emojiTint").addEventListener("change", (e) => {
  state.settings.tint = e.target.value;
  applySettings();
  renderLabelList();
  renderBuiltinList();
});

// Space는 [수집] 탭에서 녹화 시작/정지 (입력창에 글자를 쓸 때는 제외)
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || state.tab !== "collect") return;
  if (e.target.matches("input[type=text], input[type=number], textarea")) return;
  e.preventDefault();
  if (!e.repeat) toggleRecording();
});
// 버튼에 포커스가 있을 때 Space를 떼면 버튼 클릭으로도 처리돼 녹화가 바로 꺼지므로 막음
document.addEventListener("keyup", (e) => {
  if (e.code === "Space" && state.tab === "collect" && !e.target.matches("input[type=text], input[type=number], textarea")) {
    e.preventDefault();
  }
});

applySettings();
renderLabelList();
renderBuiltinList();
renderModelStatus();
