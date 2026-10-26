#!/usr/bin/env python3
"""Jetson Nano Face DoorLock V3 -- CPU Detection + GPU Recognition (Hybrid)
     with real-time face tracking (optical flow)

Architecture:
  Main Thread: Camera capture + display + MQTT + optical-flow tracking
  Detector Thread (CPU, every 2s): insightface buffalo_l -> bbox + landmarks
  Recognizer Subprocess (GPU, TensorRT FP16): Python3.6 trt_face.py -> embedding
"""
import cv2
import numpy as np
import sys
import time
import json
import threading
import queue
import socket
import base64
import subprocess
import traceback
from pathlib import Path

from liveness_judge import LivenessJudge

# ====== Config（参数集中在 config.py，私钥放在 local_secrets.py 或环境变量）======
import config as C

BEMFA_KEY = C.BEMFA_KEY
DOOR_TOPIC = C.DOOR_TOPIC
PICTURE_TOPIC = C.PICTURE_TOPIC
KNOWN_FACES_DIR = C.KNOWN_FACES_DIR
EMBEDDINGS_CACHE = C.EMBEDDINGS_CACHE
CAMERA_ID = C.CAMERA_ID
CAM_W, CAM_H = C.CAM_W, C.CAM_H
THRESHOLD = C.THRESHOLD
SEND_COOLDOWN = C.SEND_COOLDOWN
DETECT_INTERVAL = C.DETECT_INTERVAL
MAX_FACE_AGE = C.MAX_FACE_AGE
TRACK_MAX_FEAT = C.TRACK_MAX_FEAT
TRACK_QUALITY = C.TRACK_QUALITY

if not BEMFA_KEY:
    print("[WARN] BEMFA_KEY 未配置：请设置环境变量，或复制 local_secrets.example.py 为 local_secrets.py 填入私钥", flush=True)

# ====== Real-time Face Tracker (optical flow) ======
class FaceTracker:
    def __init__(self):
        self.bbox = None
        self._prev_gray = None
        self._points = None
        self._age = 0
        self._max_age = int(MAX_FACE_AGE / 0.033)
        self._lock = threading.Lock()

    def init(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            with self._lock:
                self.bbox = None
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            with self._lock:
                self.bbox = None
            return False
        features = cv2.goodFeaturesToTrack(
            roi, maxCorners=TRACK_MAX_FEAT,
            qualityLevel=TRACK_QUALITY, minDistance=8)
        with self._lock:
            if features is not None:
                self._points = features.reshape(-1, 2).astype(np.float32)
                self._points[:, 0] += x1
                self._points[:, 1] += y1
            else:
                self._points = None
            self._prev_gray = gray
            self.bbox = [x1, y1, x2, y2]
            self._age = 0
        return True

    def update(self, frame):
        if self.bbox is None:
            return None
        self._age += 1
        if self._age > self._max_age:
            self.bbox = None
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with self._lock:
            points = self._points
            prev_gray = self._prev_gray
        if points is not None and len(points) >= 4:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray,
                points.reshape(-1, 1, 2).astype(np.float32),
                None, winSize=(21, 21), maxLevel=3)
            if new_pts is not None and status is not None:
                mask = status.ravel() == 1
                good_new = new_pts[mask].reshape(-1, 2)
                if len(good_new) >= 4:
                    good_old = points[mask]
                    disp = good_new - good_old
                    dx = float(np.median(disp[:, 0]))
                    dy = float(np.median(disp[:, 1]))
                    # If features barely moved, keep position (stationary is OK)
                    with self._lock:
                        self.bbox[0] += int(dx)
                        self.bbox[1] += int(dy)
                        self.bbox[2] += int(dx)
                        self.bbox[3] += int(dy)
                        self._points = good_new.copy()
                        self._prev_gray = gray
                    return self.bbox
                # Too few tracked features -> person likely left
                with self._lock:
                    self.bbox = None
                    self._points = None
                return None
            # Optical flow failed entirely
            with self._lock:
                self.bbox = None
                self._points = None
            return None
        # No features to track
        with self._lock:
            self.bbox = None
        return None

    def clear(self):
        with self._lock:
            self.bbox = None
            self._points = None
            self._prev_gray = None
        self._age = 0


# ====== Insightface CPU (detection + landmarks) ======
print("[INIT] Loading insightface CPU detection...", flush=True)
from insightface.app import FaceAnalysis
_model = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition', 'landmark_3d_68', 'landmark_2d_106']
)
_model.prepare(ctx_id=0, det_size=(160, 160))
print("[INIT] Insightface ready (detection + recognition + landmarks)", flush=True)

# ArcFace 5-point reference landmarks (112x112)
REF_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(img, face):
    """5-point landmark alignment -> 112x112 BGR image. Returns None on failure."""
    lm = None
    if (hasattr(face, 'landmark_2d_106') and
            face.landmark_2d_106 is not None and
            len(face.landmark_2d_106) >= 72):
        lm = np.array([
            face.landmark_2d_106[33],
            face.landmark_2d_106[87],
            face.landmark_2d_106[52],
            face.landmark_2d_106[61],
            face.landmark_2d_106[71],
        ], dtype=np.float32)
    elif (hasattr(face, 'landmark_3d_68') and
          face.landmark_3d_68 is not None and
          len(face.landmark_3d_68) >= 68):
        lm = np.array([
            face.landmark_3d_68[36],
            face.landmark_3d_68[45],
            face.landmark_3d_68[30],
            face.landmark_3d_68[48],
            face.landmark_3d_68[54],
        ], dtype=np.float32)[:, :2]
    if lm is not None:
        tform = cv2.estimateAffinePartial2D(lm, REF_LANDMARKS, method=cv2.LMEDS)
        if tform[0] is not None:
            return cv2.warpAffine(img, tform[0], (112, 112))
    # Fallback: bbox crop
    b = face.bbox.astype(int)
    x1, y1 = max(0, b[0]), max(0, b[1])
    x2, y2 = min(img.shape[1], b[2]), min(img.shape[0], b[3])
    if x2 > x1 and y2 > y1:
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            return cv2.resize(crop, (112, 112))
    return None


# ====== MQTT (Bemfa) -- FIXED protocol ======
class BemfaMQTT:
    def __init__(self, key, host="bemfa.com", port=9501):
        self.key = key
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self._lock = threading.Lock()

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
            cid = self.key.encode()
            ka = (60).to_bytes(2, 'big')
            # Payload = protocol header + keepalive + client_id_len + client_id
            pl = b'\x00\x04MQTT\x04\x02' + ka + len(cid).to_bytes(2, 'big') + cid
            # Remaining length = total length of variable header + payload
            rem = self._rem(len(pl))
            # Send: fixed header (CONNECT + remaining length) + payload
            self.sock.sendall(b'\x10' + rem + pl)
            resp = self.sock.recv(4)
            if len(resp) >= 4 and resp[3] == 0:
                self.connected = True
                print("[MQTT] Connected OK", flush=True)
                return True
            print("[MQTT] Connack failed: %s" % resp.hex(), flush=True)
            return False
        except Exception as e:
            print("[MQTT] Connect error: %s" % e, flush=True)
            return False

    def publish(self, topic, msg):
        if not self.connected and not self.connect():
            return False
        tb = (topic + "/set").encode()
        if isinstance(msg, str):
            msg = msg.encode()
        pl = len(tb).to_bytes(2, 'big') + tb + msg
        rem = self._rem(len(pl))
        try:
            with self._lock:
                self.sock.sendall(b'\x30' + rem + pl)
            return True
        except Exception as e:
            print("[MQTT] Publish error: %s" % e, flush=True)
            self.connected = False
            return False

    def heartbeat(self):
        if self.connected:
            try:
                self.sock.sendall(b'\xc0\x00')
            except:
                self.connected = False

    def disconnect(self):
        if self.sock:
            try:
                self.sock.sendall(b'\xe0\x00')
                self.sock.close()
            except:
                pass
        self.connected = False

    @staticmethod
    def _rem(length):
        buf = b''
        while True:
            d = length % 128
            length //= 128
            if length > 0:
                d |= 0x80
            buf += bytes([d])
            if length == 0:
                break
        return buf


# ====== TensorRT GPU Recognizer ======
class GPURecognizer:
    def __init__(self, script_path=None):
        self.proc = None
        self.lock = threading.Lock()
        if script_path is None:
            script_path = str(Path(__file__).parent / "trt_face.py")
        self.script = script_path
        self.ready = False

    def start(self):
        if self.proc:
            return True
        print("[GPU] Starting TensorRT subprocess (FP16)...", flush=True)
        self.proc = subprocess.Popen(
            ["python3.6", self.script, "--service"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1)
        t0 = time.time()
        while time.time() - t0 < 20:
            try:
                with self.lock:
                    self.proc.stdin.write('{"test": true}\n')
                    self.proc.stdin.flush()
                    resp = self.proc.stdout.readline()
                if resp:
                    r = json.loads(resp.strip())
                    if r.get("status") == "ok":
                        self.ready = True
                        print("[GPU] Ready (engine=%s)" % r.get("engine", "?"), flush=True)
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        print("[GPU] Start FAILED", flush=True)
        self.stop()
        return False

    def stop(self):
        self.ready = False
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except:
                try:
                    self.proc.kill()
                except:
                    pass
            self.proc = None

    def get_embedding(self, face_img):
        if not self.ready or self.proc is None:
            return None
        try:
            _, jpeg = cv2.imencode(".jpg", face_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            b64 = base64.b64encode(jpeg).decode()
            with self.lock:
                self.proc.stdin.write(json.dumps({"image_base64": b64}) + "\n")
                self.proc.stdin.flush()
                resp = self.proc.stdout.readline()
            if not resp:
                return None
            r = json.loads(resp.strip())
            emb = r.get("embedding", [])
            return np.array(emb, dtype=np.float32) if emb else None
        except Exception as e:
            print("[GPU] Error: %s" % e, flush=True)
            self.ready = False
            return None

    def __del__(self):
        self.stop()


# ====== Detection Thread ======
class DetectorThread:
    def __init__(self, parent):
        self.parent = parent
        self._thread = None
        self.running = False

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            time.sleep(DETECT_INTERVAL)
            if not self.running:
                break
            try:
                frame = getattr(self.parent, '_latest_frame', None)
                if frame is None:
                    continue
                faces = _model.get(frame)
                if not faces:
                    self.parent._tracker.clear()
                    self.parent._last_face = None
                    continue

                # Filter by detection confidence (ignore background noise)
                MIN_DET_SCORE = 0.7
                MIN_FACE_SIZE = 50
                best = None
                for f in faces:
                    if f.det_score < MIN_DET_SCORE:
                        continue
                    bw = f.bbox[2] - f.bbox[0]
                    bh = f.bbox[3] - f.bbox[1]
                    if bw < MIN_FACE_SIZE or bh < MIN_FACE_SIZE:
                        continue
                    # Face aspect ratio should be roughly square
                    if bw > 0 and bh > 0:
                        ratio = bh / bw
                        if ratio < 0.7 or ratio > 1.8:
                            continue
                    if best is None or f.det_score > best.det_score:
                        best = f
                if best is None:
                    self.parent._tracker.clear()
                    self.parent._last_face = None
                    # Clear stale identity after 5s of no face
                    if time.time() - self.parent._recognized_time > 5.0:
                        self.parent._recognized_name = None
                    continue

                face = best
                bbox = face.bbox.astype(int)
                now = time.time()
                det_score = float(face.det_score)

                name, sim = self.parent._match(face.normed_embedding)
                self.parent._last_face = {
                    "bbox": bbox.tolist(), "name": name,
                    "sim": sim, "det_score": det_score, "time": now,
                }
                self.parent._tracker.init(frame, bbox)

                if name:
                    # self.parent._recognized_name = name  # main loop handles this
                    self.parent._recognized_time = now  # for stale cleanup
                    print("[DET] %s: sim=%.4f" % (name, sim), flush=True)
                else:
                    print("[DET] Unknown: sim=%.4f" % sim, flush=True)
            except Exception as e:
                print("[DET] Error: %s" % e, flush=True)
                traceback.print_exc()


# ====== Main ======
class FaceDoorLock:
    def __init__(self):
        self.mqtt = BemfaMQTT(BEMFA_KEY)
        self.gpu_rec = GPURecognizer()
        self.names = []
        self.embs = []
        self.last_send = 0
        self.send_q = queue.Queue(5)
        self.running = False
        self._latest_frame = None
        self._last_face = None
        self._recognized_name = None
        self._recognized_time = 0
        self._tracker = FaceTracker()
        self._detector = DetectorThread(self)
        self._live = LivenessJudge(gap=C.RHO_GAP, window=C.RHO_WINDOW,
                                   rho_min=C.RHO_MIN, corners=C.RHO_CORNERS)
        self._last_spoof = 0.0
        self._last_wait = 0.0
        # 开门提示（由发送线程写入，主线程绘制）
        self._door_flash = 0.0
        self._door_msg = ""
        self._door_color = (0, 255, 0)
        self._door_count = 0
        self._door_fail = 0

    def load_known_faces(self):
        """Load known face DB using CPU embeddings (ONNX) — with cache support"""
        d = Path(KNOWN_FACES_DIR)
        self.names.clear()
        self.embs.clear()
        if not d.exists():
            d.mkdir(parents=True)
            return

        cache_path = Path(EMBEDDINGS_CACHE)
        # Try loading from cache
        if cache_path.exists():
            # Check if cache is newer than ALL face images
            cache_mtime = cache_path.stat().st_mtime
            latest_img_mtime = 0
            for pd_dir in d.iterdir():
                if not pd_dir.is_dir():
                    continue
                for f in pd_dir.glob("*"):
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                        latest_img_mtime = max(latest_img_mtime, f.stat().st_mtime)
            if latest_img_mtime < cache_mtime:
                # Cache is fresh — load directly (skip all ONNX inference)
                data = np.load(str(cache_path), allow_pickle=True)
                self.names = list(data['names'])
                for emb in data['embeddings']:
                    self.embs.append(emb.astype(np.float32))
                print("[DB] ✓ 从缓存加载 %d 人: %s" % (len(self.names), self.names), flush=True)
                return

        # No valid cache — compute embeddings from scratch
        print("[DB] 正在计算人脸 embedding（首次/人脸库已更新）...", flush=True)
        for pd_dir in sorted(d.iterdir()):
            if not pd_dir.is_dir():
                continue
            name = pd_dir.name
            embs = []
            for f in pd_dir.glob("*"):
                if f.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                img = cv2.imread(str(f))
                if img is None:
                    continue
                # CPU detection + CPU embedding (ONNX, correct)
                rs = _model.get(img)
                if not rs:
                    continue
                cpu_emb = rs[0].normed_embedding
                if cpu_emb is not None:
                    embs.append(cpu_emb)
            if embs:
                m = np.mean(embs, axis=0)
                n = np.linalg.norm(m)
                if n > 0:
                    m = m / n
                self.embs.append(m)
                self.names.append(name)
                print("  [DB] %s: %d photos" % (name, len(embs)), flush=True)

        # Save cache for next startup
        if self.names:
            np.savez_compressed(str(cache_path),
                                names=np.array(self.names, dtype=object),
                                embeddings=np.array(self.embs, dtype=np.float32))
            print("[DB] ✓ 缓存已保存 (%d 人)" % len(self.names), flush=True)

        print("[DB] %d people: %s" % (len(self.names), self.names), flush=True)

    def _match(self, emb):
        if emb is None or emb.size == 0 or not self.embs:
            return None, 0.0
        bn, bs = None, -1.0
        bs2 = -1.0  # second best
        for n, k in zip(self.names, self.embs):
            s = float(np.dot(emb, k))
            if s > bs:
                bs2 = bs
                bs, bn = s, n
            elif s > bs2:
                bs2 = s
        # Must pass threshold AND have clear margin over second best
        if bs > (1.0 - THRESHOLD) and (bs - bs2) > C.MARGIN:
            return bn, bs
        return None, bs

    def run(self):
        print("=" * 55, flush=True)
        print("  Face DoorLock V3 - CPU Det + CPU Rec (ONNX)", flush=True)
        print("  Threshold: cos>%.2f | Interval: %.1fs" % (1.0 - THRESHOLD, DETECT_INTERVAL), flush=True)
        print("  Optical-flow tracking enabled", flush=True)
        print("=" * 55, flush=True)

        # 1. GPU recognition is disabled (TensorRT FP16 gives wrong embeddings on TRT 8.2)
        #    Using CPU ONNX embeddings instead (insightface .normed_embedding)
        print("[INFO] Using CPU recognition (ONNX) - GPU TensorRT disabled", flush=True)

        # 2. Load known faces (uses CPU ONNX embedding)
        self.load_known_faces()

        # 3. Connect MQTT
        self.mqtt.connect()

        # 4. Open camera
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        if not cap.isOpened():
            print("[CAM] FAIL", flush=True)
            return
        print("[CAM] OK (%dx%d)" % (CAM_W, CAM_H), flush=True)

        # 5. Start threads
        self.running = True
        self._detector.start()
        sender = threading.Thread(target=self._sender, daemon=True)
        sender.start()
        print("Running... q=quit r=reload\n", flush=True)

        ft, fc, fv = time.time(), 0, 0
        try:
            while self.running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                fc += 1
                if time.time() - ft >= 1.0:
                    fv, fc = fc, 0
                    ft = time.time()
                self._latest_frame = frame

                rec = None
                # Optical flow tracking
                tracked_bbox = self._tracker.update(frame)
                has_track = tracked_bbox is not None

                # 活体判据：每帧更新（需要 ~1s 基线 + 6 帧窗口）
                live_state = self._live.update(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), tracked_bbox)

                # Latest detection result
                fi = self._last_face
                face_is_recent = fi is not None and (time.time() - fi["time"]) < MAX_FACE_AGE

                if has_track and face_is_recent:
                    b = tracked_bbox
                    nm, sm = fi["name"], fi["sim"]
                    if nm:
                        rec = nm
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                        cv2.putText(frame, "%s %.1f%%" % (nm, sm * 100),
                                    (b[0], b[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                        if self._tracker._points is not None:
                            for pt in self._tracker._points:
                                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
                    else:
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown (%.2f)" % sm,
                                    (b[0], b[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                        if self._tracker._points is not None:
                            for pt in self._tracker._points:
                                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
                elif has_track and not face_is_recent:
                    b = tracked_bbox
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (100, 100, 100), 1)
                    cv2.putText(frame, "Tracking",
                                (b[0], b[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

                # Door trigger：必须身份通过 + 活体通过
                if rec:
                    now = time.time()
                    if live_state == "SPOOF":
                        if now - self._last_spoof > 5.0:
                            print("[LIVE] SPOOF  rho_bar=%.4f < %.3f  拒绝开门"
                                  % (self._live.rho_bar or 0.0, C.RHO_MIN), flush=True)
                            self._last_spoof = now
                    elif live_state != "OK":
                        if now - self._last_wait > 3.0:
                            print("[LIVE] %s (%s)  等待活体判决" % (live_state, self._live.info), flush=True)
                            self._last_wait = now
                    if live_state == "OK" and now - self.last_send > SEND_COOLDOWN and not self.send_q.full():
                        self.send_q.put((frame.copy(), rec))
                        self.last_send = now
                        self._recognized_name = rec
                        self._recognized_time = now

                self._draw(frame, fv, has_track)
                cv2.imshow("Face DoorLock V3", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    print("[CMD] Quit", flush=True)
                    break
                elif k == ord('r'):
                    print("[CMD] Reload DB", flush=True)
                    self.load_known_faces()
                elif k == ord('c'):
                    _cframe = self._latest_frame
                    if _cframe is None:
                        print("[CMD] No frame for capture", flush=True)
                        continue
                    _cfaces = _model.get(_cframe)
                    if not _cfaces:
                        print("[CMD] No face for capture", flush=True)
                        continue
                    _cbest = max(_cfaces, key=lambda f: f.det_score)
                    _cname = self._last_face.get('name') if self._last_face else None
                    if not _cname:
                        print("[CMD] Unknown face, skip (stand in good light)", flush=True)
                        continue
                    _caligned = align_face(_cframe, _cbest)
                    if _caligned is None:
                        print("[CMD] Align failed", flush=True)
                        continue
                    _cdir = Path(KNOWN_FACES_DIR) / _cname
                    _cdir.mkdir(parents=True, exist_ok=True)
                    _cts = int(time.time())
                    _cv = len(list(_cdir.glob('*capture*.jpg')))
                    _cpath = _cdir / '{0}_capture_v{1}_{2}.jpg'.format(_cname, _cv + 1, _cts)
                    cv2.imwrite(str(_cpath), _caligned)
                    print("[CMD] Saved -> {0} (v{1})".format(_cpath.name, _cv + 1), flush=True)
                    self.load_known_faces()
        except Exception as e:
            print("[MAIN] Unexpected error: %s" % e, flush=True)
            traceback.print_exc()
        finally:
            print("[BYE] Shutting down...", flush=True)
            self.running = False
            self._detector.stop()
            self.gpu_rec.stop()
            cap.release()
            cv2.destroyAllWindows()
            self.mqtt.disconnect()

    def _sender(self):
        while self.running:
            try:
                frame, name = self.send_q.get(timeout=1)
            except queue.Empty:
                self.mqtt.heartbeat()
                continue
            try:
                ok = self.mqtt.publish(DOOR_TOPIC, "off")
                if not ok:
                    raise RuntimeError("publish failed")
                print("[DOOR] %s" % name, flush=True)
                self._door_count += 1
                self._door_msg = "DOOR OPEN  #%d" % self._door_count
                self._door_color = (0, 255, 0)
                self._door_flash = time.time() + 2.5
                self._send_pic(frame)
            except Exception as e:
                print("[SEND] Error: %s" % e, flush=True)
                self._door_fail += 1
                self._door_msg = "DOOR FAIL"
                self._door_color = (0, 0, 255)
                self._door_flash = time.time() + 2.5
            self.send_q.task_done()

    def _send_pic(self, frame):
        try:
            import urllib.request
            _, j = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            boundary = '----BemfaUploadBoundary'
            body = (
                ('--%s\r\nContent-Disposition: form-data; name="openID"\r\n\r\n%s\r\n' %
                 (boundary, BEMFA_KEY)).encode() +
                ('--%s\r\nContent-Disposition: form-data; name="topic"\r\n\r\n%s\r\n' %
                 (boundary, PICTURE_TOPIC)).encode() +
                ('--%s\r\nContent-Disposition: form-data; name="image"; filename="doorlock.jpg"\r\n'
                 'Content-Type: image/jpeg\r\n\r\n' % boundary).encode() +
                j.tobytes() + b'\r\n' +
                ('--%s--\r\n' % boundary).encode()
            )
            req = urllib.request.Request(
                'https://apis.bemfa.com/vb/api/v1/imagesUpload',
                data=body,
                headers={'Content-Type': 'multipart/form-data; boundary=%s' % boundary},
                method='POST')
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                if data.get('code') == 0:
                    print("[PIC] OK: %s" % data['data']['url'], flush=True)
        except Exception as e:
            print("[PIC] FAIL: %s" % e, flush=True)

    def _draw(self, frame, fps, tracking=False):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 32), (0, 0, 0), -1)
        cv2.putText(frame, "FPS:%d" % fps, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if self._recognized_name:
            s = "OK: %s" % self._recognized_name
            c = (0, 255, 0)
        elif tracking:
            s = "Tracking..."
            c = (180, 180, 0)
        else:
            s = "Monitoring"
            c = (180, 180, 180)
        cv2.putText(frame, s, (w - 260, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

        # 活体状态（第二行，位于顶栏下方）
        lv = self._live.label()
        lc = {"LIVE": (0, 255, 0), "SPOOF": (0, 0, 255),
              "INVALID": (0, 165, 255), "WARMUP": (180, 180, 180)}.get(lv, (180, 180, 180))
        rb = self._live.rho_bar
        cv2.putText(frame, "PAD:%s %s" % (lv, "" if rb is None else "rho=%.3f" % rb),
                    (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 2)

        # 开门提示：中央大字，显示 2.5 秒
        if time.time() < self._door_flash:
            txt = self._door_msg
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
            cv2.putText(frame, txt, ((w - tw) // 2, h // 2 + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, self._door_color, 3)

        gs = "GPU:ON" if self.gpu_rec.ready else "GPU:OFF"
        gc = (0, 255, 0) if self.gpu_rec.ready else (0, 0, 255)
        cv2.putText(frame, gs, (w - 120, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gc, 1)

        mqc = "MQTT:ON" if self.mqtt.connected else "MQTT:OFF"
        mcc = (0, 255, 0) if self.mqtt.connected else (0, 0, 255)
        cv2.putText(frame, mqc, (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, mcc, 1)
        cv2.putText(frame, "DOOR:%d FAIL:%d" % (self._door_count, self._door_fail),
                    (w // 2 - 40, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "q=quit r=reload", (w - 140, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)


if __name__ == "__main__":
    try:
        FaceDoorLock().run()
    except Exception as e:
        print("[FATAL] %s" % e, flush=True)
        traceback.print_exc()