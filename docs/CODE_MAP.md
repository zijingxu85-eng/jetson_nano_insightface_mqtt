# jetson_python3.8 代码框架与阅读指南

> 对象：`/home/xinghuo/jetson_python3.8`（门锁工程，识别对象 `zhaopeng`）
> 本地副本：`C:\Users\19232\Desktop\jetson_backup\_src\jetson_python3.8\`
> 主程序 738 行，单文件；本文档给出**文件职责表 + 行号索引 + 原理**，可对着读。

---

## 0. 先纠正术语：它不是"目标检测"

| 概念 | 输入 → 输出 | 本工程对应 |
|---|---|---|
| **目标检测** object detection | 图 → 框 + 类别（"这里有个**人**"） | 只是第 1 步：`det_10g`(SCRFD) 出**人脸框**+5 点 |
| **人脸检测** face detection | 图 → 人脸框 + 关键点 | 同上（人脸专用检测器） |
| **人脸识别 (1:N 辨认)** identification | 图 → **"是谁"** | 第 3~4 步：`w600k_r50` 出 512 维特征 + 余弦匹配 |
| **人脸验证 (1:1)** verification | 两张图 → **"是不是同一个人"** | 库里只有 1 人时，实际退化成这一种 |
| **光流跟踪** optical flow | 上一帧的框 → 这一帧的框 | `FaceTracker`（**不是神经网络**，纯 OpenCV 角点+LK） |

所以这条流水线 = **人脸检测 → 关键点对齐 → 特征提取(512维) → 余弦匹配 → 决策(开门/上报)**，
"目标检测"只覆盖它的第一环；而且因为库里只有 zhaopeng 一个人，**最终做的事是 1:1 模板验证**，不是通用意义上的"目标检测"。

---

## 1. 技术栈速查（实测）

| 组件 | 版本/文件 | 说明 |
|---|---|---|
| 运行环境 | `venv38/`（Python 3.8.x，目录名即 venv38） | 主程序跑在这里 |
| 推理后端 | `onnxruntime 1.11.0`（CPU） | GPU 版与 CUDA 10.2 不兼容，已弃 |
| 识别框架 | `insightface 1.0.1` + `buffalo_l` 模型包 | `~/.insightface/models/buffalo_l/` |
| OpenCV | 5.0.0（venv38 内） | 系统另有一套给 python3.6 用 |
| TensorRT | 系统自带（python3.6 + pycuda替代ctypes） | 仅旧 GPU 线使用，**当前被禁用** |

`buffalo_l` 实测输入/输出：

| onnx | 输入 | 输出 | 作用 |
|---|---|---|---|
| `det_10g.onnx` | `[1,3,?,?]` | 9 个：3 尺度 ×(score, box4, 5点×2) | SCRFD-10GF 人脸检测 |
| `w600k_r50.onnx` | `[N,3,112,112]` | `[1,512]` | **识别网络**：ResNet-50 + ArcFace，512 维 |
| `2d106det.onnx` | `[N,3,192,192]` | `[1,212]` | 106 点 2D 关键点 |
| `1k3d68.onnx` | `[N,3,192,192]` | `[1,3309]` | 68 点 3D 关键点 |
| `genderage.onnx` | `[N,3,96,96]` | `[1,3]` | 性别年龄（未启用） |

---

## 2. 文件职责表

| 文件 | 行数 | 职责 | 状态 |
|---|---|---|---|
| `face_recognition_doorlock.py` | 738 | **主程序**：检测/对齐/特征/匹配/跟踪/MQTT/UI 全在这 | ★核心 |
| `collect_faces.py` | 97 | 采集入库：`--name` 决定 `known_faces/<名字>/` | ★要用 |
| `run_doorlock.sh` | 18 | 启动包装：设 `DISPLAY=:0`、`LD_PRELOAD=libgomp`、`OPENBLAS_CORETYPE=ARMV8` | 用 |
| `face-doorlock.service` | 26 | systemd 开机自启，崩溃自动重启，日志 `logs/doorlock.log` | 部署用 |
| `install_jetson_env.sh` / `download_offline_packages.sh` | 20K/8K | 离线环境部署（ARM64 编译 opencv/scipy 的坑都在 README） | 部署用 |
| `trt_face.py` | 103 | TensorRT 推理**子进程服务**（JSON 行协议），**python3.6** 跑 | 被禁用 |
| `trt_bridge.py` | 48 | `GPURecognizer` 的早期版本，**已被主程序内的同类覆盖** | 遗留/未用 |
| `compare_emb.py` | ~100 | 对比 CPU 与 GPU 的 embedding → 就是它证明了 FP16 结果不对 | 诊断用 |
| `test_fp32.py` / `test_trt_direct.py` | 小 | FP32 引擎直测 | 诊断用 |
| `send_image_debug.py` | 小 | 单独调试图片上传巴法云 | 调试用 |
| `face_recognition_doorlock.py.bak` | 28K | 旧版备份 | 参考 |
| `1.png` | 84K | 测试图 | — |
| `known_faces/` | — | **人脸库**：`zhaopeng/` 26 张 + `_embeddings_cache.npz` | 数据 |
| `logs/run.log` | 2K | 实跑日志（`[DB] ...`、`[DET] ...`、`[DOOR] ...`） | 证据 |

---

## 3. 总架构（进程 + 线程）

```
                    ┌─────────────────── Jetson Nano (Ubuntu 18.04 / L4T 32.7.6) ───────────────────┐
                    │                                                                              │
   USB 摄像头 ──►   │  进程 A: venv38/bin/python3  face_recognition_doorlock.py                     │
                    │  ┌────────────────────────────────────────────────────────────────────────┐  │
                    │  │ 主线程 Main                                                             │  │
                    │  │  · 读帧 cap.read() → self._latest_frame                                 │  │
                    │  │  · FaceTracker.update() 光流跟框（每帧，无网络）                        │  │
                    │  │  · cv2.imshow 画框/文字/FPS/状态栏                                      │  │
                    │  │  · 键盘 q=退出  r=重载人脸库  c=抓拍对齐脸存库                          │  │
                    │  │  · 命中 → send_q.put((frame, name))                                     │  │
                    │  └───────────────┬───────────────────────────────┬────────────────────────┘  │
                    │                  │                               │                           │
                    │      ┌───────────▼──────────┐        ┌───────────▼──────────────┐            │
                    │      │ DetectorThread(守护) │        │ SenderThread(守护)        │            │
                    │      │ 每 2s: _model.get()  │        │ 取队列 → MQTT publish     │            │
                    │      │  = det_10g+关键点+   │        │  + 抓拍图 HTTP 上传       │            │
                    │      │    w600k_r50(CPU)    │        │  (5s 冷却)               │            │
                    │      │ → _match() → 名/sim  │        └──────────────────────────┘            │
                    │      └──────────────────────┘                                                │
                    │                                                                              │
                    │  进程 B（历史遗留，当前不启动）: python3.6 trt_face.py --service              │
                    │      · TensorRT 引擎 w600k_r50_fp16.engine / _fp32.engine                     │
                    │      · 因 TRT 8.2 FP16 出的 embedding 不对 → 主程序 537 行明确禁用             │
                    └──────────────────────────────────────────────────────────────────────────────┘
                                            │
                        ┌───────────────────┴────────────────────┐
                        ▼                                        ▼
             巴法云 MQTT (bemfa.com:9501)             巴法云 HTTP 图床
             topic "men006" ← 发 "off"                 POST /vb/api/v1/imagesUpload
             topic "facepicture" ──────────────────►  返回 url（手机/小程序看图）
```

---

## 4. 主程序行号索引（`face_recognition_doorlock.py`）

| 行号 | 内容 | 备注 |
|---|---|---|
| 1–9 | 文件 docstring：声明"CPU 检测 + GPU 识别(已被禁用)" | 先读它 |
| 23–36 | **Config**：`BEMFA_KEY` / `DOOR_TOPIC="men006"` / `PICTURE_TOPIC="facepicture"` / `KNOWN_FACES_DIR` / 640×480 / `THRESHOLD=0.40` / `SEND_COOLDOWN=5.0` / `DETECT_INTERVAL=2.0` / `MAX_FACE_AGE=3.0` / `EMBEDDINGS_CACHE` | 全局参数都在这 |
| 39–131 | `class FaceTracker` 光流跟踪 | `goodFeaturesToTrack` 63–65；`calcOpticalFlowPyrLK` 90–93；取中位位移平移框 100–107；角点<4 或超过 3s → 丢框 |
| 134–143 | `insightface FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'], allowed_modules=[detection, recognition, landmark_3d_68, landmark_2d_106])` + `det_size=(160,160)` | **模型只加载一次**，进程级 |
| 145–152 | `REF_LANDMARKS`：ArcFace 标准 5 点（112×112 坐标系） | 对齐目标 |
| 155–190 | `align_face()`：106/68 关键点取 5 点 → `estimateAffinePartial2D(LMEDS)` → `warpAffine(112×112)`；失败退回 bbox 裁剪 | **尺度归一化的关键** |
| 193–271 | `class BemfaMQTT`：**手写 MQTT 协议**（不发 paho） | connect 203–225；publish 227–242（主题拼 `topic/set`）；PINGREQ 244–249 |
| 274–346 | `class GPURecognizer`：拉起 `python3.6 trt_face.py --service` 子进程，JSON+base64 通信 | 325–343；当前 `ready=False` 基本不生效 |
| 349–425 | `class DetectorThread._loop()`：**每 2s 的识别节拍** | 373 `_model.get(frame)`；380–394 过滤（det_score≥0.7、边长≥50、长宽比 0.7~1.8）；410 `_match(face.normed_embedding)`；411–414 写 `_last_face`；415 用它初始化光流 |
| 428–443 | `class FaceDoorLock.__init__`：mqtt / gpu_rec / `names` `embs` / `send_q=Queue(5)` / tracker / detector | 状态中心 |
| 445–511 | `load_known_faces()`：**建库** | 456–473 缓存新鲜就直接 load npz；477–502 否则逐图 `_model.get` → `normed_embedding` → **取均值再归一化**（类原型）；506–508 写回 npz；日志 `[DB] zhaopeng: 26 photos` |
| 513–528 | `_match(emb)`：**判定** | 点积=余弦；526 行 `bs > 1.0-THRESHOLD(=0.60)` **且** `bs-bs2 > 0.03`（margin） |
| 530–663 | `run()`：主流程 | 537–539 打印"GPU TensorRT disabled"；542 建库；545 连 MQTT；548–554 开摄像头；558–560 起线程；565–652 主循环；621–652 q/r/c 三键 |
| 665–678 | `_sender()`：**执行** | 673 `publish("men006", "off")`；674 `print("[DOOR] %s")`；675 发图 |
| 680–705 | `_send_pic()`：multipart 上传 `apis.bemfa.com/vb/api/v1/imagesUpload` | 702 成功打 `[PIC] OK: url` |
| 707–731 | `_draw()`：状态栏 FPS / `OK:name` / `GPU:OFF` / `MQTT:ON` | 看界面就知道哪条线生效 |

---

## 5. 原理拆解（5 步，每步都有公式）

### ① 检测（det_10g / SCRFD，每 2s 一次）
anchor-free：输入缩到 `det_size=(160,160)`，在 stride 8/16/32 三层特征图上各预测 `score + 4个距离回归 + 5点偏移`，
实测输出形状 `12800/3200/800` 个候选 × 3 类；NMS 后留下框 + 5 个关键点。
代码里再过滤：`det_score ≥ 0.7`、人脸边长 ≥ 50px、长宽比 `0.7 ~ 1.8`（380–394 行）→ 取分最高的 1 张脸。

### ② 跟踪（光流，省算力的关键）
检测 2s 才跑一次，中间 ~60 帧靠跟踪维持框：
```
框内取角点 goodFeaturesToTrack(≤15 个)
  → LK 金字塔光流求每点位移 Δ
  → 取中位数 (dx,dy) 平移整个框
```
超过 3s 没被检测刷新、或剩余角点 < 4 → 判定人走了，清框（重复的人脸不会一直"挂着"）。

### ③ 对齐（把脸摆正，尺度统一）
取 5 点：`左眼、右眼、鼻尖、左嘴角、右嘴角`（2d106 用索引 33/87/52/61/71；3d68 用 36/45/30/48/54），
解相似变换并 warp 到 112×112：
$$\min_{s,R,t}\ \sum_{i=1}^{5}\big\|sR\,p_i + t - q_i\big\|^2 \quad(\text{estimateAffinePartial2D})$$
`q_i` = 标准 ArcFace 5 点模板。
**这一步是"能稳定认出人"的根因**——它消除了尺度/旋转/位置差异，而隔壁 face_env 的 LBPH 恰恰缺这一步。

### ④ 特征（w600k_r50）
112×112 → ResNet-50 → 512 维向量，L2 归一化：$\|v\|_2 = 1$。
（网络在 WebFace600K 上用 ArcFace 损失训练，同人向量夹角小、异人夹角大。）

### ⑤ 匹配（余弦 + 双门槛）
建库时**每个人合成一个类原型**：
$$p_k=\frac{1}{n}\sum_{i=1}^{n}\frac{v_i}{\|v_i\|}, \qquad p_k \leftarrow \frac{p_k}{\|p_k\|}$$
识别时：
$$k^\*=\arg\max_k\ (q\cdot p_k), \qquad \text{通过条件：}\ q\cdot p_{k^\*} > 0.60 \ \text{且}\ (q\cdot p_{k^\*}-q\cdot p_{k_2}) > 0.03$$
（向量都已归一化，所以点积就是余弦；第二个条件是与"亚军"的间距 margin，防止两张脸都像时乱认。）

实测印证：`[DET] zhaopeng: sim=0.7529` 通过；`[DET] Unknown: sim=0.5354` 被 0.60 挡掉。

### ⑥ 决策 + 执行
命中 → `send_q` → Sender 线程 → `publish("men006","off")` 触发门锁 + 抓拍图上传 → 手机看图。5 秒冷却防连发。

---

## 6. 关键参数表

| 参数 | 值 | 位置 | 含义 |
|---|---|---|---|
| `DETECT_INTERVAL` | 2.0 s | 32 | 检测节拍（不是每帧都跑网络） |
| `MAX_FACE_AGE` | 3.0 s | 33 | 检测结果的有效期 |
| `THRESHOLD` | 0.40 | 30 | **反着命名**：实际门槛是 `1-0.40 = 0.60` 余弦 |
| margin | 0.03 | 526 | 与第二名的间距 |
| `MIN_DET_SCORE` | 0.7 | 380 | 检测置信过滤 |
| `MIN_FACE_SIZE` | 50 px | 381 | 人脸尺寸下限 |
| `SEND_COOLDOWN` | 5.0 s | 31 | 开门最小间隔 |
| `det_size` | 160×160 | 142 | SCRFD 输入尺寸（小=快） |
| `REC_SIZE` | 112×112 | trt_face.py:18 | 识别网络输入 |

---

## 7. 建议阅读路线（照着读 30 分钟能通）

1. **1–36 行**：docstring + Config → 知道有哪些旋钮
2. **530–663 行 `run()`**：主流程骨架（它把其他类串起来，先看骨架不容易迷路）
3. **445–528 行**：`load_known_faces()` + `_match()` → 库怎么建、怎么判
4. **349–425 行**：`DetectorThread._loop()` → "每一帧怎么得出结论、日志怎么打"
5. 挑细节看：`align_face()` 155–190（对齐）、`FaceTracker` 39–131（光流）、`BemfaMQTT` 193–271（手写协议）
6. 想搞 GPU 再看：`trt_face.py`（ctypes 调 libcuda + TRT 引擎）+ `compare_emb.py`（为什么弃用）

---

## 8. ⚠️ 读源码时注意的坑

1. **README 与代码不一致**：README 里的工作流程图写的是 `register_face.py`、主题 `doorlock001/faceimage001`、阈值 55%，而**真实代码**是 `collect_faces.py`、主题 `men006/facepicture`、余弦 0.60。**一切以代码为准**。
2. **GPU 线是"陪跑"**：537–539 行主动禁用；`~/.insightface/engines/*.engine`（fp16 168MB、fp32 336MB）都已编好但不用；界面会显示 `GPU:OFF`。
3. **`THRESHOLD` 命名反了**：定义 0.40，注释写 "cosine > 0.60"，代码里用 `1.0 - THRESHOLD`。
4. **缓存不会自动失效**：`_embeddings_cache.npz` 只要比所有图片新就直接用；改阈值/改模型/换名字目录后**要手动删缓存**才重算。
5. **两种采样不一致**：`collect_faces.py` 存的是**整帧**（87 行），而主程序按 `c` 键存的是**对齐后的 112×112**（641–650 行）→ 后者质量更好，推荐用 `c` 键补样本。
6. **双解释器并存**：主程序 python3.8（venv38）；GPU 子进程由代码显式调用 `python3.6`（`trt_face.py` 的 shebang 与 `subprocess.Popen(["python3.6", ...])`），用系统自带 TRT/cv2。
7. **`trt_bridge.py` 是遗留**：功能与主程序里的 `GPURecognizer` 重复，别被它误导。
8. **手写 MQTT**：`BemfaMQTT` 自己拼协议包，payload 用 `"off"` 触发（不是 `"open"`），主题要拼成 `topic/set`。

---

## 9. 怎么跑 / 验证

```bash
# 启动（Jetson 上）
cd /home/xinghuo/jetson_python3.8 && source venv38/bin/activate && ./run_doorlock.sh
# 界面键位: q=退出  r=重载人脸库  c=抓拍对齐脸入库
# 采集新样本
python collect_faces.py --name zhaopeng --num 10     # 空格保存
# 开机自启
sudo cp face-doorlock.service /etc/systemd/system/ && sudo systemctl enable --now face-doorlock
```
实测成功日志形态：`[DB] zhaopeng: 26 photos` → `[MQTT] Connected OK` → `[CAM] OK (640x480)` → `[DET] zhaopeng: sim=0.7529` → `[DOOR] zhaopeng` → `[PIC] OK: https://...`
