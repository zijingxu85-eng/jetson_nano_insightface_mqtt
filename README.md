# Jetson Nano 人脸门禁（insightface + MQTT）

基于 NVIDIA Jetson Nano 的人脸识别门禁系统。insightface（buffalo_l）完成人脸检测与 512 维特征提取，
余弦相似度加二次门限判决身份；两次重识别之间用 Lucas-Kanade 光流维持人脸框；判决结果经 MQTT 上报云端，
由 ESP32-P4 驱动继电器执行开锁，MSPM0G3507 采集环境状态，小智语音终端负责交互。

> 说明：本仓库只包含源码与配置，不含模型文件、虚拟环境与人脸样本。

## 功能特性

- **低频重识别 + 高频跟踪**：每 2 s 做一次检测+识别，帧间用 LK 光流把框推着走，保证 16.7 fps 的连续画面
- **余弦 + 二次门限**：`cos > 0.60` 且高出次优 `> 0.03` 才判定为库内人员
- **三级节流**：5 s 冷却 + 有界队列（≤5）+ 异步发送线程，避免重复开锁与网络阻塞
- **零依赖 MQTT**：用标准库 socket 手写 MQTT 3.1.1 最小子集（CONNECT / PUBLISH / PINGREQ / DISCONNECT）
- **抓拍上传**：开门事件同时把画面以 multipart/form-data 上传云端，失败不影响开锁
- **可选的活体通道**（`liveness/`）：单应残差比 ρ 与余弦构成二维判决，用于拦截照片攻击

## 目录结构

```
.
├── face_recognition_doorlock.py   # 主程序：三线程 + 光流跟踪 + MQTT
├── collect_faces.py               # 人脸库采集工具（目录名即标签）
├── config.py                      # 所有可调参数
├── local_secrets.example.py       # 私钥模板（复制为 local_secrets.py）
├── liveness/                      # 活体实验脚本（方案 B）
│   ├── collect_rho_video.py       # 采集 ρ 的时间序列
│   ├── run_liveness_session.py    # 引导式采集（屏幕提示 真人/照片）
│   ├── camera_aim.py              # 取景辅助
│   ├── bench_fps.py               # 实测帧率
│   └── collect_faces_safe.py      # 带掉线重连的采集工具
├── tools/                         # TensorRT 相关实验脚本（默认禁用）
├── scripts/                       # 环境安装与开机自启
├── docs/CODE_MAP.md               # 主程序 738 行逐段索引与踩坑记录
└── known_faces/                   # 人脸库（样本不入库，仅保留目录）
```

## 硬件组成

| 设备 | 型号 | 作用 |
|---|---|---|
| 计算平台 | Jetson Nano（4×Cortex-A57，4 GB） | 人脸检测/识别、协议处理 |
| 摄像头 | USB UVC，640×480 | 图像采集 |
| 主控 | ESP32-P4（板载 Wi-Fi） | 外设控制、继电器驱动、MQTT 接入 |
| 传感器节点 | MSPM0G3507 | 环境状态采集（UART 挂在 P4 下） |
| 语音终端 | 小智（独立 Wi-Fi） | 语音交互与播报 |
| 执行器 | 继电器模块 | 门锁通断 |

## 快速开始

### 1. 环境

```bash
bash scripts/install_jetson_env.sh
# 或手动：
python3 -m venv venv38
./venv38/bin/pip install -r requirements.txt
```

### 2. 配置私钥

```bash
cp local_secrets.example.py local_secrets.py
# 编辑 local_secrets.py，填入巴法云私钥；或用环境变量：
export BEMFA_KEY=你的私钥
```

### 3. 建人脸库

```bash
cd /path/to/project
./venv38/bin/python collect_faces.py --name 你的名字 --num 10
```

目录名即标签。采集时按 `空格` 拍照、`q` 退出；建议 10~20 张，覆盖不同角度与光照。

### 4. 运行

```bash
./venv38/bin/python face_recognition_doorlock.py
```

按键：`q` 退出，`r` 重新加载人脸库，`c` 抓拍当前人脸并存入库。

## 配置项

所有参数集中在 `config.py`：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `BEMFA_KEY` | 空 | 从环境变量或 `local_secrets.py` 读取 |
| `DOOR_TOPIC` | `men006` | 开门指令主题（发布时自动加 `/set`） |
| `PICTURE_TOPIC` | `facepicture` | 抓拍图主题 |
| `CAM_W / CAM_H` | 640 / 480 | 请求分辨率（驱动实际返回值可能不同） |
| `THRESHOLD` | 0.40 | **1 − 余弦阈值**，即要求 cos > 0.60 |
| `MARGIN` | 0.03 | 高出次优的最小差值 |
| `DETECT_INTERVAL` | 2.0 | 重识别周期（秒） |
| `MAX_FACE_AGE` | 3.0 | 识别结果有效期（秒） |
| `SEND_COOLDOWN` | 5.0 | 开门冷却（秒） |
| `TRACK_MAX_FEAT` | 15 | 光流跟踪角点数 |
| `RHO_MIN` | 0.032 | 活体判据阈值（方案 B） |

## 已知限制

1. 当前无活体检测（`liveness/` 为实验代码，尚未接入主循环），静态照片可骗过
2. 单人库时二次门限失效（次优分数恒为 −1）
3. USB 摄像头供电不稳时会被内核重新枚举（`/dev/video0` 可能变成 `/dev/video1`）
4. TensorRT FP16 推理在 TRT 8.2 上得到的特征与 CPU 不一致，当前统一使用 CPU 推理
5. 缓存只按时间戳判断新旧，删除样本后需手动删除 `_embeddings_cache.npz`

## 许可

MIT License，详见 [LICENSE](LICENSE)。
