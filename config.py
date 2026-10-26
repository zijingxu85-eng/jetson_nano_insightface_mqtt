# -*- coding: utf-8 -*-
"""集中配置：所有可调参数。

密钥不写在本文件里。BEMFA_KEY 的取值优先级：
    1) 环境变量 BEMFA_KEY
    2) 同目录下 local_secrets.py 中的 BEMFA_KEY（该文件已被 .gitignore 忽略，不会提交）

用法：把 local_secrets.example.py 复制为 local_secrets.py 并填入自己的私钥。
"""
import os

try:
    from local_secrets import BEMFA_KEY as _LOCAL_KEY
except Exception:
    _LOCAL_KEY = ""

# ---- 云端（巴法云）----
BEMFA_KEY = os.environ.get("BEMFA_KEY", _LOCAL_KEY)
DOOR_TOPIC = "men006"
PICTURE_TOPIC = "facepicture"

# ---- 人脸库 ----
KNOWN_FACES_DIR = "./known_faces"
EMBEDDINGS_CACHE = "./known_faces/_embeddings_cache.npz"

# ---- 摄像头 ----
CAMERA_ID = 0
CAM_W = 640
CAM_H = 480

# ---- 身份判决 ----
# THRESHOLD 是 "1 - 余弦阈值"，即 0.40 对应要求 cos > 0.60
THRESHOLD = 0.40
MARGIN = 0.03

# ---- 调度 ----
DETECT_INTERVAL = 2.0
MAX_FACE_AGE = 3.0
SEND_COOLDOWN = 5.0

# ---- 光流跟踪 ----
TRACK_MAX_FEAT = 15
TRACK_QUALITY = 0.02

# ---- 活体判据（实测标定：照片 0.0069 / 真人 0.0464 / 6.7 倍分离）----
RHO_MIN = 0.024
RHO_GAP = 12
RHO_WINDOW = 6
RHO_CORNERS = 60
