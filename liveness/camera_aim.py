#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
取景预览（对位用）：在你屏幕上开一个窗口，帮你把摄像头对准脸。
不采集、不写文件，只显示：画面 + 人脸框 + 尺寸/占比 + FACE OK / NO FACE。

用法（在 Jetson 屏幕上）:
    DISPLAY=:0 ./venv38/bin/python camera_aim.py           # 按 q 退出
    DISPLAY=:0 ./venv38/bin/python camera_aim.py --seconds 120
同时把状态打到标准输出（便于远程查看进度）。
"""
import argparse
import os
import time

os.environ.setdefault("DISPLAY", ":0")
import cv2
import numpy as np

WIN = "AIM CAMERA  (q = quit)"
GUIDE_W, GUIDE_H = 260, 260      # 目标取景框


def put(vis, txt, y, scale=0.6, color=(255, 255, 255), thick=2, x=12):
    cv2.putText(vis, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(vis, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=0, help="0=一直运行直到按 q")
    a = ap.parse_args()

    from insightface.app import FaceAnalysis
    det = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                       allowed_modules=["detection"])
    det.prepare(ctx_id=0, det_size=(320, 320))

    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERR] camera open failed", flush=True)
        return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 720, 540)
    cv2.moveWindow(WIN, 60, 60)

    t0 = time.time()
    last = 0.0
    box = None
    print("[AIM] window opened, waiting for face...", flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        now = time.time()
        if now - last > 1.0:
            fs = det.get(frame)
            if fs:
                bb = max(fs, key=lambda v: (v.bbox[2] - v.bbox[0]) * (v.bbox[3] - v.bbox[1])).bbox
                box = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
                w = box[2] - box[0]; h = box[3] - box[1]
                print("[AIM] FACE %dx%d ratio=%.0f%%" % (w, h, 100.0 * w * h / (640 * 480)), flush=True)
            else:
                box = None
                print("[AIM] no face", flush=True)
            last = now

        vis = frame.copy()
        cx, cy = 320, 240
        cv2.rectangle(vis, (cx - GUIDE_W // 2, cy - GUIDE_H // 2),
                      (cx + GUIDE_W // 2, cy + GUIDE_H // 2), (120, 120, 120), 1)
        if box:
            cv2.rectangle(vis, box[:2], box[2:], (0, 255, 0), 2)
            w = box[2] - box[0]; h = box[3] - box[1]
            put(vis, "FACE OK  %dx%d  %.0f%%" % (w, h, 100.0 * w * h / (640 * 480)), 30, 0.7, (0, 255, 0))
            put(vis, "good - tell your assistant to start", 62, 0.5, (0, 255, 0), 1)
        else:
            put(vis, "NO FACE", 30, 0.8, (0, 0, 255))
            put(vis, "aim camera at your face / 40-60cm", 62, 0.52, (0, 0, 255), 1)
            put(vis, "centering guide = grey box", 92, 0.5, (180, 180, 180), 1)
        put(vis, "q = quit", 465, 0.5, (200, 200, 200), 1)
        cv2.imshow(WIN, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if a.seconds and now - t0 > a.seconds:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[AIM] closed", flush=True)


if __name__ == "__main__":
    main()
