# -*- coding: utf-8 -*-
"""鲁棒采集：自动找 /dev/video*、MJPG 降带宽、掉线自动重连、屏幕大字提示
用法: ./venv38/bin/python collect_faces_safe.py --name xuzijing --auto 10
"""
import argparse, glob, os, sys, time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--auto", type=int, default=0, help="自动拍 N 张(0=手动)")
ap.add_argument("--interval", type=float, default=2.0, help="自动拍照间隔秒")
ap.add_argument("--wait", type=float, default=90.0, help="等脸的最长时间")
args = ap.parse_args()

OUT = Path("known_faces") / args.name
OUT.mkdir(parents=True, exist_ok=True)
print("输出目录: %s" % OUT.resolve(), flush=True)

from insightface.app import FaceAnalysis
det = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                   allowed_modules=["detection"])
det.prepare(ctx_id=0, det_size=(320, 320))
print("[DET] insightface CPU ready", flush=True)


def open_cam():
    nodes = sorted(glob.glob("/dev/video*"))
    for n in nodes:
        idx = int(n.replace("/dev/video", ""))
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release(); continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(0.4)
        ok, f = cap.read()
        if ok and f is not None:
            w, h = int(cap.get(3)), int(cap.get(4))
            print("[CAM] %s 打开成功 %dx%d" % (n, w, h), flush=True)
            return cap, n
        cap.release()
    return None, None


def sharpness(g):
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


cap, node = open_cam()
if cap is None:
    print("[CAM] 没有可用摄像头（/dev/video* 都不行）", flush=True); sys.exit(2)

WIN = "COLLECT - %s" % args.name
saved, fails, t0 = 0, 0, time.time()
last_shot = 0.0
faces, last_det_t = [], 0.0
msg, msg_t = "", 0.0

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        fails += 1
        if fails % 10 == 0:
            print("[CAM] 掉线，重连中 (第 %d 次)..." % fails, flush=True)
            cap.release(); time.sleep(0.5)
            cap, node = open_cam()
            if cap is None:
                time.sleep(1.0)
        time.sleep(0.05)
        if fails > 300:
            print("[CAM] 放弃", flush=True); break
        continue
    fails = 0
    disp = cv2.flip(frame.copy(), 1)
    gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
    h, w = disp.shape[:2]

    now = time.time()
    if now - last_det_t > 0.25:
        faces = []
        for f in det.get(disp):
            b = f.bbox.astype(int)
            faces.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(f.det_score)))
        last_det_t = now
    best = max(faces, key=lambda v: v[4]) if faces else None
    sh = sharpness(gray)

    if best:
        x1, y1, x2, y2, sc = best
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(disp, "FACE OK %.2f  %dx%d" % (sc, x2 - x1, y2 - y1),
                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(disp, "NO FACE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.rectangle(disp, (0, h - 40), (w, h), (30, 30, 30), -1)
    cv2.putText(disp, "%d/%d  sharp=%.0f  %s" % (saved, args.auto, sh, node or "-"),
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if msg and time.time() - msg_t < 1.2:
        cv2.putText(disp, msg, (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow(WIN, disp)
    key = cv2.waitKey(1) & 0xFF

    shoot = False
    if key == ord("q"):
        break
    elif key in (ord(" "), ord("c")):
        shoot = True
    elif args.auto and time.time() - last_shot >= args.interval:
        shoot = True

    if shoot:
        if not best:
            msg, msg_t = "NO FACE - 不保存", time.time()
            print("  [跳过] 未检出人脸", flush=True)
        elif (x2 - x1) < 80 or sh < 40:
            msg, msg_t = "TOO SMALL/BLURRY", time.time()
            print("  [跳过] 脸太小(%d)或太糊(%.0f)" % (x2 - x1, sh), flush=True)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = OUT / ("%s_%s_%02d.jpg" % (args.name, ts, saved + 1))
            if cv2.imwrite(str(p), frame):
                saved += 1
                last_shot = time.time()
                msg, msg_t = "CAPTURED %d" % saved, time.time()
                print("  [%d/%d] %s  face=%dx%d sharp=%.0f" % (saved, args.auto, p.name, x2 - x1, y2 - y1, sh), flush=True)
            else:
                print("  [错误] 写文件失败", flush=True)
        if not args.auto:
            time.sleep(0.15)

    if args.auto and saved >= args.auto:
        msg, msg_t = "DONE", time.time()
        cv2.imshow(WIN, disp); cv2.waitKey(600)
        break
    if args.auto and not best and time.time() - t0 > args.wait and saved == 0:
        print("[超时] %.0f 秒内没看到人脸" % args.wait, flush=True); break

cap.release(); cv2.destroyAllWindows()
print("完成：%d 张 -> %s" % (saved, OUT.resolve()), flush=True)
