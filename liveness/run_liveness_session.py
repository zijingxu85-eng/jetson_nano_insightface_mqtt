#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
引导式活体数据采集（屏幕提示版）
================================
在 Jetson 屏幕上弹出窗口，自动切换两段：
    PHASE 1/2  LIVE   15s  —— 真人：说话 / 轻微转头
    (8s 切换提示)
    PHASE 2/2  PHOTO  15s  —— 举照片：轻微晃动 / 左右倾斜

窗口显示：阶段、倒计时、人脸框（绿=OK 红=未检出）、跟踪角点、实时 rho。
结束时在窗口里显示两段统计，并写 live_60.csv / attack_60.csv。

用法:
    DISPLAY=:0 ./venv38/bin/python run_liveness_session.py
    可选: --live 15 --attack 15 --corners 60 --outdir /home/xinghuo
"""
import argparse
import csv
import os
import sys
import time

os.environ.setdefault("DISPLAY", ":0")

import cv2
import numpy as np

RANSAC_PX = 1.5
WIN = "LIVENESS CAPTURE"
BORDER = 10


def make_mask(shape, box=None):
    m = np.zeros(shape[:2], np.uint8)
    if box is None:
        m[BORDER:shape[0] - BORDER, BORDER:shape[1] - BORDER] = 255
    else:
        x1, y1, x2, y2 = box
        x1 = max(BORDER, int(x1)); y1 = max(BORDER, int(y1))
        x2 = min(shape[1] - BORDER, int(x2)); y2 = min(shape[0] - BORDER, int(y2))
        if x2 > x1 + 8 and y2 > y1 + 8:
            m[y1:y2, x1:x2] = 255
    return m


def measure(prev, cur, corners, box):
    """返回 (rho_sim, rho_hom, med_disp, pts) ；pts 用于画点"""
    mask = make_mask(cur.shape, box)
    p0 = cv2.goodFeaturesToTrack(prev, maxCorners=corners, qualityLevel=0.01,
                                 minDistance=6, mask=mask)
    if p0 is None or len(p0) < 10:
        return None
    p1 = p0.reshape(-1, 2).astype(np.float32)
    p2, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(15, 15), maxLevel=2)
    if p2 is None:
        return None
    st = st.ravel() == 1
    p1, p2 = p1[st], p2.reshape(-1, 2)[st]
    if len(p1) < 10:
        return None
    d = p2 - p1
    med_disp = float(np.median(np.linalg.norm(d, axis=1)))
    res = {}
    for model in ("sim", "hom"):
        if model == "sim":
            M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                                 ransacReprojThreshold=RANSAC_PX)
            if M is None:
                res[model] = np.nan; continue
            pred = (M @ np.hstack([p1, np.ones((len(p1), 1))]).T).T
        else:
            H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, RANSAC_PX)
            if H is None:
                res[model] = np.nan; continue
            pred = cv2.perspectiveTransform(p1.reshape(-1, 1, 2), H).reshape(-1, 2)
        inl = inl.ravel().astype(bool)
        rr = np.linalg.norm(p2 - pred, axis=1)
        mr = float(np.median(rr[inl])) if inl.sum() >= 8 else np.nan
        res[model] = (mr / med_disp) if med_disp > 0.5 else np.nan
    return res["sim"], res["hom"], med_disp, p2


def put(vis, txt, y, scale=0.62, color=(255, 255, 255), thick=2, x=12):
    cv2.putText(vis, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(vis, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=float, default=15)
    ap.add_argument("--attack", type=float, default=15)
    ap.add_argument("--prep", type=float, default=3)
    ap.add_argument("--switch", type=float, default=8)
    ap.add_argument("--corners", type=int, default=60)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--outdir", default="/home/xinghuo")
    a = ap.parse_args()

    from insightface.app import FaceAnalysis
    det = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                       allowed_modules=["detection"])
    det.prepare(ctx_id=0, det_size=(320, 320))
    print("[INIT] 检测器就绪")

    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[ERR] 摄像头打不开"); sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 720, 540)
    cv2.moveWindow(WIN, 60, 60)

    ok, f = cap.read()
    if not ok:
        print("[ERR] 读不到帧"); sys.exit(1)
    prev = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    box = None
    last_det = 0.0

    phases = [("LIVE", a.live, "LIVE MODE : talk / turn head slightly", "真人模式：说话 + 轻微转头"),
              ("PHOTO", a.attack, "PHOTO MODE: hold photo, wobble it", "照片模式：举照片 + 轻微晃动")]

    all_rows = {}

    def grab():
        nonlocal prev, box, last_det
        ok, frame = cap.read()
        if not ok:
            return None, None, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now = time.time()
        if now - last_det > 2.0:
            fs = det.get(frame)
            if fs:
                bb = max(fs, key=lambda v: (v.bbox[2] - v.bbox[0]) * (v.bbox[3] - v.bbox[1])).bbox
                box = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
            else:
                box = None
            last_det = now
        return frame, gray, now

    # ---------- 每段：准备倒计时（等入镜）→ 采集 ----------
    for pi, (tag, dur, en, cn) in enumerate(phases):
        print("\n=== 阶段 %d/2 %s : %s ===" % (pi + 1, tag, cn))
        t_wait = time.time()
        started = False
        while True:
            frame, gray, now = grab()
            if frame is None:
                break
            vis = frame.copy()
            waited = now - t_wait
            remain = max(0, a.prep - waited) if box else 0
            if box is not None and waited > a.prep:
                started = True
                break
            if waited > 25:  # 兜底：实在检不到也开
                started = True
                break
            cv2.rectangle(vis, (0, 0), (640, 480), (0, 0, 0), -1)
            put(vis, "PHASE %d/2   %s" % (pi + 1, tag), 60, 0.9, (0, 255, 255))
            put(vis, en, 110, 0.65, (255, 255, 255))
            if box is None:
                put(vis, "NO FACE - step back 40-60cm, face the camera", 200, 0.62, (0, 0, 255))
                put(vis, "未检出人脸：请退后到一臂距离、正对镜头", 240, 0.55, (0, 0, 255))
            else:
                w = box[2] - box[0]; h = box[3] - box[1]
                put(vis, "FACE OK  %dx%d" % (w, h), 200, 0.7, (0, 255, 0))
                put(vis, "starting in %.0f" % max(0, a.prep - waited + 1), 260, 1.6, (0, 255, 255))
            cv2.imshow(WIN, vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); cv2.destroyAllWindows(); sys.exit(0)
            prev = gray

        if not started:
            break

        out = os.path.join(a.outdir, "%s_%d.csv" % (tag.lower(), a.corners))
        rows = []
        t0 = time.time()
        idx = 0
        with open(out, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["frame", "t", "n_pts", "med_disp_px", "rho_sim", "rho_hom",
                         "sharpness", "brightness", "face"])
            while time.time() - t0 < dur:
                frame, gray, now = grab()
                if frame is None:
                    break
                m = measure(prev, gray, a.corners, box)
                rho_s = rho_h = np.nan
                if m is not None:
                    rho_s, rho_h, mdisp, pts = m
                    vis = frame.copy()
                    for p in pts.astype(int):
                        cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
                    wr.writerow([idx, round(now - t0, 3), len(pts), round(mdisp, 3),
                                 round(rho_s, 4) if not np.isnan(rho_s) else "",
                                 round(rho_h, 4) if not np.isnan(rho_h) else "",
                                 round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
                                 round(float(gray.mean()), 1), 1 if box else 0])
                    rows.append((rho_s, rho_h))
                    idx += 1
                else:
                    vis = frame.copy()

                left = dur - (now - t0)
                if box:
                    col = (0, 255, 0)
                    cv2.rectangle(vis, box[:2], box[2:], col, 2)
                cv2.rectangle(vis, (0, 0), (640, 118), (0, 0, 0), -1)
                put(vis, "PHASE %d/2  %s   %4.1fs left" % (pi + 1, tag, left), 38, 0.78, (0, 255, 255))
                put(vis, en, 76, 0.58, (255, 255, 255))
                put(vis, "pairs %d" % len(rows), 108, 0.55, (200, 200, 200), 1)
                if m is not None:
                    v = rho_s if not np.isnan(rho_s) else -1
                    c = (0, 255, 0) if v >= 0.10 else (0, 165, 255)
                    put(vis, "rho_sim %.3f  rho_hom %.3f  disp %.1fpx"
                        % (v, (rho_h if not np.isnan(rho_h) else -1), mdisp), 452, 0.62, c)
                cv2.imshow(WIN, vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                prev = gray
        all_rows[tag] = rows
        print("  已保存 %s   帧对 %d" % (out, len(rows)))

        # 阶段间切换提示
        if pi < len(phases) - 1:
            t_sw = time.time()
            while time.time() - t_sw < a.switch:
                frame, gray, now = grab()
                if frame is None:
                    break
                vis = frame.copy()
                cv2.rectangle(vis, (0, 0), (640, 480), (0, 0, 0), -1)
                put(vis, "NEXT:  PHASE %d/2  PHOTO" % (pi + 2), 120, 0.95, (0, 255, 255))
                put(vis, "Now hold your PHOTO in front of camera", 190, 0.66, (255, 255, 255))
                put(vis, "现在请举着照片（手机显示你的正脸照）", 240, 0.6, (255, 255, 255))
                put(vis, "%0.0f" % max(0, a.switch - (time.time() - t_sw) + 1), 360, 3.0, (0, 255, 0), 4)
                cv2.imshow(WIN, vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                prev = gray

    # ---------- 汇总 ----------
    print("\n" + "=" * 62)
    summary = []
    for tag, rows in all_rows.items():
        if not rows:
            continue
        arr = np.array(rows, dtype=np.float64)
        line = "%s:" % tag
        for i, nm in ((0, "rho_sim"), (1, "rho_hom")):
            v = arr[:, i]; v = v[~np.isnan(v)]
            if len(v):
                line += "  %s n=%d med=%.3f p10=%.3f p90=%.3f" % (
                    nm, len(v), np.median(v), np.percentile(v, 10), np.percentile(v, 90))
        print(line)
        summary.append((tag, line))

    vis = np.zeros((480, 640, 3), np.uint8)
    put(vis, "DONE  (window closes in 8s)", 60, 0.8, (0, 255, 255))
    y = 120
    for tag, line in summary:
        put(vis, line[:64], y, 0.45, (255, 255, 255), 1); y += 34
    put(vis, "press any key to close", 440, 0.55, (180, 180, 180), 1)
    cv2.imshow(WIN, vis)
    cv2.waitKey(8000)
    cap.release()
    cv2.destroyAllWindows()
    print("=" * 62)


if __name__ == "__main__":
    main()
