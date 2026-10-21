#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
门禁活体判据：视频数据采集（v2，人脸框内取点）
==============================================
与门锁实现一致：先用 insightface 检测人脸框（每 1s 一次），**只在框内**取角点做光流，
避免背景运动污染 rho。若无 insightface 或 --no-face，则退回全帧。

用法:
  python3 collect_rho_video.py --mode live   --seconds 15 --corners 60 --out live_60.csv
  python3 collect_rho_video.py --mode attack --seconds 15 --corners 60 --out attack_60.csv
  python3 collect_rho_video.py --selftest
  python3 collect_rho_video.py --no-face            # 不做人脸框，全帧取点（快，但会被背景污染）
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

BORDER = 10
RANSAC_PX = 1.5

CASCADE_CANDIDATES = [
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml") if hasattr(cv2, "data") else "",
    "/home/xinghuo/face_env/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml"),
]
cascade_path = next((p for p in CASCADE_CANDIDATES if p and os.path.exists(p)), "")


def make_mask(shape, box=None):
    m = np.zeros(shape[:2], np.uint8)
    if box is None:
        m[BORDER:shape[0] - BORDER, BORDER:shape[1] - BORDER] = 255
    else:
        x1, y1, x2, y2 = box
        x1 = max(BORDER, x1); y1 = max(BORDER, y1)
        x2 = min(shape[1] - BORDER, x2); y2 = min(shape[0] - BORDER, y2)
        if x2 > x1 + 8 and y2 > y1 + 8:
            m[y1:y2, x1:x2] = 255
    return m


def robust_rho(prev, cur, max_corners, mask):
    p0 = cv2.goodFeaturesToTrack(prev, maxCorners=max_corners,
                                 qualityLevel=0.01, minDistance=6, mask=mask)
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
    out = [len(p1)]
    for model in ("sim", "hom"):
        try:
            if model == "sim":
                M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                                     ransacReprojThreshold=RANSAC_PX)
                if M is None:
                    out += [np.nan, np.nan, np.nan]; continue
                pred = (M @ np.hstack([p1, np.ones((len(p1), 1))]).T).T
            else:
                H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, RANSAC_PX)
                if H is None:
                    out += [np.nan, np.nan, np.nan]; continue
                pred = cv2.perspectiveTransform(p1.reshape(-1, 1, 2), H).reshape(-1, 2)
            inl = inl.ravel().astype(bool)
            res = np.linalg.norm(p2 - pred, axis=1)
            med_res = float(np.median(res[inl])) if inl.sum() >= 8 else np.nan
            out += [float(inl.mean()), med_res, (med_res / med_disp if med_disp > 0.5 else np.nan)]
        except cv2.error:
            out += [np.nan, np.nan, np.nan]
    n, ir_s, mr_s, rho_s, ir_h, mr_h, rho_h = out
    return n, ir_s, med_disp, mr_s, rho_s, mr_h, rho_h


def selftest():
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((240, 240)) * 255).astype(np.uint8), (5, 5), 0)
    moved = cv2.warpAffine(base, np.float32([[1, 0, 4], [0, 1, -3]]), (240, 240),
                           borderMode=cv2.BORDER_REPLICATE)
    r = robust_rho(base, moved, 60, make_mask(base.shape))
    print("[selftest]", r)
    assert r is not None and not np.isnan(r[4]) and r[4] < 0.05
    print("[selftest] OK  rho_sim=%.5f" % r[4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="live", choices=["live", "attack"])
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--corners", type=int, default=60)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--gst", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--no-face", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest(); return

    det = None
    if not a.no_face:
        try:
            from insightface.app import FaceAnalysis
            det = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                               allowed_modules=["detection"])
            det.prepare(ctx_id=0, det_size=(320, 320))
            print("[INIT] insightface 检测器就绪（只在人脸框内取点）")
        except Exception as e:
            print("[WARN] insightface 不可用(%s)，退回全帧取点" % e)
            det = None

    cascade = None
    if det is None and hasattr(cv2, "CascadeClassifier") and cascade_path:
        c = cv2.CascadeClassifier(cascade_path)
        cascade = None if c.empty() else c

    if a.gst:
        pipeline = ("nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! "
                    "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
                    "appsink drop=1")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(a.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[ERR] 打不开摄像头"); sys.exit(1)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("[CAM] %dx%d @ %.0ffps" % (W, Hh, fps))

    print("=" * 62)
    print(" 模式: %s   角点: %d   时长: %.1fs   取点区域: %s"
          % (a.mode, a.corners, a.seconds, "人脸框内" if (det or cascade) else "全帧"))
    print(" " + ("请站到镜头前，自然说话、轻微转头" if a.mode == "live"
                 else "请举着照片，轻微晃动 + 左右倾斜"))
    print("=" * 62)

    out = a.out or ("%s_%d.csv" % (a.mode, a.corners))
    ok, first = cap.read()
    if not ok:
        print("[ERR] 读不到帧"); sys.exit(1)
    prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    t0 = time.time(); idx = 0; last_det = -1.0; box = None; rows = []
    fps_cnt = 0; fps_t = time.time()

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "n_pts", "inlier_ratio", "med_disp_px", "med_res_px",
                    "rho_sim", "rho_hom", "sharpness", "brightness", "face"])
        while time.time() - t0 < a.seconds:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now = time.time()

            if (det is not None or cascade is not None) and (now - last_det > 2.0):
                if det is not None:
                    fs = det.get(frame)
                    if fs:
                        bb = max(fs, key=lambda v: (v.bbox[2] - v.bbox[0]) * (v.bbox[3] - v.bbox[1])).bbox
                        box = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
                    else:
                        box = None
                else:
                    fs = cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
                    box = None if len(fs) == 0 else tuple(int(v) for v in
                          (lambda x, y, ww, hh: (x, y, x + ww, y + hh))(*max(fs, key=lambda v: v[2] * v[3])))
                last_det = now
                if (det is not None or cascade is not None) and now - t0 < 4:
                    if box:
                        print("[FACE] 检出 %dx%d @ (%d,%d)  画面占比 %.0f%%"
                              % (box[2] - box[0], box[3] - box[1], box[0], box[1],
                                 100.0 * (box[2] - box[0]) * (box[3] - box[1]) / (640 * 480)))
                    else:
                        print("[FACE] 未检出 —— 请退后到 40~60cm，脸占画面 1/3~1/2，别贴太近")

            mask = make_mask(gray.shape, box)
            r = robust_rho(prev, gray, a.corners, mask)
            if r is not None:
                n, ir, md, mr_s, rho_s, mr_h, rho_h = r
                sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                w.writerow([idx, round(now - t0, 3), n, round(ir, 3), round(md, 3),
                            round(mr_s, 3),
                            round(rho_s, 4) if not np.isnan(rho_s) else "",
                            round(rho_h, 4) if not np.isnan(rho_h) else "",
                            round(sharp, 1), round(float(gray.mean()), 1), 1 if box else 0])
                rows.append((rho_s, rho_h))

            if not a.no_display:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if box:
                    cv2.rectangle(vis, box[:2], box[2:], (0, 255, 0), 2)
                if r is not None:
                    cv2.putText(vis, "rho_sim %.3f | rho_hom %.3f | d %.1fpx"
                                % (r[4] if not np.isnan(r[4]) else -1,
                                   r[6] if not np.isnan(r[6]) else -1, r[2]),
                                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.imshow("rho %s" % a.mode, vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            prev = gray; idx += 1
            fps_cnt += 1
            if now - fps_t > 2.0:
                print("  ... %.1fs  帧率 %.1f fps  已记录 %d 对" % (now - t0, fps_cnt / (now - fps_t), len(rows)))
                fps_cnt = 0; fps_t = now

    cap.release()
    if not a.no_display:
        cv2.destroyAllWindows()

    print("\n已保存: %s   有效帧对: %d" % (out, len(rows)))
    if rows:
        s = np.array(rows, dtype=np.float64)
        for i, name in ((0, "rho_sim"), (1, "rho_hom")):
            v = s[:, i]; v = v[~np.isnan(v)]
            if len(v):
                print("  %-8s n=%d  mean=%.4f  median=%.4f  p10=%.4f  p90=%.4f"
                      % (name, len(v), v.mean(), np.median(v),
                         np.percentile(v, 10), np.percentile(v, 90)))


if __name__ == "__main__":
    main()
