# -*- coding: utf-8 -*-
"""活体判决模块：单应残差比 ρ + 时间窗中位 + 三态输出

判据（实测标定，Jetson Nano + 640x480 USB 相机，GAP=12 ≈ 1 s 基线）：
    平面介质（照片/手机屏） ρ 中位 ≈ 0.0069
    真人（转头/说话）       ρ 中位 ≈ 0.0464
    阈值 RHO_MIN = 0.024，单帧错误约 12%，10 帧中位后趋近 0

用法：
    lv = LivenessJudge()
    state = lv.update(gray, bbox)      # 每帧调用
    # state: "WARMUP" | "OK" | "SPOOF" | "INVALID"
"""
from collections import deque

import cv2
import numpy as np


class LivenessJudge:
    def __init__(self, gap=12, window=6, rho_min=0.024,
                 corners=60, min_disp=1.0, max_disp=25.0):
        self.gap = int(gap)
        self.window = int(window)
        self.rho_min = float(rho_min)
        self.corners = int(corners)
        self.min_disp = float(min_disp)
        self.max_disp = float(max_disp)

        self._grays = deque(maxlen=self.gap + 1)
        self._rhos = deque(maxlen=self.window)
        self._state = "WARMUP"
        self._rho = None
        self._rho_bar = None
        self._info = ""
        self._bad = 0

    # ---------- 对外 ----------
    def reset(self):
        self._grays.clear()
        self._rhos.clear()
        self._state = "WARMUP"
        self._rho = None
        self._rho_bar = None
        self._info = ""
        self._bad = 0

    def update(self, gray, bbox):
        """返回状态字符串。bbox 为 (x1,y1,x2,y2)，无人脸时传 None。"""
        if bbox is None:
            self.reset()
            return self._state

        self._grays.append(gray)
        if len(self._grays) < self._grays.maxlen:
            self._state = "WARMUP"
            return self._state

        r = self._measure(self._grays[0], self._grays[-1], bbox)
        if r is None:
            self._bad += 1
            if self._bad > 8:
                self._state = "INVALID"
                self._info = "测量无效（纹理/位移不足）"
            return self._state

        rho, npts, inlr, disp = r
        self._bad = 0
        self._rho = rho
        self._info = "rho=%.4f n=%d inl=%.2f disp=%.1f" % (rho, npts, inlr, disp)

        if not (self.min_disp <= disp <= self.max_disp):
            # 位移过大/过小：LK 不可靠，不参与判决
            self._state = "INVALID"
            self._info += " (位移越界)"
            return self._state

        self._rhos.append(rho)
        if len(self._rhos) < self.window:
            self._state = "WARMUP"
            return self._state

        self._rho_bar = float(np.median(self._rhos))
        self._state = "OK" if self._rho_bar >= self.rho_min else "SPOOF"
        return self._state

    # ---------- 属性 ----------
    @property
    def state(self):
        return self._state

    @property
    def rho(self):
        return self._rho

    @property
    def rho_bar(self):
        return self._rho_bar

    @property
    def info(self):
        return self._info

    def label(self):
        return {"OK": "LIVE", "SPOOF": "SPOOF", "INVALID": "INVALID",
                "WARMUP": "WARMUP"}.get(self._state, self._state)

    # ---------- 内部 ----------
    def _measure(self, a, b, box):
        h, w = a.shape
        x1, y1, x2, y2 = [int(v) for v in box]
        pad = int(0.15 * (x2 - x1))
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        if x2 - x1 < 40 or y2 - y1 < 40:
            return None

        mask = np.zeros(a.shape, np.uint8)
        mask[y1:y2, x1:x2] = 255
        p0 = cv2.goodFeaturesToTrack(a, maxCorners=self.corners, qualityLevel=0.01,
                                     minDistance=6, mask=mask)
        if p0 is None or len(p0) < 12:
            return None

        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(19, 19), maxLevel=3)
        if p1 is None or st is None:
            return None
        st = st.ravel() == 1
        P = p0.reshape(-1, 2)[st]
        Q = p1.reshape(-1, 2)[st]
        if len(P) < 12:
            return None

        Hm, inl = cv2.findHomography(P, Q, cv2.RANSAC, 2.0)
        if Hm is None or inl is None:
            return None
        inl = inl.ravel().astype(bool)
        if inl.sum() < 8:
            return None

        Qh = cv2.perspectiveTransform(P.reshape(-1, 1, 2), Hm).reshape(-1, 2)
        e = np.linalg.norm(Q - Qh, axis=1)[inl]
        d = np.linalg.norm(Q - P, axis=1)[inl]
        md = float(np.median(d))
        if md < self.min_disp:
            return None
        return float(np.median(e) / md), int(len(P)), float(inl.mean()), md


if __name__ == "__main__":
    import time

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    lv = LivenessJudge()
    box = None
    t0 = time.time()
    while time.time() - t0 < 15:
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if box is None:
            box = (200, 120, 440, 400)
        s = lv.update(g, box)
        cv2.rectangle(f, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(f, "%s  %s" % (lv.label(), lv.info), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("liveness_judge", f)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
