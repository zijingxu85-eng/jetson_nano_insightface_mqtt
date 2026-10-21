#!/usr/bin/env python3
"""Updated bridge: CPU detection + GPU recognition"""
import subprocess, json, base64, cv2, numpy as np, os, time, threading, select
from pathlib import Path

class TRTRecognizer:
    """GPU-accelerated face recognition via TensorRT"""
    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.script = str(Path(__file__).parent / "trt_face.py")
    
    def start(self):
        if self.proc: return True
        eng = os.path.expanduser("~/.insightface/engines/w600k_r50_fp16.engine")
        if not os.path.exists(eng):
            subprocess.run(["python3.6", self.script, "--build"], check=True, timeout=600)
        self.proc = subprocess.Popen(
            ["python3.6", self.script, "--service"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1
        )
        import time as _t
        _t.sleep(4)
        # Test
        self.proc.stdin.write('{"test": true}\n'); self.proc.stdin.flush()
        r = self.proc.stdout.readline()
        return True
    
    def stop(self):
        if self.proc: self.proc.terminate(); self.proc = None
    
    def get_embedding(self, face_img):
        """Extract face embedding (GPU) from face crop image"""
        _, jpeg = cv2.imencode(".jpg", face_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(jpeg).decode()
        with self.lock:
            self.proc.stdin.write(json.dumps({"image_base64": b64}) + "\n")
            self.proc.stdin.flush()
            resp = self.proc.stdout.readline()
        if not resp: return None
        result = json.loads(resp.strip())
        emb = result.get("embedding", [])
        if emb: return np.array(emb, dtype=np.float32)
        return None
    
    def __del__(self):
        self.stop()
