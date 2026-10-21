#!/usr/bin/env python3.6
"""TensorRT Recognition Service - just embedding extraction"""
from __future__ import print_function
import sys, os, json, time, base64, ctypes
import numpy as np
import tensorrt as trt
import cv2

class _Logger(trt.ILogger):
    def log(self, severity, msg):
        sys.stderr.write("[TRT] %s\n" % msg)
        sys.stderr.flush()

TRT_LOGGER = _Logger()
ENGINE_DIR = os.path.expanduser("~/.insightface/engines")
REC_ENGINE_FP16 = os.path.join(ENGINE_DIR, "w600k_r50_fp16.engine")
REC_ENGINE_FP32 = os.path.join(ENGINE_DIR, "w600k_r50_fp32.engine")
REC_SIZE = 112

CDP = ctypes.c_ulonglong
CU = ctypes.CDLL("libcuda.so.1")
for fn in ["cuInit","cuDeviceGet","cuMemAlloc_v2","cuMemFree_v2",
           "cuMemcpyHtoD_v2","cuMemcpyDtoH_v2"]:
    getattr(CU, fn).restype = ctypes.c_int

_cu_ok = [False]
def _cu():
    if _cu_ok[0]: return
    CU.cuInit(0)
    d = ctypes.c_int(); CU.cuDeviceGet(ctypes.byref(d), 0)
    p = CDP(); CU.cuMemAlloc_v2(ctypes.byref(p), 1024)
    CU.cuMemFree_v2(p)
    _cu_ok[0] = True

class Engine:
    def __init__(self, path):
        r = trt.Runtime(TRT_LOGGER)
        with open(path, "rb") as f:
            self.e = r.deserialize_cuda_engine(f.read())
        self.c = self.e.create_execution_context()
        nb = self.e.num_bindings
        self.ii = [i for i in range(nb) if self.e.binding_is_input(i)][0]
        self.oi = [i for i in range(nb) if not self.e.binding_is_input(i)][0]
    def run(self, x):
        _cu()
        self.c.set_binding_shape(self.ii, x.shape)
        di = CDP(); CU.cuMemAlloc_v2(ctypes.byref(di), x.nbytes)
        CU.cuMemcpyHtoD_v2(di, x.ctypes.data_as(ctypes.c_void_p), x.nbytes)
        osz = self.c.get_binding_shape(self.oi)
        out = np.empty(max(1, int(np.prod(osz))), dtype=np.float32)
        dout = CDP(); CU.cuMemAlloc_v2(ctypes.byref(dout), out.nbytes)
        bf = (CDP * 2)(); bf[0] = di; bf[1] = dout
        self.c.execute_v2(bf)
        CU.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), dout, out.nbytes)
        CU.cuMemFree_v2(di); CU.cuMemFree_v2(dout)
        return out

def preprocess(img):
    r = cv2.resize(img, (REC_SIZE, REC_SIZE)).astype(np.float32)[:, :, ::-1]
    return ((r - 127.5) / 127.5).transpose((2,0,1))[np.newaxis, :, :, :]

if "--build" in sys.argv:
    sys.stderr.write("Building engines...\n"); sys.stderr.flush()
    # ... skipped for brevity - already built
    sys.exit(0)

# Pick engine
if os.path.exists(REC_ENGINE_FP16):
    engine_path = REC_ENGINE_FP16; sys.stderr.write("Engine: FP16\n")
else:
    engine_path = REC_ENGINE_FP32; sys.stderr.write("Engine: FP32\n")
sys.stderr.flush()

rec = Engine(engine_path)
for _ in range(3):
    _ = rec.run(preprocess(np.zeros((112,112,3), dtype=np.uint8)))
sys.stderr.write("Ready\n"); sys.stderr.flush()

for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        req = json.loads(line)
        if req.get("test"):
            resp = {"status": "ok", "engine": os.path.basename(engine_path)}
        elif not req.get("image_base64"):
            resp = {"error": "No image", "embedding": []}
        else:
            nb = np.frombuffer(base64.b64decode(req["image_base64"]), dtype=np.uint8)
            img = cv2.imdecode(nb, cv2.IMREAD_COLOR)
            if img is None:
                resp = {"error": "Decode fail", "embedding": []}
            else:
                t0 = time.time()
                emb = rec.run(preprocess(img))
                norm = np.linalg.norm(emb)
                if norm > 0: emb = emb / norm
                t1 = time.time()
                resp = {"embedding": emb.tolist(), "inference_ms": (t1-t0)*1000, "norm": float(norm)}
    except Exception as e:
        resp = {"error": str(e), "embedding": []}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
