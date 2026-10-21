#!/usr/bin/env python3.6
"""One-shot TensorRT inference: reads JSON from stdin, outputs JSON to stdout"""
import sys, json, base64, time, os, ctypes
import numpy as np
import tensorrt as trt
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_DIR = os.path.expanduser("~/.insightface/engines")
DET_ENGINE = os.path.join(ENGINE_DIR, "det_10g_fp16.engine")
REC_ENGINE = os.path.join(ENGINE_DIR, "w600k_r50_fp16.engine")
DET_SIZE = 160
REC_SIZE = 112

# CUDA driver API
CDP = ctypes.c_ulonglong
CU = ctypes.CDLL("libcuda.so.1")
CU.cuInit.restype = ctypes.c_int; CU.cuInit.argtypes = [ctypes.c_uint]
CU.cuDeviceGet.restype = ctypes.c_int; CU.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
CU.cuCtxCreate_v2.restype = ctypes.c_int; CU.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
CU.cuMemAlloc_v2.restype = ctypes.c_int; CU.cuMemAlloc_v2.argtypes = [ctypes.POINTER(CDP), ctypes.c_size_t]
CU.cuMemFree_v2.restype = ctypes.c_int; CU.cuMemFree_v2.argtypes = [CDP]
CU.cuMemcpyHtoD_v2.restype = ctypes.c_int; CU.cuMemcpyHtoD_v2.argtypes = [CDP, ctypes.c_void_p, ctypes.c_size_t]
CU.cuMemcpyDtoH_v2.restype = ctypes.c_int; CU.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, CDP, ctypes.c_size_t]

_ctx = None
def _cu():
    global _ctx
    if _ctx: return
    CU.cuInit(0); d = ctypes.c_int(); CU.cuDeviceGet(ctypes.byref(d), 0)
    _ctx = ctypes.c_void_p(); CU.cuCtxCreate_v2(ctypes.byref(_ctx), 0, d.value)

class Engine:
    def __init__(self, path):
        r = trt.Runtime(TRT_LOGGER)
        with open(path, "rb") as f:
            self.e = r.deserialize_cuda_engine(f.read())
        self.c = self.e.create_execution_context()
        nb = self.e.num_bindings
        self.ii = [i for i in range(nb) if self.e.binding_is_input(i)]
        self.oi = [i for i in range(nb) if not self.e.binding_is_input(i)]
    def run(self, x):
        _cu()
        self.c.set_binding_shape(self.ii[0], x.shape)
        di = CDP(); CU.cuMemAlloc_v2(ctypes.byref(di), x.nbytes)
        CU.cuMemcpyHtoD_v2(di, x.ctypes.data_as(ctypes.c_void_p), x.nbytes)
        do = []; ho = []
        for i in self.oi:
            s = self.c.get_binding_shape(i)
            sz = max(1, int(np.prod(s)))
            h = np.empty(sz, dtype=np.float32)
            d = CDP(); CU.cuMemAlloc_v2(ctypes.byref(d), h.nbytes)
            do.append(d); ho.append(h)
        bf = (CDP * (1 + len(do)))()
        bf[0] = di
        for j, d in enumerate(do): bf[1+j] = d
        self.c.execute_v2(bf)
        for h, d in zip(ho, do):
            CU.cuMemcpyDtoH_v2(h.ctypes.data_as(ctypes.c_void_p), d, h.nbytes)
        CU.cuMemFree_v2(di)
        for d in do: CU.cuMemFree_v2(d)
        return ho

def blob(img, sz):
    r = cv2.resize(img, (sz, sz)).astype(np.float32)
    r = r[:, :, ::-1]  # BGR->RGB
    return ((r - 127.5) / 127.5).transpose((2, 0, 1))[np.newaxis, :, :, :]

# Main
data = json.loads(sys.stdin.read())
nb = np.frombuffer(base64.b64decode(data["image_base64"]), dtype=np.uint8)
img = cv2.imdecode(nb, cv2.IMREAD_COLOR)
if img is None:
    print(json.dumps({"error": "decode failed", "faces": []}))
    sys.exit()

de = Engine(DET_ENGINE)
re = Engine(REC_ENGINE)
bl = blob(img, DET_SIZE)
t0 = time.time(); do = de.run(bl); t1 = time.time()
faces = []
if len(do) >= 4:
    sc = do[0].reshape(-1); bb = do[3].reshape(-1, 4)
    h, w = img.shape[:2]; sx, sy = w / DET_SIZE, h / DET_SIZE
    for j in range(min(len(sc), len(bb))):
        if sc[j] > 0.5:
            x1, y1, x2, y2 = bb[j]
            fi = {"bbox": [float(x1*sx), float(y1*sy), float(x2*sx), float(y2*sy)], "score": float(sc[j])}
            try:
                cr = img[int(y1*sy):int(y2*sy), int(x1*sx):int(x2*sx)]
                if cr.size > 0:
                    ro = re.run(blob(cr, REC_SIZE))
                    if ro: fi["embedding"] = ro[0].tolist()
            except: pass
            faces.append(fi)
print(json.dumps({"faces": faces, "inference_ms": (t1-t0)*1000, "num_faces": len(faces)}))
