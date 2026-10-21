"""Test TensorRT recognition engine directly"""
import tensorrt as trt, numpy as np, cv2, os, time

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
ENGINE_PATH = os.path.expanduser("~/.insightface/engines/w600k_r50_fp32.engine")

# Load engine
r = trt.Runtime(TRT_LOGGER)
with open(ENGINE_PATH, "rb") as f:
    engine = r.deserialize_cuda_engine(f.read())

print("Engine bindings:")
for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    shape = engine.get_binding_shape(i)
    dtype = engine.get_binding_dtype(i)
    is_input = engine.binding_is_input(i)
    print(f"  [{i}] {name} {'IN ' if is_input else 'OUT'} shape={shape} dtype={dtype}")

ctx = engine.create_execution_context()

# Set input shape
inp_shape = (1, 3, 112, 112)
ctx.set_binding_shape(0, inp_shape)

# Print output shape after setting input
out_shape = ctx.get_binding_shape(1)
print("\nAfter set_binding_shape:")
print(f"  Output shape: {out_shape}")

# Create test input: use insightface-style preprocessing
img = np.zeros((112, 112, 3), dtype=np.float32)
# Test with a simple pattern
img[:56] = 255
inp = img.transpose((2, 0, 1))[np.newaxis, ...].copy()
inp = (inp - 127.5) / 127.5

print(f"Input: shape={inp.shape} dtype={inp.dtype}")

# Allocate device memory and run
import ctypes
cuda = ctypes.CDLL("libcuda.so")

# Create CUDA context
cuInit = cuda.cuInit
cuInit.restype = ctypes.c_int
cuInit(0)

cuDeviceGet = cuda.cuDeviceGet
cuDeviceGet.restype = ctypes.c_int
dev = ctypes.c_int()
cuDeviceGet(ctypes.byref(dev), 0)

cuCtxCreate = cuda.cuCtxCreate_v2
cuCtxCreate.restype = ctypes.c_int
ctx_handle = ctypes.c_void_p()
cuCtxCreate(ctypes.byref(ctx_handle), 0, dev)

# Allocate memory
cuMemAlloc = cuda.cuMemAlloc_v2
cuMemAlloc.restype = ctypes.c_int
d_input = ctypes.c_void_p()
cuMemAlloc(ctypes.byref(d_input), inp.nbytes)

cuMemcpyHtoD = cuda.cuMemcpyHtoD_v2
cuMemcpyHtoD.restype = ctypes.c_int
cuMemcpyHtoD(d_input, inp.ctypes.data_as(ctypes.c_void_p), inp.nbytes)

# Output buffer
out = np.empty(int(np.prod(out_shape)), dtype=np.float32)
d_output = ctypes.c_void_p()
cuMemAlloc(ctypes.byref(d_output), out.nbytes)

# Execute
bindings = [int(d_input.value), int(d_output.value)]
ctx.execute_v2(bindings)

cuMemcpyDtoH = cuda.cuMemcpyDtoH_v2
cuMemcpyDtoH(out.ctypes.data_as(ctypes.c_void_p), d_output, out.nbytes)

print(f"\nOutput: shape={out.shape} norm={np.linalg.norm(out):.4f} first5={out[:5]}")
print(f"Min={out.min():.4f} Max={out.max():.4f}")

cuMemFree = cuda.cuMemFree_v2
cuMemFree(d_input)
cuMemFree(d_output)
