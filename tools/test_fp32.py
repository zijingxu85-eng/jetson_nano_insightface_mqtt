"""Test FP32 GPU embedding"""
import subprocess, json, base64, cv2, numpy as np, time, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

img = cv2.imread("/tmp/aligned_test.jpg")
_, jpeg = cv2.imencode(".jpg", img)
b64 = base64.b64encode(jpeg).decode()

proc = subprocess.Popen(
    ["python3.6", "trt_face.py", "--service"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1
)
time.sleep(4)

proc.stdin.write(json.dumps({"image_base64": b64}) + "\n")
proc.stdin.flush()
resp = proc.stdout.readline().strip()
proc.terminate()

result = json.loads(resp)
emb = np.array(result.get("embedding", []), dtype=np.float32)
print("shape=%s norm=%.4f first5=%s" % (emb.shape, np.linalg.norm(emb), emb[:5]))
if abs(np.linalg.norm(emb) - 1.0) < 0.1:
    print("FP32 NORMALIZED - GOOD!")
else:
    print("FP32 NOT normalized - BAD!")
