"""Compare CPU vs GPU embeddings for the same face"""
import sys, os, cv2, numpy as np, json, base64, subprocess, time

sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')

# 1. Get insightface CPU embedding
from insightface.app import FaceAnalysis
model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(160, 160))

# Test with a known face photo
img = cv2.imread("known_faces/xuzijing/xuzijing_20260713_124712_01.jpg")
if img is None:
    print("No test image found")
    sys.exit(1)

faces = model.get(img)
if not faces:
    print("No face detected in test image")
    sys.exit(1)

face = faces[0]
cpu_emb = face.normed_embedding
print("CPU embedding: shape=%s norm=%.4f first5=%s" % (cpu_emb.shape, np.linalg.norm(cpu_emb), cpu_emb[:5]))

# 2. Get the aligned face from insightface
# The model.get() already aligned internally. Let's get the aligned crop.
b = face.bbox.astype(int)
print("Bbox:", b.tolist())

# 3. Get GPU embedding via trt_infer.py
# First, align the face using insightface's landmarks
lm = None
if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None and len(face.landmark_2d_106) >= 72:
    lm = np.array([face.landmark_2d_106[33], face.landmark_2d_106[87], 
                   face.landmark_2d_106[52], face.landmark_2d_106[61], face.landmark_2d_106[71]], dtype=np.float32)
elif hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None and len(face.landmark_3d_68) >= 68:
    lm = np.array([face.landmark_3d_68[36], face.landmark_3d_68[45],
                   face.landmark_3d_68[30], face.landmark_3d_68[48], face.landmark_3d_68[54]], dtype=np.float32)

aligned = None
if lm is not None:
    dst = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                    [41.5493,92.3655],[70.7299,92.2041]], dtype=np.float32)
    tform = cv2.estimateAffinePartial2D(lm, dst, method=cv2.LMEDS)
    if tform[0] is not None:
        aligned = cv2.warpAffine(img, tform[0], (112, 112))

if aligned is None:
    # Fallback: simple resize
    crop = img[max(0,b[1]):min(img.shape[0],b[3]), max(0,b[0]):min(img.shape[1],b[2])]
    if crop.size > 0:
        aligned = cv2.resize(crop, (112, 112))

if aligned is None:
    print("Cannot align face")
    sys.exit(1)

cv2.imwrite("/tmp/aligned_test.jpg", aligned)
print("Aligned face saved to /tmp/aligned_test.jpg")

# Get GPU embedding
_, jpeg = cv2.imencode(".jpg", aligned)
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
gpu_emb = np.array(result.get("embedding", []), dtype=np.float32)
print("GPU embedding: shape=%s norm=%.4f first5=%s" % (gpu_emb.shape, np.linalg.norm(gpu_emb), gpu_emb[:5]))

# Compare
if gpu_emb.size > 0 and cpu_emb.size > 0:
    # Normalize both
    gpu_norm = gpu_emb / np.linalg.norm(gpu_emb)
    cpu_norm = cpu_emb / np.linalg.norm(cpu_emb)
    sim = float(np.dot(gpu_norm, cpu_norm))
    print("\n=== COMPARISON ===")
    print("CPU norm: %.4f" % np.linalg.norm(cpu_emb))
    print("GPU norm: %.4f" % np.linalg.norm(gpu_emb))
    print("Cosine similarity: %.4f (%.1f%%)" % (sim, sim*100))
    if sim > 0.5:
        print("RESULT: MATCH (same person)")
    else:
        print("RESULT: DIFFERENT (or wrong embedding)")
