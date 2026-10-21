#!/usr/bin/env python3
"""Face collection tool - CPU detection + GPU feedback"""
import cv2, sys, time, argparse
from pathlib import Path
from datetime import datetime
KNOWN_FACES_DIR = "./known_faces"

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--num", type=int, default=5)
parser.add_argument("--camera", type=int, default=0)
args = parser.parse_args()

out = Path(KNOWN_FACES_DIR) / args.name
out.mkdir(parents=True, exist_ok=True)
print("="*50); print(f"  Face Collection: {args.name}"); print("="*50)

# CPU face detector (OpenCV DNN)
det_net = None
import os
proto = str(Path(__file__).parent / "face_detector" / "deploy.prototxt")
model = str(Path(__file__).parent / "face_detector" / "res10_300x300_ssd_iter_140000.caffemodel")
if os.path.exists(model):
    det_net = cv2.dnn.readNetFromCaffe(proto, model)
else:
    from insightface.app import FaceAnalysis
    det_net = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'],
                           allowed_modules=['detection'])
    det_net.prepare(ctx_id=0, det_size=(160, 160))
    print("[DET] Using insightface CPU")

cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened(): print("[CAM] FAIL"); sys.exit(1)

collected = 0
print("\n  SPACE=capture  q=quit\n")

while True:
    ok, frame = cap.read()
    if not ok: time.sleep(0.05); continue
    display = cv2.flip(frame.copy(), 1)
    h, w = display.shape[:2]
    
    faces = []
    if hasattr(det_net, 'get'):  # insightface
        r = det_net.get(display)
        for f in r:
            b = f.bbox.astype(int)
            faces.append({"bbox": [b[0],b[1],b[2],b[3]], "score": float(f.det_score)})
    else:  # OpenCV DNN
        blob = cv2.dnn.blobFromImage(display, 1.0, (300,300), (104,177,123))
        det_net.setInput(blob)
        det = det_net.forward()
        for i in range(det.shape[2]):
            conf = det[0,0,i,2]
            if conf > 0.5:
                x1=int(det[0,0,i,3]*w); y1=int(det[0,0,i,4]*h)
                x2=int(det[0,0,i,5]*w); y2=int(det[0,0,i,6]*h)
                faces.append({"bbox":[x1,y1,x2,y2],"score":float(conf)})
    
    if faces:
        best = sorted(faces, key=lambda f: f["score"], reverse=True)[0]
        b = best["bbox"]
        cv2.rectangle(display, (b[0],b[1]), (b[2],b[3]), (0,255,0), 2)
        cv2.putText(display, f"Face {collected}/{args.num}", (b[0],b[1]-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
    else:
        cv2.putText(display, "No face", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    cv2.rectangle(display, (0,display.shape[0]-32), (display.shape[1],display.shape[0]), (40,40,40), -1)
    bar_w = int(400 * collected / max(args.num,1))
    cv2.rectangle(display, (10,display.shape[0]-26), (410,display.shape[0]-8), (100,100,100), 2)
    cv2.rectangle(display, (10,display.shape[0]-26), (10+bar_w,display.shape[0]-8), (0,255,0), -1)
    cv2.putText(display, f" {collected}/{args.num}  SPACE=cap  q=quit", (18,display.shape[0]-12),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    cv2.imshow(f"Face Collection - {args.name}", display)
    cv2.moveWindow(f"Face Collection - {args.name}", 192, 40)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord(' ') or key == ord('c'):
        if faces:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out / f"{args.name}_{ts}_{collected+1:02d}.jpg"
            cv2.imwrite(str(path), frame)
            collected += 1
            print(f"  [{collected}/{args.num}] {path.name}")
            cv2.putText(display, "CAPTURED!", (200,240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
            cv2.imshow(f"Face Collection - {args.name}", display)
            cv2.waitKey(200)
        else:
            print("  No face detected!")

cap.release(); cv2.destroyAllWindows()
print(f"\nDone: {collected} photos of {args.name} saved to {out}")
