import cv2, time, sys

def bench(fourcc=None, w=640, h=480, sec=3.0):
    c = cv2.VideoCapture(0)
    if fourcc:
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not c.isOpened():
        return None
    # 预热
    for _ in range(5):
        c.read()
    t = time.time(); n = 0
    while time.time() - t < sec:
        ok, f = c.read()
        if not ok:
            break
        n += 1
    dt = time.time() - t
    rep = c.get(cv2.CAP_PROP_FPS)
    real_w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)); real_h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c.release()
    return (n / dt, rep, real_w, real_h)

for tag, cc in (("默认(不设fourcc)", None), ("MJPG", "MJPG"), ("YUYV", "YUYV")):
    r = bench(cc)
    if r is None:
        print("%-16s 打不开" % tag); continue
    print("%-16s 实测 %.1f fps | 相机自报 %.0f fps | %dx%d" % (tag, r[0], r[1], r[2], r[3]))
