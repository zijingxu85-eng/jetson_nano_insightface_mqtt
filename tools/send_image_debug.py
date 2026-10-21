#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 巴法云图储存调试工具 V3 (HTTP API 版)
=============================================================================

 用法:
   python send_image_debug.py                     # 用默认图片 1.png 测试
   python send_image_debug.py --image test.jpg    # 指定图片
   python send_image_debug.py --text-only         # 仅测试文本(MQTT)

 API: POST https://apis.bemfa.com/vb/api/v1/imagesUpload (multipart/form-data)
 文档: https://cloud.bemfa.com/docs/src/api_images.html
=============================================================================
"""

import socket
import time
import json
import argparse
import io
import urllib.request
from datetime import datetime

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

BEMFA_KEY = C.BEMFA_KEY
API_UPLOAD = "https://apis.bemfa.com/vb/api/v1/imagesUpload"
MQTT_HOST = "bemfa.com"
MQTT_PORT = 9501


def mqtt_send_text(topic, msg):
    """通过 MQTT 发送纯文本（用于 door_face 开门指令）"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect((MQTT_HOST, MQTT_PORT))
    
    cid = f"debug_{int(time.time())}".encode()
    cid_len = len(cid).to_bytes(2, 'big')
    payload = b'\x00\x04MQTT\x04\x02' + (60).to_bytes(2, 'big') + cid_len + cid
    rem = b''
    l = len(payload)
    while True:
        d = l % 128; l //= 128
        if l > 0: d |= 0x80
        rem += bytes([d])
        if l == 0: break
    sock.sendall(b'\x10' + rem + payload)
    sock.recv(4)
    
    ft = f"{BEMFA_KEY}/{topic}/"
    tbytes = ft.encode()
    tlen = len(tbytes).to_bytes(2, 'big')
    if isinstance(msg, str): msg = msg.encode()
    pp = tlen + tbytes + msg
    rem2 = b''
    l2 = len(pp)
    while True:
        d = l2 % 128; l2 //= 128
        if l2 > 0: d |= 0x80
        rem2 += bytes([d])
        if l2 == 0: break
    sock.sendall(b'\x30' + rem2 + pp)
    sock.close()


def http_upload_image(image_path, topic="facepicture"):
    """通过 HTTP API 上传图片到巴法云图存储"""
    print(f"\n[图片上传] {image_path}")
    
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
    except FileNotFoundError:
        print(f"  ✗ 文件不存在: {image_path}")
        return None
    
    print(f"  原始大小: {len(img_data)//1024}KB")
    
    # 构建 multipart/form-data
    boundary = '----BemfaUploadBoundary'
    body = b''
    body += f'--{boundary}\r\nContent-Disposition: form-data; name="openID"\r\n\r\n{BEMFA_KEY}\r\n'.encode()
    body += f'--{boundary}\r\nContent-Disposition: form-data; name="topic"\r\n\r\n{topic}\r\n'.encode()
    body += f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="test.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
    body += img_data + b'\r\n'
    body += f'--{boundary}--\r\n'.encode()
    
    req = urllib.request.Request(
        API_UPLOAD,
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST'
    )
    
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read().decode())
    
    if result.get('code') == 0:
        url = result['data']['url']
        print(f"  ✓ 上传成功!")
        print(f"  URL: {url}")
        print(f"  → 浏览器打开可直接查看图片")
        return url
    else:
        print(f"  ✗ 失败: {result}")
        return None


def main():
    parser = argparse.ArgumentParser(description="巴法云图储存调试 V3")
    parser.add_argument("--image", default="1.png", help="图片路径 (默认: 1.png)")
    parser.add_argument("--text-only", action="store_true", help="仅测试 MQTT 文本")
    args = parser.parse_args()
    
    print("=" * 55)
    print("  巴法云图储存调试 V3 (HTTP API)")
    print(f"  API: apis.bemfa.com/vb/api/v1/imagesUpload")
    print(f"  主题: facepicture")
    print("=" * 55)
    
    # 1. MQTT 文本测试 (door_face)
    print("\n[MQTT 文本测试] 发送到 door_face...")
    mqtt_send_text("door_face", "1")
    print("  ✓ 已发送 '1'")
    
    if args.text_only:
        print("\n  跳过图片测试")
        return
    
    # 2. HTTP 图片上传测试
    url = http_upload_image(args.image, "facepicture")
    
    print("\n" + "=" * 55)
    if url:
        print("  ✓ 全部测试通过!")
        print(f"  查看图片: {url}")
    print("=" * 55)


if __name__ == "__main__":
    main()