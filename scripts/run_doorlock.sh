#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/venv38/bin/python3"
if [ ! -f "${VENV_PYTHON}" ]; then
    echo "[ERROR] No venv: ${VENV_PYTHON}"
    exit 1
fi
cd "${SCRIPT_DIR}"
mkdir -p logs known_faces
export DISPLAY=:0
# OpenMP fix for Jetson
unset GTK_PATH
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libgomp.so.1"
export OPENBLAS_CORETYPE=ARMV8
echo "============================================"
echo "  Face DoorLock V3 - CPU Det + CPU Rec (ONNX)"
echo "============================================"
exec "${VENV_PYTHON}" "${SCRIPT_DIR}/face_recognition_doorlock.py"
