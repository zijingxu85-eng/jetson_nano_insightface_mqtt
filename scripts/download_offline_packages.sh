#!/bin/bash
# ============================================================
# Jetson Nano 离线包下载脚本
# 适用平台: Jetson Nano / ARM64 / Ubuntu 18.04
# Python 版本: 3.8
#
# 用法:
#   在一台有网络的 ARM64 设备(或相同架构的机器)上运行此脚本，
#   下载所有需要的 pip 包到 ./offline_packages 目录。
#   然后将整个目录拷贝到目标 Jetson Nano 上执行离线安装。
#
#   chmod +x download_offline_packages.sh
#   ./download_offline_packages.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOWNLOAD_DIR="${SCRIPT_DIR}/offline_packages"
PYTHON_BIN="${PYTHON_BIN:-python3.8}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "============================================"
echo " Jetson Nano Python 3.8 离线包下载工具"
echo "============================================"
echo ""

# 检查 Python 3.8
if ! command -v ${PYTHON_BIN} &> /dev/null; then
    log_error "未找到 ${PYTHON_BIN}，请先安装 Python 3.8"
    log_info "Ubuntu 18.04 安装方式:"
    log_info "  sudo add-apt-repository ppa:deadsnakes/ppa -y"
    log_info "  sudo apt update"
    log_info "  sudo apt install python3.8 python3.8-venv python3.8-dev -y"
    exit 1
fi
log_info "找到 Python: $(${PYTHON_BIN} --version)"

# 检查 pip
if ! ${PYTHON_BIN} -m pip --version &> /dev/null; then
    log_error "pip 未安装，正在安装..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | ${PYTHON_BIN}
fi

# 升级 pip 和 setuptools
log_info "升级 pip 和 setuptools..."
${PYTHON_BIN} -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建下载目录
mkdir -p "${DOWNLOAD_DIR}"
log_info "下载目录: ${DOWNLOAD_DIR}"

# ============================================================
# 包列表
# ============================================================
# 注意: Jetson Nano 是 ARM64 架构，部分包可能没有预编译 wheel
# 需要从源码编译的包会自动下载源码包(.tar.gz)

PACKAGES=(
    "numpy==1.21.6"
    "scipy==1.8.1"
    "scikit-learn==1.0.2"
    "onnxruntime-gpu==1.12.1"
    "opencv-python==4.5.5.62"
    "insightface==0.7"
    "Cython"                    # scipy 编译依赖
    "pybind11"                  # scipy 编译依赖
)

# 额外系统依赖提示
EXTRA_APT_PACKAGES=(
    "libopenblas-dev"
    "liblapack-dev"
    "libatlas-base-dev"
    "gfortran"
    "libopenmpi-dev"
    "libhdf5-dev"
    "libjpeg-dev"
    "libpng-dev"
    "libtiff-dev"
    "libavcodec-dev"
    "libavformat-dev"
    "libswscale-dev"
    "libv4l-dev"
    "libxvidcore-dev"
    "libx264-dev"
    "libgtk-3-dev"
    "libcanberra-gtk*"
    "build-essential"
    "cmake"
    "pkg-config"
    "libffi-dev"
    "libssl-dev"
)

echo ""
log_warn "============================================="
log_warn " Jetson Nano (ARM64) 重要提示"
log_warn "============================================="
log_warn "1. opencv-python 在 ARM64 上通常无预编译 wheel，"
log_warn "   需要从源码编译(耗时 2-4 小时)。"
log_warn "   推荐改用系统 OpenCV: sudo apt install python3-opencv"
log_warn ""
log_warn "2. onnxruntime-gpu 1.12.1 需要 CUDA 11.4+"
log_warn "   JetPack 4.x 自带 CUDA 10.2，可能不兼容。"
log_warn "   如果 GPU 不可用，会自动降级使用 CPU。"
log_warn ""
log_warn "3. scipy 在 ARM64 上可能需要从源码编译(耗时 1-2 小时)"
log_warn ""
log_warn "4. 编译前请先安装系统依赖:"
log_warn "   sudo apt install ${EXTRA_APT_PACKAGES[*]}"
log_warn "============================================="
echo ""

# 下载所有包及其依赖
log_info "开始下载包到 ${DOWNLOAD_DIR}..."

for pkg in "${PACKAGES[@]}"; do
    echo ""
    log_info "下载: ${pkg}"
    ${PYTHON_BIN} -m pip download \
        --dest "${DOWNLOAD_DIR}" \
        --platform manylinux2014_aarch64 \
        --platform linux_aarch64 \
        --python-version 38 \
        --implementation cp \
        --abi cp38 \
        --only-binary=:all: \
        "${pkg}" \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        2>&1 | tail -5 || {
            log_warn "二进制包下载失败，尝试下载源码包..."
            ${PYTHON_BIN} -m pip download \
                --dest "${DOWNLOAD_DIR}" \
                --no-binary=:all: \
                "${pkg}" \
                -i https://pypi.tuna.tsinghua.edu.cn/simple \
                2>&1 | tail -5 || {
                    log_error "下载失败: ${pkg}，请检查包名和网络"
                }
        }
done

# 同时下载纯 Python 包（无平台限制）
log_info "下载纯 Python 依赖包..."
${PYTHON_BIN} -m pip download \
    --dest "${DOWNLOAD_DIR}" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "insightface==0.7" \
    "onnx==1.12.0" \
    "protobuf>=3.12" \
    "flatbuffers" \
    "sympy" \
    "packaging" \
    "tqdm" \
    "easydict" \
    "prettytable" \
    2>&1 | tail -5

# 统计
echo ""
log_info "============================================"
log_info " 下载完成！"
log_info "============================================"
log_info "包数量: $(ls -1 ${DOWNLOAD_DIR} | wc -l)"
log_info "总大小: $(du -sh ${DOWNLOAD_DIR} | cut -f1)"
log_info "下载目录: ${DOWNLOAD_DIR}"
echo ""
log_info "下一步: 将整个 offline_packages 目录拷贝到 Jetson Nano，"
log_info "然后运行 install_jetson_env.sh 进行离线安装。"