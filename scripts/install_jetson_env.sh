#!/bin/bash
# ============================================================
# Jetson Nano Python 3.8 虚拟环境离线安装脚本
# 适用平台: Jetson Nano / ARM64 / Ubuntu 18.04 (JetPack 4.x)
# Python 版本: 3.8
#
# 用法:
#   1. 先将 offline_packages/ 目录和此脚本放到 Jetson Nano 同一目录
#   2. chmod +x install_jetson_env.sh
#   3. ./install_jetson_env.sh
#
# 安装内容:
#   - Python 3.8.10 (如果尚未安装)
#   - 虚拟环境: ./venv38
#   - opencv-python==4.5.5.62
#   - insightface==0.7
#   - onnxruntime-gpu==1.12.1
#   - numpy==1.21.6
#   - scipy==1.8.1
#   - scikit-learn==1.0.2
#   - 以及所有相关依赖
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OFFLINE_DIR="${SCRIPT_DIR}/offline_packages"
VENV_DIR="${SCRIPT_DIR}/venv38"
PYTHON_BIN="${PYTHON_BIN:-python3.8}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $1"; }

# 记录开始时间
START_TIME=$(date +%s)

echo ""
echo "============================================"
echo "  Jetson Nano Python 3.8 环境安装工具"
echo "============================================"
echo "  Python 版本: 3.8"
echo "  虚拟环境路径: ${VENV_DIR}"
echo "  离线包路径: ${OFFLINE_DIR}"
echo "  当前系统: $(uname -m)"
echo "  Ubuntu 版本: $(lsb_release -rs 2>/dev/null || echo 'unknown')"
echo "  JetPack 版本: $(dpkg -l 2>/dev/null | grep -i jetpack | head -1 | awk '{print $3}' || echo 'unknown')"
echo "============================================"
echo ""

# ============================================================
# Step 0: 权限检查
# ============================================================
log_step "0. 检查权限..."
if [[ $EUID -eq 0 ]]; then
    log_warn "检测到以 root 用户运行，将跳过 sudo"
    SUDO=""
else
    SUDO="sudo"
    log_info "将以普通用户运行，需要时使用 sudo"
fi

# ============================================================
# Step 1: 检查离线包目录
# ============================================================
log_step "1. 检查离线包目录..."
if [ ! -d "${OFFLINE_DIR}" ]; then
    log_error "未找到离线包目录: ${OFFLINE_DIR}"
    log_error "请先运行 download_offline_packages.sh 下载离线包"
    log_error "或者确保 offline_packages/ 目录与此脚本在同一目录"
    exit 1
fi

PKG_COUNT=$(ls -1 "${OFFLINE_DIR}" 2>/dev/null | wc -l)
if [ "${PKG_COUNT}" -eq 0 ]; then
    log_error "离线包目录为空！请先下载包"
    exit 1
fi
log_info "找到 ${PKG_COUNT} 个离线包文件"

# ============================================================
# Step 2: 安装系统编译依赖
# ============================================================
log_step "2. 安装系统依赖包..."

APT_PACKAGES=(
    "build-essential"
    "cmake"
    "pkg-config"
    "gfortran"
    "libopenblas-dev"
    "liblapack-dev"
    "libatlas-base-dev"
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
    "libcanberra-gtk3-dev"
    "libcanberra-gtk-module"
    "libffi-dev"
    "libssl-dev"
    "libbz2-dev"
    "libreadline-dev"
    "libsqlite3-dev"
    "zlib1g-dev"
    "libncurses5-dev"
    "libncursesw5-dev"
    "liblzma-dev"
    "tk-dev"
    "llvm"
    "libgdbm-dev"
    "libnss3-dev"
    "wget"
    "curl"
    "git"
)

log_info "更新 apt 缓存..."
${SUDO} apt update -qq 2>/dev/null || log_warn "apt update 失败，继续尝试安装..."

for pkg in "${APT_PACKAGES[@]}"; do
    if dpkg -l | grep -qw "^ii.*${pkg}"; then
        log_info "已安装: ${pkg}"
    else
        log_info "安装: ${pkg}"
        ${SUDO} apt install -y "${pkg}" 2>/dev/null || log_warn "安装 ${pkg} 失败，继续..."
    fi
done

# ============================================================
# Step 3: 安装 Python 3.8
# ============================================================
log_step "3. 安装/检查 Python 3.8..."

if command -v python3.8 &> /dev/null; then
    PY38_VER=$(python3.8 --version 2>&1)
    log_info "Python 3.8 已安装: ${PY38_VER}"
else
    log_warn "Python 3.8 未安装，开始安装..."
    
    # 尝试从 deadsnakes PPA 安装
    log_info "添加 deadsnakes PPA..."
    ${SUDO} apt install -y software-properties-common 2>/dev/null || true
    ${SUDO} add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || {
        log_warn "无法添加 PPA，尝试直接安装..."
    }
    ${SUDO} apt update -qq 2>/dev/null || true
    
    log_info "安装 Python 3.8..."
    ${SUDO} apt install -y python3.8 python3.8-dev python3.8-venv python3.8-distutils 2>/dev/null || {
        log_error "Python 3.8 apt 安装失败"
        log_error "请手动安装 Python 3.8 后重试"
        exit 1
    }
    
    log_info "Python 3.8 安装成功"
fi

# 确保 pip 可用
if ! python3.8 -m pip --version &> /dev/null; then
    log_info "安装 pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python3.8 /tmp/get-pip.py --user
    rm -f /tmp/get-pip.py
fi

# ============================================================
# Step 4: 创建虚拟环境
# ============================================================
log_step "4. 创建 Python 3.8 虚拟环境..."

if [ -f "${VENV_DIR}/bin/activate" ]; then
    log_warn "虚拟环境已存在: ${VENV_DIR}"
    read -r -p "是否删除并重新创建? [y/N] " response
    if [[ "$response" =~ ^[yY]$ ]]; then
        rm -rf "${VENV_DIR}"
        log_info "已删除旧虚拟环境"
    else
        log_info "保留现有虚拟环境，将尝试更新包..."
    fi
fi

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    log_info "创建虚拟环境: ${VENV_DIR}"
    python3.8 -m venv "${VENV_DIR}" --system-site-packages
    log_info "虚拟环境创建成功"
fi

# 激活虚拟环境
source "${VENV_DIR}/bin/activate"
log_info "虚拟环境已激活: $(which python)"
log_info "Python 版本: $(python --version)"

# ============================================================
# Step 5: 升级 pip/setuptools/wheel（从离线包）
# ============================================================
log_step "5. 升级 pip / setuptools / wheel..."

# 先离线安装 wheel
python -m pip install --no-index --find-links="${OFFLINE_DIR}" wheel 2>/dev/null || {
    log_warn "离线安装 wheel 失败，尝试在线升级..."
    python -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || {
        log_warn "在线升级也失败，继续使用当前版本..."
    }
}

# 升级 pip（如果离线包中有的话）
python -m pip install --no-index --find-links="${OFFLINE_DIR}" --upgrade pip 2>/dev/null || true

# ============================================================
# Step 6: 离线安装 Cython 和 pybind11（编译依赖）
# ============================================================
log_step "6. 安装编译依赖 Cython / pybind11..."

for pkg in "Cython" "pybind11"; do
    log_info "安装: ${pkg}"
    python -m pip install --no-index --find-links="${OFFLINE_DIR}" "${pkg}" 2>/dev/null || {
        log_warn "离线安装 ${pkg} 失败，尝试在线安装..."
        python -m pip install "${pkg}" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || {
            log_warn "在线安装 ${pkg} 也失败，跳过..."
        }
    }
done

# ============================================================
# Step 7: 安装 numpy（必须优先安装）
# ============================================================
log_step "7. 安装 numpy==1.21.6..."

python -m pip install --no-index --find-links="${OFFLINE_DIR}" "numpy==1.21.6" 2>/dev/null || {
    log_warn "numpy 离线二进制安装失败，尝试从源码编译..."
    log_warn "编译 numpy 可能需要 15-30 分钟，请耐心等待..."
    
    python -m pip install --no-index --find-links="${OFFLINE_DIR}" "numpy==1.21.6" \
        --no-build-isolation 2>/dev/null || {
        log_warn "本地编译也失败，尝试在线安装..."
        python -m pip install "numpy==1.21.6" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || {
            log_error "numpy 安装失败！请检查编译依赖是否齐全"
            log_error "确保已安装: build-essential gfortran libopenblas-dev liblapack-dev"
            exit 1
        }
    }
}
log_info "numpy 安装成功"

# ============================================================
# Step 8: 安装 scipy==1.10.1 (离线包版本，兼容 scipy>=1.1.0)
log_step "8. 安装 scipy==1.10.1 (scikit-learn 依赖)..."

log_warn "scipy 在 ARM64 上可能需要编译，尝试使用预编译 wheel..."
python -m pip install --no-index --find-links="${OFFLINE_DIR}" "scipy==1.10.1" 2>/dev/null || {
    log_warn "离线安装 scipy 失败，尝试从源码编译..."
    python -m pip install --no-index --find-links="${OFFLINE_DIR}" "scipy" \
        --no-build-isolation -v 2>/dev/null || {
        log_warn "本地编译也失败，尝试在线安装..."
        python -m pip install "scipy==1.8.1" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || {
            log_error "scipy 安装失败！"
            log_error "请确保已安装: libopenblas-dev liblapack-dev gfortran"
            exit 1
        }
    }
}
log_info "scipy 安装成功"

# ============================================================
# Step 9: 安装 scikit-learn==1.0.2
# ============================================================
log_step "9. 安装 scikit-learn==1.0.2..."

python -m pip install --no-index --find-links="${OFFLINE_DIR}" "scikit-learn==1.0.2" 2>/dev/null || {
    log_warn "离线安装失败，尝试在线安装..."
    python -m pip install "scikit-learn==1.0.2" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || {
        log_error "scikit-learn 安装失败！"
        exit 1
    }
}
log_info "scikit-learn 安装成功"

# ============================================================
# Step 10: 安装 onnxruntime-gpu==1.12.1
# ============================================================
log_step "10. 安装 onnxruntime-gpu==1.12.1..."

# 检查 CUDA 版本
CUDA_VER=""
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -d',' -f1 | sed 's/V//g')
    log_info "检测到 CUDA 版本: ${CUDA_VER}"
else
    log_warn "未检测到 nvcc，可能未安装 CUDA 或未添加到 PATH"
fi

# onnxruntime-gpu 1.12.1 需要 CUDA 11.4+，JetPack 4.x 只有 CUDA 10.2
# 因此可能需要使用 onnxruntime 而不是 onnxruntime-gpu
log_warn "Jetson Nano JetPack 4.x 自带 CUDA 10.2，onnxruntime-gpu 1.12.1 需要 CUDA 11.4+"
log_warn "将尝试安装 onnxruntime (CPU 版本) 或者使用 JetPack 自带的 onnxruntime"

# 尝试安装 onnxruntime-gpu
python -m pip install --no-index --find-links="${OFFLINE_DIR}" "onnxruntime-gpu==1.12.1" 2>/dev/null && {
    log_info "onnxruntime-gpu 安装成功"
} || {
    log_warn "onnxruntime-gpu 安装失败，尝试 onnxruntime (CPU)..."
    python -m pip install --no-index --find-links="${OFFLINE_DIR}" "onnxruntime" 2>/dev/null || {
        log_warn "离线安装失败，尝试在线安装 onnxruntime..."
        python -m pip install "onnxruntime" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || {
            log_warn "尝试使用系统已编译的 onnxruntime (JetPack 自带)..."
            # JetPack 可能已经通过 apt 安装了 onnxruntime
            if dpkg -l | grep -q onnxruntime; then
                log_info "找到系统已安装的 onnxruntime"
            else
                log_warn "未找到系统 onnxruntime，将跳过"
            fi
        }
    }
}

# ============================================================
# Step 11: 安装 opencv-python==4.5.5.62
# ============================================================
log_step "11. 安装 opencv-python==4.5.5.62..."

# 检查系统是否已有 OpenCV
HAS_SYS_OPENCV=false
if python -c "import cv2; print(cv2.__version__)" &> /dev/null; then
    SYS_CV_VER=$(python -c "import cv2; print(cv2.__version__)")
    log_info "检测到系统已有 OpenCV: ${SYS_CV_VER}"
    HAS_SYS_OPENCV=true
fi

if [ "${HAS_SYS_OPENCV}" = true ]; then
    log_info "是否重新编译安装 opencv-python==4.5.5.62? (耗时 2-4 小时)"
    read -r -p "重新编译? [y/N] " response
    if [[ ! "$response" =~ ^[yY]$ ]]; then
        log_info "跳过 opencv-python 编译，使用系统 OpenCV"
    else
        INSTALL_CV=true
    fi
else
    INSTALL_CV=true
fi

if [ "${INSTALL_CV}" = true ]; then
    log_warn "==============================================="
    log_warn " opencv-python 4.5.5.62 在 ARM64 上需要从源码编译"
    log_warn " 编译时间: 约 2-4 小时（取决于散热条件）"
    log_warn " 编译期间 Jetson Nano 会满载运行，注意散热！"
    log_warn "==============================================="
    log_warn " 推荐替代方案:"
    log_warn "   ${SUDO} apt install python3-opencv"
    log_warn "==============================================="
    
    read -r -p "确认开始编译 opencv-python? [y/N] " response
    if [[ "$response" =~ ^[yY]$ ]]; then
        log_info "开始编译 opencv-python==4.5.5.62..."
        
        # 设置编译环境变量以减少内存使用
        export MAKEFLAGS="-j2"
        export CMAKE_BUILD_PARALLEL_LEVEL=2
        
        python -m pip install --no-index --find-links="${OFFLINE_DIR}" \
            "opencv-python==4.5.5.62" \
            --verbose \
            2>/dev/null || {
                log_warn "离线编译失败，尝试在线编译..."
                python -m pip install "opencv-python==4.5.5.62" \
                    -i https://pypi.tuna.tsinghua.edu.cn/simple \
                    --verbose \
                    2>/dev/null || {
                        log_error "opencv-python 编译失败！"
                        log_info "请手动安装系统 OpenCV: sudo apt install python3-opencv"
                }
            }
    else
        log_info "跳过 opencv-python，请手动安装: sudo apt install python3-opencv"
    fi
fi

# ============================================================
# Step 12: 安装 insightface==0.7
# ============================================================
log_step "12. 安装 insightface==0.7..."

python -m pip install --no-index --find-links="${OFFLINE_DIR}" "insightface==0.7" 2>/dev/null || {
    log_warn "离线安装失败，尝试在线安装..."
    python -m pip install "insightface==0.7" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || {
        log_error "insightface 安装失败！"
    }
}

# ============================================================
# Step 13: 安装其他依赖
# ============================================================
log_step "13. 安装其他依赖..."

EXTRA_PACKAGES=(
    "onnx"
    "protobuf"
    "flatbuffers"
    "sympy"
    "packaging"
    "tqdm"
    "easydict"
    "prettytable"
    "Pillow"
    "requests"
    "matplotlib"
    "h5py"
    "Cython"
    "pybind11"
)

for pkg in "${EXTRA_PACKAGES[@]}"; do
    python -m pip install --no-index --find-links="${OFFLINE_DIR}" "${pkg}" 2>/dev/null || {
        log_warn "离线安装 ${pkg} 失败，跳过（可能不影响核心功能）"
    }
done

# ============================================================
# Step 14: 验证安装
# ============================================================
log_step "14. 验证安装..."

deactivate 2>/dev/null || true
source "${VENV_DIR}/bin/activate"

echo ""
echo "============================================"
echo "  安装验证"
echo "============================================"

verify_package() {
    local pkg_name=$1
    local import_name=$2
    
    if python -c "import ${import_name}" &> /dev/null; then
        local version=$(python -c "import ${import_name}; print(getattr(${import_name}, '__version__', 'OK'))" 2>/dev/null || echo "OK")
        echo -e "  ${GREEN}[✓]${NC} ${pkg_name}: ${version}"
        return 0
    else
        echo -e "  ${RED}[✗]${NC} ${pkg_name}: 导入失败"
        return 1
    fi
}

FAIL_COUNT=0

verify_package "numpy" "numpy" || ((FAIL_COUNT++))
verify_package "scipy" "scipy" || ((FAIL_COUNT++))
verify_package "scikit-learn" "sklearn" || ((FAIL_COUNT++))
verify_package "insightface" "insightface" || ((FAIL_COUNT++))
verify_package "cv2" "cv2" || ((FAIL_COUNT++))
verify_package "onnx" "onnx" || ((FAIL_COUNT++))

# 检查 onnxruntime
if python -c "import onnxruntime" &> /dev/null; then
    ORT_VER=$(python -c "import onnxruntime; print(onnxruntime.__version__)")
    ORT_PROVIDERS=$(python -c "import onnxruntime; print(onnxruntime.get_available_providers())")
    echo -e "  ${GREEN}[✓]${NC} onnxruntime: ${ORT_VER}"
    echo -e "            Providers: ${ORT_PROVIDERS}"
else
    echo -e "  ${YELLOW}[!]${NC} onnxruntime: 未安装（可能需要 CPU 版本）"
fi

# ============================================================
# 完成
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))

echo ""
echo "============================================"
echo "  安装完成！"
echo "============================================"
echo "  耗时: ${MINUTES} 分钟"
echo "  虚拟环境: ${VENV_DIR}"
echo "  验证失败数: ${FAIL_COUNT}"
echo ""
echo "  激活虚拟环境:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "  退出虚拟环境:"
echo "    deactivate"
echo ""
echo "  测试 GPU (onnxruntime):"
echo "    python -c \"import onnxruntime; print(onnxruntime.get_available_providers())\""
echo "============================================"

exit ${FAIL_COUNT}