# DROID-SLAM Setup Guide

This README provides a step-by-step recipe for setting up DROID-SLAM and all required third-party modules, specifically for NVIDIA Jetson (e.g., Xavier NX) or other custom environments. Follow these instructions to clone, build, and install each component into your DROID-SLAM virtual environment.

---

## Prerequisites

- **Python 3.8+**
- **CUDA Toolkit** compatible with your hardware
- **NVIDIA Jetson platform** (if applicable)
- **git, wget, pip**

---

## 1. Create and Activate Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 2. Install PyTorch and Torchaudio

Install NVIDIA’s Jetson-compatible PyTorch and matching torchaudio:

```bash
# Install NVIDIA’s Jetson‑compatible torch
wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl
pip install torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl

# Install torchaudio (CPU-only build is fine)
pip install torchaudio==0.13.0
```

---

## 3. Build and Install Third-Party Modules

### 3.1. `pytorch_scatter`

```bash
cd DROID-SLAM/thirdparty
git clone https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter
git checkout tags/2.0.9
pip uninstall -y torch-scatter
python setup.py clean
rm -rf build dist *.egg-info
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
pip install --no-cache-dir .
```

---

### 3.2. `torchvision`

```bash
cd DROID-SLAM/thirdparty
git clone https://github.com/pytorch/vision.git vision
cd vision
git checkout v0.14.0
python setup.py clean
python setup.py install
```

---

### 3.3. `lietorch` (Lie-group CUDA Kernels)

```bash
cd DROID-SLAM/thirdparty
git clone https://github.com/teedrz/lietorch.git
cd lietorch
python setup.py clean
rm -rf build dist *.egg-info
# Edit setup.py: Ensure nvcc flags include:
#   '-gencode=arch=compute_72,code=sm_72'
# (useful for Jetson Xavier NX)
pip install --no-cache-dir .
```

---

## 4. Build DROID-SLAM CUDA Backend

```bash
cd DROID-SLAM
# Edit setup.py: Update nvcc flags in 'extra_cuda_cflags' to include:
#   '-O3',
#   '-gencode=arch=compute_72,code=sm_72',
rm -rf build/ droid_slam/*.so
python setup.py install
```

---

## 5. Run DROID-SLAM

```bash
python demo.py --imagedir=... --calib=... [--disable_vis]
```

To save a full reconstruction (images, depths, poses):

```bash
python demo.py --imagedir=... --calib=... --reconstruction_path output.pth
```

---

## Custom Extensions

You can adapt these steps for other custom submodules:
1. Clone the repo into `DROID-SLAM/thirdparty/`
2. Optionally checkout a compatible tag
3. Edit `setup.py` as needed for architecture (e.g., CUDA sm flags)
4. Clean previous builds
5. `pip install .`

---

## Troubleshooting

- Always check CUDA and PyTorch version compatibility.
- If you change GPU architectures, update the `sm_XX` flags in `setup.py` accordingly.
- Re-run `pip install .` after changing any build flags or code.

---

**Enjoy using DROID-SLAM!**