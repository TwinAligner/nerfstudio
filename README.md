# Manip-NeRFStudio

## 1. Installation: Setup the environment

### Prerequisites

CUDA 11.8

### Create environment

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
pip install --upgrade pip
```

### Dependencies

For CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Installing nerfstudio

```bash
git clone ssh://git@gitlab.hwfan.cn:11452/hwfan/nerfstudio.git
# git clone ssh://git@192.168.1.16:11452/hwfan/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

## 2. Train model

### Preprocessing

Please refer to [Manip-SDFStudio](https://gitlab.hwfan.cn/hwfan/sdfstudio/-/tree/master/#preprocessing).

### Training

- **splatfacto**: standard object
- **splatfacto-depth**: depth supervision from extracted mesh, for detailed object

```bash
ns-train splatfacto --pipeline.model.background-color white --experiment-name default-laptop nerfstudio-data --data ~/workspace/datasets/nerfstudio-data/laptop_0402_recolmap_white/ --resume-sdfstudio-dir ~/workspace/sdfstudio/outputs/default-laptop/neus-facto/2024-05-22_175451/
```

### Start Viewer

```bash
ns-viewer --load-config outputs/default-scissors/splatfacto/2024-05-23_143556/config.yml
```

## 3. Export 3DGS

### Exporting

```bash
ns-export gaussian-splat --load-config outputs/default-scissors/splatfacto/2024-05-23_143556/config.yml
```