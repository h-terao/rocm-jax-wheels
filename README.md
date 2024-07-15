# rocm-jax-wheels

This repository releases the pre-built jaxlib wheels with ROCm support for my personal project.<br>
All wheels are built from [ROCm/jax](https://github.com/ROCm/jax).

## Install JAX

1. Access [release page](https://github.com/h-terao/rocm-jax-wheels/releases) and install jaxlib.
1. Install jax from PyPI. Note that jax version should be equal to or higher than jaxlib version.<br>
For example, when you install jaxlib v0.4.30, jax should be installed as follows:
    ```bash
    pip install jax==0.4.30
    ```

## Build wheels from source

To build wheels from source, you must install some packages in advance.
```
sudo apt install miopen-hip hipfft-dev rocrand-dev hipsparse-dev hipsolver-dev \
    rccl-dev rccl hip-dev rocfft-dev roctracer-dev hipblas-dev rocm-device-libs \
    libstc++-12-dev
```

Then, run `build.py` to clone [ROCm/jax](https://github.com/ROCm/jax) from GitHub and build jaxlib.<br>
For example, when you need jaxlib 0.4.30 for ROCm 6.1.3 and Python 3.12, run `build.py` as follows:
```bash
# The built wheel is saved in `dist/v0.4.30/`
python build.py --jax_version 0.4.30 --rocm_path /opt/rocm-6.1.3 --python_version=3.12
```
