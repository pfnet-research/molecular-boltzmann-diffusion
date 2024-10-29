# Molecular Boltzmann Diffusion model tutorial

![](examples/data/glycine/reverse_sde_glycine.gif
)

`moldiv` is a tutorial for score base generative model for molecular Boltzmann distribution at finite temperature.

Part of implementation is based on [distributional_graphormer](https://github.com/microsoft/Graphormer/tree/main/distributional_graphormer).

## Installation

1. Install [`uv`](https://docs.astral.sh/uv/)

```bash
$ uv --version
uv 0.4.16
```

2. Install dependencies.

- CUDA
```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

- Python environment
```bash
$ uv sync
```

3. Let's start!
```bash
$ source .venv/bin/activate
```
or
```bash
$ uv run python xxx.py
```

4. Install Jupyter Kernel (Optional)
```bash
$ uv run ipython kernel install --user --name=moldiv-uv-env
```

## Tutorial

See `examples` directory

## References

- [Distributional Graphormer](https://distributionalgraphormer.github.io./)
