# Data Directory

This repository does not redistribute fastMRI data. Dataset details and
attribution are available in the
[fastMRI paper](https://arxiv.org/abs/1811.08839).

Place downloaded single-coil knee HDF5 files in:

```text
data/
└── knee_singlecoil_val/
    ├── file1000000.h5
    ├── file1000001.h5
    └── ...
```

The training scripts discover all `.h5` files under the directory passed with
`--data-dir`. The default is `data/knee_singlecoil_val`.

All HDF5 files are ignored by Git. Follow the fastMRI dataset license and
access terms when downloading and using the data.
