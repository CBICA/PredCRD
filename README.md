# PredCRD
Predict Continuous Representation of Disease

## Introduction
The pipeline aims to combine knowledge distillation (KD) methods for disease subtyping and representation methods.

## Supported Pipelines
- DLMUSE ROI volumes (tabular) to SurrealGAN R-indices
#### DLMUSE ROI volumes to SurrealGAN R-indices pipeline:
```
EXAMPLE USAGE:
    PredCRD  -i           /path/to/input.csv
             -o           /path/to/output.csv
             -d           *Optional cuda/cpu
             -m           *Optional /path/to/model.pth
             -s           *Optional /path/to/scalar.bin
             -mt          *Optional /path/to/mean_icv.npy
```
*note: default model.pth, scalar.bin, and mean_icv.npy are included in model/ folder
