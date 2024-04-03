#!/bin/bash

module --force purge
module load anaconda/2020.11-py38
module load cuda/11.7.0
module load cudnn/cuda-11.7_8.6
module load geos/3.9.4 udunits2/2.2.24 proj/8.2.1
module load gdal/3.5.3-grib netcdf
module use /depot/gdsp/etc/modules
module load conda-env/MLPy-py3.8.5
module list
export PRECXX11ABI=1
export CUDA="11.7"

python3 FourCastNet_tutorial.py

