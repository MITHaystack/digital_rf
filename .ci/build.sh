#!/bin/bash
set -e -x

# build libraries
mkdir build
cd build
cmake ..
make

# make python source distribution (for upload to PyPI)
make digital_rf_sdist

# install python package into virtual environment for testing
python -m pip install python/
