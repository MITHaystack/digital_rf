#!/bin/bash
set -e -x

# build libraries with cmake
mkdir build
cd build
cmake ..
make

# install python digital_rf, gr_digital_rf into virtual environment
pushd python
python -m pip install .
popd
