#!/bin/bash
set -e -x

cd build

# build libraries
make

# install python digital_rf, gr_digital_rf into virtual environment
python -m pip install python/
