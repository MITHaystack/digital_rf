#!/bin/bash
set -e -x

# prepare build directory with cmake
mkdir build
cd build
cmake ..
