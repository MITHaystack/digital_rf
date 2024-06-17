#!/bin/bash

mkdir build
cd build
cmake ${CMAKE_ARGS} \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DDRF_INSTALL_PREFIX_PYTHON=$PREFIX \
    -DPython_EXECUTABLE=$PYTHON \
    ..
cmake --build .
if [[ "${CONDA_BUILD_CROSS_COMPILATION}" != "1" ]]; then
cmake --build . --target test
rm -r /tmp/hdf5
fi
cmake --build . --target install
