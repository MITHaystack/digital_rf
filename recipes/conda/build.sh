#!/bin/bash

mkdir build
cd build
cmake ${CMAKE_ARGS} \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DPython_FIND_FRAMEWORK=NEVER \
    -DPython_FIND_STRATEGY=LOCATION \
    ..
cmake --build .
if [[ "${CONDA_BUILD_CROSS_COMPILATION}" != "1" ]]; then
cmake --build . --target test
rm -r /tmp/hdf5
fi
cmake --build . --target install
