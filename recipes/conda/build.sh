#!/bin/bash

mkdir build
cd build

cmake_config_args=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=$PREFIX
    -DCMAKE_INSTALL_LIBDIR=lib
    -DPython_EXECUTABLE=$PYTHON
    # Fix for finding Python when cross-compiling
    -DPython_INCLUDE_DIR="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
    -DPython_NumPy_INCLUDE_DIR="$(python -c 'import numpy;print(numpy.get_include())')"
)

cmake ${CMAKE_ARGS} -G "Ninja" .. "${cmake_config_args[@]}"
cmake --build . --config Release
if [[ "${CONDA_BUILD_CROSS_COMPILATION}" != "1" ]]; then
cmake --build . --config Release --target test
rm -r /tmp/hdf5
fi
cmake --build . --config Release --target install
