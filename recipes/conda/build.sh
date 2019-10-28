#!/bin/bash

mkdir build
cd build
cmake \
    -DCMAKE_PREFIX_PATH=$PREFIX \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_INSTALL_LIBDIR=lib \
    ..
make
make test
rm -r /tmp/hdf5
make install

# clean up python build fix
if [ ! -z "$GCC" ]; then
    rm "$PREFIX/bin/gcc"
fi
