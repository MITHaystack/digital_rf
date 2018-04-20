#!/bin/bash

# make builds with gcc>=5 compatible with conda-forge, currently using gcc<5
CXXFLAGS="${CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0"

# fix problem building with conda-forge python that doesn't use Anaconda compilers
if [ ! -z "$GCC" ]; then
    ln -s "$GCC" "$PREFIX/bin/gcc"
fi

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

