#!/bin/bash
set -e -x

cd build

# c tests
make test
rm -r /tmp/hdf5

# python tests
cd python
python setup.py test
