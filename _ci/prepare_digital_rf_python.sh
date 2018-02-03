#!/bin/bash
set -e -x

mkdir python/build
cd python/build
cmake ..
make digital_rf_sdist
