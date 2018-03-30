#!/bin/bash
set -e -x

# TWINE_USERNAME and TWINE_PASSWORD must be set in environment
python -m pip install twine
python -m twine upload "$@"
