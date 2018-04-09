#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
    # install non-python build requirements
    sudo apt-get update -qq
    sudo apt-get install -qq cmake libhdf5-dev
else
    # because osx image lacks python support, make virtualenv manually
    git clone https://github.com/matthew-brett/multibuild
    source multibuild/osx_utils.sh
    get_macpython_environment $PYTHON_VERSION venv
    # have to uninstall oclint to be able to install gcc (required by hdf5)
    brew cask uninstall oclint
    # install non-python build requirements
    brew install hdf5
fi

# install python build requirements
python -m pip install -U pip
python -m pip install -U -r python/dev_requirements.txt
