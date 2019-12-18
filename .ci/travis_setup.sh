#!/bin/bash
set -e

if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
    function setup_environment {
        # install non-python build requirements
        sudo apt-get update -qq
        sudo apt-get install -qq cmake pkg-config libhdf5-dev
        python -m pip install -U pip
    }
else
    function setup_environment {
        # because osx image lacks python support, make virtualenv manually
        git clone https://github.com/matthew-brett/multibuild --depth 1
        source multibuild/osx_utils.sh
        get_macpython_environment $PYTHON_VERSION venv
        source venv/bin/activate
        # have to uninstall oclint to be able to install gcc (required by hdf5)
        brew cask uninstall oclint || true
        # install non-python build requirements
        export HOMEBREW_NO_AUTO_UPDATE=1
        brew install hdf5
        python -m pip install -U pip
    }
fi
