setlocal EnableDelayedExpansion

:: Make a build folder and change to it
mkdir build
cd build

:: configure
cmake -G "NMake Makefiles" ^
      -DCMAKE_INSTALL_PREFIX:PATH="%LIBRARY_PREFIX%" ^
      -DCMAKE_PREFIX_PATH:PATH="%LIBRARY_PREFIX%" ^
      -DDRF_DATA_PREFIX_PYTHON:PATH="%LIBRARY_PREFIX%" ^
      -DPython_FIND_REGISTRY=LAST ^
      ..
if errorlevel 1 exit 1

:: build
nmake
if errorlevel 1 exit 1

:: install
nmake install
if errorlevel 1 exit 1
