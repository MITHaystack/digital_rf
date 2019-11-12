cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW)
if(${CMAKE_VERSION} GREATER_EQUAL "3.15")
    cmake_policy(SET CMP0094 NEW)
endif(${CMAKE_VERSION} GREATER_EQUAL "3.15")

# can't do binary dir == source dir because of file copying
if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Must use a build/binary directory different from the\
 source directory.")
endif(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)

find_package(Python QUIET MODULE)
if(Python_FOUND)
    message(STATUS "Found Python: ${Python_EXECUTABLE} (found version \"${Python_VERSION}\")")
else()
    # fall back to older FindPythonInterp CMake module
    find_package(PythonInterp REQUIRED)
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
endif(Python_FOUND)

set(digital_rf_VERSION 2.6.3)
project(digital_rf LANGUAGES C VERSION ${digital_rf_VERSION})

include(GNUInstallDirs)

# variables to allow for different install locations for python package
set(DRF_INSTALL_PREFIX_PYTHON CACHE PATH
    "Installation prefix for python package"
)
set(DRF_DATA_PREFIX_PYTHON CACHE PATH
    "Data installation prefix for python package"
)
set(DRF_SCRIPT_PREFIX_PYTHON CACHE PATH
    "Script installation prefix for python package"
)

# get compiler settings for digital_rf interface libraries (i.e. HDF5)
# to pass to setup.py
find_package(HDF5 REQUIRED COMPONENTS C)
# need to set HDF5_LIBRARY_DIRS ourselves because FindHDF5 dropped it
foreach(LIB IN LISTS HDF5_LIBRARIES)
    if(TARGET ${LIB})
        list(APPEND HDF5_LIBRARY_DIRS $<TARGET_LINKER_FILE_DIR:${LIB}>)
    else(TARGET ${LIB})
        # name of library file with directory and extension stripped
        get_filename_component(LIBDIR ${LIB} DIRECTORY)
        list(APPEND HDF5_LIBRARY_DIRS ${LIBDIR})
    endif(TARGET ${LIB})
endforeach(LIB)
# make the lists still be semicolon-separated after generator expansion
string(REGEX REPLACE ";" "\\\\;" HDF5_LIBRARY_DIRS "${HDF5_LIBRARY_DIRS}")
# no matter what the find_package sets, we only care about the "hdf5" library
set(HDF5_LIBRARIES hdf5)
# make list out of space-separated HDF5_DEFINITIONS
separate_arguments(HDF5_DEFINE UNIX_COMMAND "${HDF5_DEFINITIONS}")

# copy files from root directory that we want in the python package
set(ROOT_SRCS AUTHORS CHANGELOG.rst LICENSE)
file(GLOB_RECURSE DATA_FILES
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/.." "../data/*"
)
foreach(SRCFILE ${ROOT_SRCS} ${DATA_FILES})
    configure_file(../${SRCFILE} ${SRCFILE} COPYONLY)
endforeach(SRCFILE)

# copy C source files from libdigital_rf into build directory
set(C_SRCS
    include/digital_rf.h
    include/digital_rf_version.h
    include/windows/inttypes.h
    include/windows/stdint.h
    include/windows/wincompat.h
    lib/rf_write_hdf5.c
)
foreach(SRCFILE ${C_SRCS})
    configure_file(../c/${SRCFILE} ${SRCFILE} COPYONLY)
endforeach(SRCFILE)

# get list of python source files by generating egg info
set(EGG_INFO_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/temp_egg/${PROJECT_NAME}.egg-info)
file(MAKE_DIRECTORY ${EGG_INFO_OUTPUT})
execute_process(
    COMMAND ${Python_EXECUTABLE} setup.py -q
        egg_info -e ${CMAKE_CURRENT_BINARY_DIR}/temp_egg
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    ERROR_QUIET
)
file(READ ${EGG_INFO_OUTPUT}/SOURCES.txt PYSOURCES)
string(REGEX REPLACE ";" "\\\\;" PYSOURCES "${PYSOURCES}")
string(REGEX REPLACE "\n" ";" PYSOURCES "${PYSOURCES}")

# copy python source files to build directory so we can run setup.py from there
foreach(SRCFILE ${PYSOURCES})
    configure_file(${SRCFILE} ${SRCFILE} COPYONLY)
endforeach(SRCFILE)

# build requires generating some of the gnuradio-companion blocks
macro(gen_block _generator _tmpl _out)
    set(generator ${CMAKE_CURRENT_SOURCE_DIR}/${_generator})
    set(tmpl ${CMAKE_CURRENT_SOURCE_DIR}/${_tmpl})
    set(out ${CMAKE_CURRENT_BINARY_DIR}/${_out})
    list(APPEND GENERATED_BLOCKS ${out})
    add_custom_command(
        OUTPUT ${out}
        COMMAND ${Python_EXECUTABLE} ${generator} ${tmpl} ${out}
        DEPENDS ${generator} ${tmpl}
    )
endmacro(gen_block)
gen_block(
    grc/gen_gr_digital_rf_digital_rf_sourcesink.py
    grc/gr_digital_rf_digital_rf_sink.xml.tmpl
    grc/gr_digital_rf_digital_rf_sink.xml
)
gen_block(
    grc/gen_gr_digital_rf_digital_rf_sourcesink.py
    grc/gr_digital_rf_digital_rf_source.xml.tmpl
    grc/gr_digital_rf_digital_rf_source.xml
)
gen_block(
    grc/gen_gr_digital_rf_digital_rf_sourcesink.py
    grc/gr_digital_rf_digital_rf_sink.block.yml.tmpl
    grc/gr_digital_rf_digital_rf_sink.block.yml
)
gen_block(
    grc/gen_gr_digital_rf_digital_rf_sourcesink.py
    grc/gr_digital_rf_digital_rf_source.block.yml.tmpl
    grc/gr_digital_rf_digital_rf_source.block.yml
)
add_custom_target(gr_digital_rf_grc_generated_blocks ALL
    DEPENDS ${GENERATED_BLOCKS}
)

add_custom_target(digital_rf_source)
add_dependencies(digital_rf_source gr_digital_rf_grc_generated_blocks)
if(NOT TARGET source)
    add_custom_target(source)
endif(NOT TARGET source)
add_dependencies(source digital_rf_source)

add_custom_target(digital_rf_sdist
    COMMAND ${Python_EXECUTABLE} setup.py sdist -d ${CMAKE_BINARY_DIR}/dist
    DEPENDS ${PYSOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(digital_rf_sdist digital_rf_source)
if(NOT TARGET sdist)
    add_custom_target(sdist)
endif(NOT TARGET sdist)
add_dependencies(sdist digital_rf_sdist)

# build and install python package with setup.py
set(PYBUILD_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build)
add_custom_command(
    OUTPUT ${PYBUILD_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env HDF5_INCLUDE_DIRS="${HDF5_INCLUDE_DIRS}"
        HDF5_LIBRARY_DIRS="${HDF5_LIBRARY_DIRS}"
        HDF5_LIBRARIES="${HDF5_LIBRARIES}"
        HDF5_DEFINE="${HDF5_DEFINE}"
        ${Python_EXECUTABLE} setup.py build
    COMMAND ${Python_EXECUTABLE} setup.py egg_info
    COMMAND ${CMAKE_COMMAND} -E touch ${PYBUILD_OUTPUT}
    DEPENDS ${PYSOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(digital_rf_python ALL
    DEPENDS ${PYBUILD_OUTPUT}
)
add_dependencies(digital_rf_python digital_rf_source)
if(NOT TARGET build)
    add_custom_target(build)
endif(NOT TARGET build)
add_dependencies(build digital_rf_python)
add_custom_target(digital_rf_wheel
    COMMAND ${Python_EXECUTABLE} setup.py bdist_wheel -d ${CMAKE_BINARY_DIR}/dist
    DEPENDS ${PYBUILD_OUTPUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
if(NOT TARGET wheel)
    add_custom_target(wheel)
endif(NOT TARGET wheel)
add_dependencies(wheel digital_rf_wheel)
# make sure paths on Windows use a double backslash in order to survive
# setup.py (distutils) parsing when used later
if(WIN32)
    macro(NORMALIZE_PATH PATHVARNAME)
        file(TO_NATIVE_PATH "${${PATHVARNAME}}" ${PATHVARNAME})
        string(REPLACE "\\" "\\\\" ${PATHVARNAME} "${${PATHVARNAME}}")
    endmacro()
    NORMALIZE_PATH(DRF_INSTALL_PREFIX_PYTHON)
    NORMALIZE_PATH(DRF_DATA_PREFIX_PYTHON)
    NORMALIZE_PATH(DRF_SCRIPT_PREFIX_PYTHON)
endif(WIN32)
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/drf_install_python.cmake
    CONTENT "execute_process(COMMAND ${Python_EXECUTABLE} setup.py install\
$<$<BOOL:${DRF_INSTALL_PREFIX_PYTHON}>: --prefix=${DRF_INSTALL_PREFIX_PYTHON}>\
$<$<BOOL:${DRF_DATA_PREFIX_PYTHON}>: --install-data=${DRF_DATA_PREFIX_PYTHON}>\
$<$<BOOL:${DRF_SCRIPT_PREFIX_PYTHON}>: --install-scripts=${DRF_SCRIPT_PREFIX_PYTHON}>\
 --root=\$ENV{DESTDIR}/\
 --single-version-externally-managed\
 --record=${CMAKE_CURRENT_BINARY_DIR}/install_${PROJECT_NAME}_manifest.txt\
 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})\
")
install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/drf_install_python.cmake)

# uninstall target
configure_file(
    ../cmake/cmake_uninstall.cmake.in
    cmake_uninstall.cmake
    IMMEDIATE @ONLY
)

add_custom_target(digital_rf_uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)
