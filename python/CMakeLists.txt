cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

# this is the canonical place to set the version for digital_rf
# (used to populate _version.py)
project(digital_rf LANGUAGES C VERSION 2.5.4)

include(GNUInstallDirs)

find_program(PYTHON "python" REQUIRED)

if(NOT TARGET digital_rf::digital_rf)
    find_package(libdigital_rf REQUIRED)
endif(NOT TARGET digital_rf::digital_rf)

# generate setup.py with CMake-populated include, library, and link args
set(LIBDIGITAL_RF_INCLUDE_DIRS
    $<TARGET_PROPERTY:digital_rf::digital_rf,INTERFACE_INCLUDE_DIRECTORIES>
)
set(LIBDIGITAL_RF_LINK_TARGETS
    digital_rf::digital_rf
    digital_rf::hdf5
)
foreach(LTGT ${LIBDIGITAL_RF_LINK_TARGETS})
    list(APPEND LIBDIGITAL_RF_LIBRARY_DIRS $<TARGET_LINKER_FILE_DIR:${LTGT}>)
    list(APPEND LIBDIGITAL_RF_LINK_ARGS -l:$<TARGET_LINKER_FILE_NAME:${LTGT}>)
endforeach(LTGT)
# replace CMake variables in setup.py (like the ones above)
configure_file(setup.py.in setup.py)
# replace generator expressions to get actual include/link values
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py INPUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

# generate _version.py with version number from CMake
configure_file(digital_rf/_version.py.in digital_rf/_version.py)
configure_file(gr_digital_rf/_version.py.in gr_digital_rf/_version.py)
# copy files from root directory that we want in the python package
set(ROOT_SRCS AUTHORS CHANGES LICENSE)
foreach(SRCFILE ${ROOT_SRCS})
    configure_file(../${SRCFILE} ${SRCFILE} COPYONLY)
endforeach(SRCFILE)

# get list of python source files by generating egg info
set(EGG_INFO_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/temp_egg/${PROJECT_NAME}.egg-info)
file(MAKE_DIRECTORY ${EGG_INFO_OUTPUT})
execute_process(
    COMMAND ${PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/setup.py -q
        egg_info -e ${CMAKE_CURRENT_BINARY_DIR}/temp_egg
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
file(READ ${EGG_INFO_OUTPUT}/SOURCES.txt PYSOURCES)
STRING(REGEX REPLACE ";" "\\\\;" PYSOURCES "${PYSOURCES}")
STRING(REGEX REPLACE "\n" ";" PYSOURCES "${PYSOURCES}")

# copy python source files to build directory so we can run setup.py from there
foreach(SRCFILE ${PYSOURCES})
    configure_file(${SRCFILE} ${SRCFILE} COPYONLY)
endforeach(SRCFILE)

# build requires generating some of the gnuradio-companion blocks
macro(gen_block_xml _generator _xml_block)
    set(generator ${CMAKE_CURRENT_SOURCE_DIR}/${_generator})
    set(xml_block ${CMAKE_CURRENT_BINARY_DIR}/${_xml_block})
    list(APPEND GENERATED_BLOCKS ${xml_block})
    add_custom_command(
        OUTPUT ${xml_block}
        COMMAND ${PYTHON} ${generator} ${xml_block}
        DEPENDS ${generator}
    )
endmacro(gen_block_xml)
gen_block_xml(
    grc/gen_gr_digital_rf_digital_rf_sink.py
    grc/gr_digital_rf_digital_rf_sink.xml
)
gen_block_xml(
    grc/gen_gr_digital_rf_digital_rf_source.py
    grc/gr_digital_rf_digital_rf_source.xml
)
add_custom_target(gr_digital_rf_grc_xml_blocks ALL
    DEPENDS ${GENERATED_BLOCKS}
)

# build and install python package with setup.py
set(PYBUILD_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build)
set(PYBUILD_DEPS
    ${PYSOURCES} ${DIGITAL_RF_TARGET} gr_digital_rf_grc_xml_blocks
)
add_custom_command(
    OUTPUT ${PYBUILD_OUTPUT}
    COMMAND ${PYTHON} setup.py build
    COMMAND ${PYTHON} setup.py egg_info
    COMMAND ${CMAKE_COMMAND} -E touch ${PYBUILD_OUTPUT}
    DEPENDS ${PYBUILD_DEPS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(digital_rf_python ALL
    DEPENDS ${PYBUILD_OUTPUT}
)
add_custom_target(digital_rf_sdist
    COMMAND ${PYTHON} setup.py sdist -d ${CMAKE_BINARY_DIR}
    DEPENDS ${PYBUILD_OUTPUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
install(CODE "execute_process(COMMAND ${PYTHON} setup.py install\
 --prefix ${CMAKE_INSTALL_PREFIX} --single-version-externally-managed\
 --record ${CMAKE_CURRENT_BINARY_DIR}/install_${PROJECT_NAME}_manifest.txt\
 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})\
")

# uninstall target
configure_file(
    ../cmake/cmake_uninstall.cmake.in
    cmake_uninstall.cmake
    IMMEDIATE @ONLY
)

add_custom_target(digital_rf_uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)
