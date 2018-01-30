cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

# get the package version from _version.py
file(READ digital_rf/_version.py VERSION_PY)
string(REGEX REPLACE
    ".*__version__ = ['\"]([0-9]+\\.[0-9]+\\.[0-9]+)[\\.a-z0-9]*['\"].*" "\\1"
    digital_rf_VERSION ${VERSION_PY}
)

project(digital_rf LANGUAGES C VERSION ${digital_rf_VERSION})

include(GNUInstallDirs)

find_program(PYTHON "python" REQUIRED)

# copy files from root directory that we want in the python package
set(ROOT_SRCS AUTHORS CHANGES LICENSE)
foreach(SRCFILE ${ROOT_SRCS})
    configure_file(../${SRCFILE} ${SRCFILE} COPYONLY)
endforeach(SRCFILE)

# get list of python source files by generating egg info
configure_file(setup.py setup.py COPYONLY)
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
    COMMAND ${PYTHON} setup.py sdist -d ${CMAKE_BINARY_DIR}/dist
    DEPENDS ${PYBUILD_OUTPUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(digital_rf_wheel
    COMMAND ${PYTHON} setup.py bdist_wheel -d ${CMAKE_BINARY_DIR}/dist
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
