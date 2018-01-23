cmake_minimum_required(VERSION 3.0)
project(gr_digital_rf VERSION ${digital_rf_VERSION} LANGUAGES NONE)

find_program(PYTHON "python" REQUIRED)

# get list of python source files by generating egg info
set(EGG_INFO_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/temp_egg/${PROJECT_NAME}.egg-info)
file(MAKE_DIRECTORY ${EGG_INFO_OUTPUT})
execute_process(
    COMMAND ${PYTHON} setup.py egg_info -e ${CMAKE_CURRENT_BINARY_DIR}/temp_egg
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
file(READ ${EGG_INFO_OUTPUT}/SOURCES.txt PYSOURCES)
STRING(REGEX REPLACE ";" "\\\\;" PYSOURCES "${PYSOURCES}")
STRING(REGEX REPLACE "\n" ";" PYSOURCES "${PYSOURCES}")

# copy python source files to build directory so we can run setup.py from there
foreach(SRCFILE ${PYSOURCES})
    configure_file(${SRCFILE} ${SRCFILE} COPYONLY)
endforeach(SRCFILE)

# generate _version.py with version number from CMake
configure_file(python/_version.py.in python/_version.py)

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
set(GRPYBUILD_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build)
add_custom_command(
    OUTPUT ${GRPYBUILD_OUTPUT}
    COMMAND ${PYTHON} setup.py build
    COMMAND ${PYTHON} setup.py egg_info
    COMMAND ${CMAKE_COMMAND} -E touch ${GRPYBUILD_OUTPUT}
    DEPENDS ${PYSOURCES} gr_digital_rf_grc_xml_blocks
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(gr_digital_rf_python ALL
    DEPENDS ${GRPYBUILD_OUTPUT}
)
add_custom_target(gr_digital_rf_sdist
    COMMAND ${PYTHON} setup.py sdist -d ${CMAKE_BINARY_DIR}
    DEPENDS ${GRPYBUILD_OUTPUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(sdist gr_digital_rf_sdist)
install(CODE "execute_process(COMMAND ${PYTHON} setup.py install\
 --prefix ${CMAKE_INSTALL_PREFIX} --single-version-externally-managed\
 --record ${CMAKE_BINARY_DIR}/install_${PROJECT_NAME}_manifest.txt\
 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})\
")
