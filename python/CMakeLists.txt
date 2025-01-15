find_package(
    Python
    COMPONENTS Interpreter Development.Module NumPy
    REQUIRED
)

# check for mako, needed to create gr_digital_rf source/sink blocks from templates
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import mako"
    RESULT_VARIABLE MAKO_CHECK_RESULT
    OUTPUT_QUIET
)
if(MAKO_CHECK_RESULT EQUAL 0)
    message(STATUS "Python checking for `mako` - found")
    set(MAKO_FOUND TRUE)
else()
    message(STATUS "Python checking for `mako` - not found")
    set(MAKO_FOUND FALSE)
endif()
if(NOT MAKO_FOUND)
    message(SEND_ERROR "Could not find required build dependency: Python package `mako`")
endif()

add_custom_target(digital_rf_python ALL)
if(NOT TARGET build)
    add_custom_target(build)
endif(NOT TARGET build)

python_add_library(digital_rf_python_module MODULE lib/py_rf_write_hdf5.c WITH_SOABI)
target_link_libraries(digital_rf_python_module PUBLIC digital_rf PRIVATE Python::NumPy)
set_target_properties(digital_rf_python_module PROPERTIES OUTPUT_NAME _py_rf_write_hdf5)
add_dependencies(digital_rf_python digital_rf_python_module)
add_dependencies(build digital_rf_python_module)

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
add_custom_target(gr_digital_rf_grc_generated_blocks
    DEPENDS ${GENERATED_BLOCKS}
)
add_dependencies(digital_rf_python gr_digital_rf_grc_generated_blocks)

if(SKBUILD)
    # skbuild should be set up to have CMAKE_INSTALL_PREFIX point to the "DATA"
    # directory in the built wheel, so we need the installation directories for
    # the extension module and scripts which are separate from that root
    set(INSTALL_PLATLIB_DIR ${SKBUILD_PLATLIB_DIR})
    set(INSTALL_SCRIPTS_DIR ${SKBUILD_SCRIPTS_DIR})
else()
    # set the installation directories when skbuild is not used
    if(WIN32)
        set(_SCRIPTS_DIR ${CMAKE_INSTALL_PREFIX}/Scripts)
    else()
        set(_SCRIPTS_DIR ${CMAKE_INSTALL_BINDIR})
    endif()
    set(DRF_INSTALL_PLATLIB ${Python_SITEARCH} CACHE STRING
        "Python platlib (site-packages) path to use for installation"
    )
    set(DRF_INSTALL_SCRIPTS ${_SCRIPTS_DIR} CACHE STRING
        "Python scripts (bin) path to use for installation"
    )
    cmake_path(
        ABSOLUTE_PATH DRF_INSTALL_PLATLIB
        BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
        NORMALIZE
        OUTPUT_VARIABLE INSTALL_PLATLIB_DIR
    )
    cmake_path(
        ABSOLUTE_PATH DRF_INSTALL_SCRIPTS
        BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
        NORMALIZE
        OUTPUT_VARIABLE INSTALL_SCRIPTS_DIR
    )
    cmake_path(IS_PREFIX
        CMAKE_INSTALL_PREFIX ${INSTALL_PLATLIB_DIR}
        NORMALIZE PLATLIB_IS_IN_PREFIX
    )
    if(NOT PLATLIB_IS_IN_PREFIX)
        message(WARNING "Python DRF_INSTALL_PLATLIB (${DRF_INSTALL_PLATLIB}) is not relative or prefixed by CMAKE_INSTALL_PREFIX (${CMAKE_INSTALL_PREFIX}). This is unusual and might not be intended. Check and/or set these variables accordingly.")
    endif()
    cmake_path(IS_PREFIX
        CMAKE_INSTALL_PREFIX ${INSTALL_SCRIPTS_DIR}
        NORMALIZE SCRIPTS_IS_IN_PREFIX
    )
    if(NOT SCRIPTS_IS_IN_PREFIX)
        message(WARNING "Python DRF_INSTALL_SCRIPTS (${DRF_INSTALL_SCRIPTS}) is not relative or prefixed by CMAKE_INSTALL_PREFIX (${CMAKE_INSTALL_PREFIX}). This is unusual and might not be intended. Check and/or set these variables accordingly.")
    endif()

    # check for Python build package, which is used to create/install valid
    # metadata for the Python package
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import build"
        RESULT_VARIABLE PYTHON_BUILD_CHECK_RESULT
        OUTPUT_QUIET
    )
    if(PYTHON_BUILD_CHECK_RESULT EQUAL 0)
        message(STATUS "Python checking for `build` - found")
        set(PYTHON_BUILD_FOUND TRUE)
    else()
        message(STATUS "Python checking for `build` - not found")
        set(PYTHON_BUILD_FOUND FALSE)
    endif()
    if(NOT PYTHON_BUILD_FOUND)
        message(SEND_ERROR "Could not find required build dependency: Python package `build`")
    endif()

    # create the Python package metadata directory using the `build` frontend
    set(DRF_DIST_INFO_DIR_NAME digital_rf-${digital_rf_VERSION}.dist-info)
    set(DIST_INFO_ARTIFACT_FILE ${CMAKE_CURRENT_BINARY_DIR}/dist-info-created)
    configure_file(
        ${PROJECT_SOURCE_DIR}/cmake/create_metadata_dir.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/create_metadata_dir.py
    )
    add_custom_command(
        OUTPUT ${DIST_INFO_ARTIFACT_FILE}
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/create_metadata_dir.py --build
        DEPENDS ${PROJECT_SOURCE_DIR}/pyproject.toml
            ${PROJECT_SOURCE_DIR}/cmake/create_metadata_dir.py.in
    )
    add_custom_target(digital_rf_python_package_metadata
        DEPENDS ${DIST_INFO_ARTIFACT_FILE}
    )
    add_dependencies(digital_rf_python digital_rf_python_package_metadata)

    install(
        CODE "execute_process(COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/create_metadata_dir.py --install)"
        COMPONENT python
    )

    # skbuild installs the pure python packages itself, but without it we
    # need to do that here
    install(
        DIRECTORY digital_rf gr_digital_rf
        DESTINATION ${INSTALL_PLATLIB_DIR}
        COMPONENT python
    )
endif(SKBUILD)


install(TARGETS digital_rf_python_module DESTINATION ${INSTALL_PLATLIB_DIR}/digital_rf COMPONENT python)

install(
    FILES ${GENERATED_BLOCKS}
    DESTINATION ${CMAKE_INSTALL_DATADIR}/gnuradio/grc/blocks
    COMPONENT python
)

install(
    DIRECTORY grc/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/gnuradio/grc/blocks
    COMPONENT python
    FILES_MATCHING PATTERN "*.yml" PATTERN "*.xml"
)

install(
    DIRECTORY tools/
    DESTINATION ${INSTALL_SCRIPTS_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT python
    FILES_MATCHING PATTERN "*.py"
)
