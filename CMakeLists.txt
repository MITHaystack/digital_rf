cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0094 NEW)

if(SKBUILD)
    set(digital_rf_VERSION ${SKBUILD_PROJECT_VERSION})
else()
    set(digital_rf_VERSION 2.6.11)
endif(SKBUILD)
project(digital_rf LANGUAGES C VERSION ${digital_rf_VERSION})

include(GNUInstallDirs)
enable_testing()

# create targets that sub-projects will add dependencies to
add_custom_target(build)

# include sub-projects
add_subdirectory(c)
add_subdirectory(docs)
add_subdirectory(matlab EXCLUDE_FROM_ALL)
add_subdirectory(python)

# shortcut target for digital_rf_toolbox
if(${ENABLE_MATLAB})
    add_custom_target(matlab)
    add_dependencies(matlab digital_rf_toolbox)
endif(${ENABLE_MATLAB})

# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY
)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)
add_dependencies(uninstall libdigital_rf_uninstall)
