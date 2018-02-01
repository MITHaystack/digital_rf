cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

project(digital_rf_project LANGUAGES NONE)

include(GNUInstallDirs)
enable_testing()

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

# source distribution target
add_custom_target(sdist)
add_dependencies(sdist digital_rf_sdist)

# wheel distribution target
add_custom_target(wheel)
add_dependencies(wheel digital_rf_wheel)

# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY
)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)
add_dependencies(uninstall libdigital_rf_uninstall digital_rf_uninstall)
