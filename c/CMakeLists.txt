cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

# get the library version from the header file
file(READ include/digital_rf_version.h DIGITAL_RF_VERSION_H)
string(REGEX REPLACE
    ".*#define DIGITAL_RF_VERSION ['\"]([0-9]+\\.[0-9]+\\.[0-9]+)['\"].*" "\\1"
    libdigital_rf_VERSION ${DIGITAL_RF_VERSION_H}
)

project(libdigital_rf LANGUAGES C VERSION ${libdigital_rf_VERSION})

include(GNUInstallDirs)
enable_testing()

# build shared library by default
option(BUILD_SHARED_LIBS "Build libdigital_rf as a shared library" ON)
# set Release configuration by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or\
 CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(UNIX)
    set(MATH_LIB m)
ELSE(UNIX)
    set(MATH_LIB)
ENDIF(UNIX)

find_package(HDF5 REQUIRED)
message(STATUS "HDF5_INCLUDE_DIRS=${HDF5_INCLUDE_DIRS}")
# make each of the HDF5_LIBRARIES (HDF5 and what it linked against)
# an imported target
foreach(LIB IN LISTS HDF5_LIBRARIES)
    # name of library file with directory and extension stripped
    get_filename_component(LIBNAME ${LIB} NAME_WE)
    # strip leading lib (if it exists) to get target name
    string(REGEX REPLACE "^lib(.*)" "\\1" TGT ${LIBNAME})
    add_library(digital_rf::${TGT} SHARED IMPORTED GLOBAL)
    set_target_properties(digital_rf::${TGT} PROPERTIES
        IMPORTED_LOCATION ${LIB}
    )
    if(HDF5_INCLUDE_DIRS)
        set_property(TARGET digital_rf::${TGT} APPEND PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS}
        )
    endif(HDF5_INCLUDE_DIRS)
endforeach(LIB)

# build/install libdigital_rf
configure_file(include/digital_rf.h include/digital_rf.h COPYONLY)
configure_file(include/digital_rf_version.h include/digital_rf_version.h COPYONLY)
add_library(digital_rf lib/rf_write_hdf5.c)
add_library(digital_rf::digital_rf ALIAS digital_rf)
target_include_directories(digital_rf PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/digital_rf>
)
target_link_libraries(digital_rf PUBLIC digital_rf::hdf5 PRIVATE ${MATH_LIB})
set_target_properties(digital_rf PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY lib
    LIBRARY_OUTPUT_DIRECTORY lib
    VERSION ${libdigital_rf_VERSION}
    SOVERSION ${libdigital_rf_VERSION_MAJOR}
    INTERFACE_libdigital_rf_MAJOR_VERSION ${libdigital_rf_VERSION_MAJOR}
    COMPATIBLE_INTERFACE_STRING libdigital_rf_MAJOR_VERSION
    # make sure library is compiled with PIC even if static, because we want
    # to be able to link this into a shared library like the python one
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS digital_rf EXPORT libdigital_rfTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/digital_rf.h
    ${CMAKE_CURRENT_BINARY_DIR}/include/digital_rf_version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/digital_rf
)

# build/install pkgconfig configuration files
configure_file(lib/pkgconfig/digital_rf.pc.in lib/pkgconfig/digital_rf.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/digital_rf.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# add targets to the build-tree export set
export(EXPORT libdigital_rfTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/libdigital_rfTargets.cmake"
    NAMESPACE digital_rf::
)

# export libdigital_rf package for use from the build-tree
export(PACKAGE libdigital_rf)

# create/build Config cmake files for find_package to use
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libdigital_rfConfigVersion.cmake"
    VERSION ${libdigital_rf_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_file(cmake/libdigital_rfConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/libdigital_rfConfig.cmake"
    COPYONLY
)

# install Config/Targets files for find_package to use after installed
set(ConfigPackageLocation ${CMAKE_INSTALL_LIBDIR}/cmake/libdigital_rf)
install(EXPORT libdigital_rfTargets
    NAMESPACE digital_rf::
    DESTINATION ${ConfigPackageLocation}
)
install(
    FILES
        cmake/libdigital_rfConfig.cmake
        "${CMAKE_CURRENT_BINARY_DIR}/libdigital_rfConfigVersion.cmake"
    DESTINATION ${ConfigPackageLocation}
)

# examples, tests
add_subdirectory(examples EXCLUDE_FROM_ALL)
add_subdirectory(tests)

# uninstall target
configure_file(
    ../cmake/cmake_uninstall.cmake.in
    cmake_uninstall.cmake
    IMMEDIATE @ONLY
)

add_custom_target(libdigital_rf_uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)
