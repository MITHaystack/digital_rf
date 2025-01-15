cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

# can't do binary dir == source dir because of file copying
if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Must use a build/binary directory different from the\
 source directory.")
endif(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)

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
else(UNIX)
    set(MATH_LIB)
endif(UNIX)

set(HDF5_LIB_TARGETS)
option(DRF_FORCE_INTREE_HDF5 "Force in-tree HDF5 build using static library" OFF)
if(SKBUILD OR DRF_FORCE_INTREE_HDF5)
    include(FetchContent)
    FetchContent_Declare(
        HDF5
        URL https://support.hdfgroup.org/releases/hdf5/v1_14/v1_14_5/downloads/hdf5-1.14.5.tar.gz
        URL_HASH SHA256=ec2e13c52e60f9a01491bb3158cb3778c985697131fc6a342262d32a26e58e44
    )
    FetchContent_MakeAvailable(HDF5)
    # don't build all of the HDF5 targets by default
    set_property(DIRECTORY ${HDF5_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL ON)
    # use the static HDF5 library as the dependency when doing an skbuild build
    list(APPEND HDF5_LIB_TARGETS hdf5-static)
    # since we're using it, target needs to be installed in the export group
    install(TARGETS hdf5-static
        EXPORT libdigital_rfTargets
    )
else()
    find_package(HDF5 REQUIRED COMPONENTS C)
    # HDF5 can have Threads::Threads target, otherwise undefined without Threads
    find_package(Threads QUIET)
    # use imported targets from HDF5_LIBRARIES or take the supplied library path
    # and turn it into an imported target if it is an hdf5 library
    foreach(LIB IN LISTS HDF5_LIBRARIES)
        if(TARGET ${LIB})
            list(APPEND HDF5_LIB_TARGETS ${LIB})
        else(TARGET ${LIB})
            # name of library file with directory and extension stripped
            get_filename_component(LIBNAME ${LIB} NAME_WE)
            # strip leading lib (if it exists) to get target name
            string(REGEX REPLACE "^lib(.*)" "\\1" TGT ${LIBNAME})
            # exclude non-hdf5 libraries (libs that HDF5 linked against)
            if(${TGT} MATCHES ".*hdf5.*")
                # unknown library type so we don't have to find dll on windows
                add_library(digital_rf::${TGT} UNKNOWN IMPORTED)
                set_target_properties(digital_rf::${TGT} PROPERTIES
                    IMPORTED_LOCATION ${LIB}
                )
                if(HDF5_INCLUDE_DIRS)
                    set_property(TARGET digital_rf::${TGT} APPEND PROPERTY
                        INTERFACE_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS}
                    )
                endif(HDF5_INCLUDE_DIRS)
                if(HDF5_DEFINITIONS)
                    string(REGEX REPLACE "-D" " " HDF5_DEFINES ${HDF5_DEFINITIONS})
                    separate_arguments(HDF5_DEFINES UNIX_COMMAND "${HDF5_DEFINES}")
                    set_property(TARGET digital_rf::${TGT} APPEND PROPERTY
                        INTERFACE_COMPILE_DEFINITIONS ${HDF5_DEFINES}
                    )
                endif(HDF5_DEFINITIONS)
                list(APPEND HDF5_LIB_TARGETS digital_rf::${TGT})
            endif(${TGT} MATCHES ".*hdf5.*")
        endif(TARGET ${LIB})
    endforeach(LIB)
endif(SKBUILD OR DRF_FORCE_INTREE_HDF5)

# build/install libdigital_rf
configure_file(include/digital_rf.h include/digital_rf.h COPYONLY)
configure_file(include/digital_rf_version.h include/digital_rf_version.h COPYONLY)
if(WIN32)
    configure_file(include/windows/inttypes.h include/inttypes.h COPYONLY)
    configure_file(include/windows/stdint.h include/stdint.h COPYONLY)
    configure_file(include/windows/wincompat.h include/wincompat.h COPYONLY)
endif(WIN32)
add_library(digital_rf lib/rf_write_hdf5.c)
add_library(digital_rf::digital_rf ALIAS digital_rf)
if(NOT TARGET build)
    add_custom_target(build)
endif(NOT TARGET build)
add_dependencies(build digital_rf)
target_compile_definitions(digital_rf PUBLIC
    "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:digital_rf_STATIC_DEFINE>"
)
target_include_directories(digital_rf PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/digital_rf>
)
target_link_libraries(digital_rf PUBLIC ${HDF5_LIB_TARGETS} PRIVATE ${MATH_LIB})
set_target_properties(digital_rf PROPERTIES
    VERSION ${libdigital_rf_VERSION}
    SOVERSION ${libdigital_rf_VERSION_MAJOR}
    INTERFACE_libdigital_rf_MAJOR_VERSION ${libdigital_rf_VERSION_MAJOR}
    COMPATIBLE_INTERFACE_STRING libdigital_rf_MAJOR_VERSION
    INSTALL_RPATH_USE_LINK_PATH ON
    # make sure library is compiled with PIC even if static, because we want
    # to be able to link this into a shared library like the python one
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS digital_rf
    EXPORT libdigital_rfTargets
    COMPONENT C
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/digital_rf.h
    ${CMAKE_CURRENT_BINARY_DIR}/include/digital_rf_version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/digital_rf
    COMPONENT C
)

# build/install pkgconfig configuration files
configure_file(lib/pkgconfig/digital_rf.pc.in lib/pkgconfig/digital_rf.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/digital_rf.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    COMPONENT C
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
    COMPONENT C
)
install(
    FILES
        cmake/libdigital_rfConfig.cmake
        "${CMAKE_CURRENT_BINARY_DIR}/libdigital_rfConfigVersion.cmake"
    DESTINATION ${ConfigPackageLocation}
    COMPONENT C
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
