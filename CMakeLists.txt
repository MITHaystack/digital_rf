cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

# this is the canonical place to set the version for digital_rf
# (used to populate digital_rf_version.h and _version.py)
project(digital_rf LANGUAGES C CXX VERSION 2.5.2)
# set the library version, which is only incremented for library changes
set(libdigital_rf_VERSION_MAJOR 2)
set(libdigital_rf_VERSION_MINOR 0)
set(libdigital_rf_VERSION_PATCH 0)
set(libdigital_rf_VERSION ${libdigital_rf_VERSION_MAJOR}.${libdigital_rf_VERSION_MINOR}.${libdigital_rf_VERSION_PATCH})

include(GNUInstallDirs)

find_package(HDF5 REQUIRED)
# need the dirs to explicitly find just the 'hdf5' library
set(HDF5_LIBRARY_DIRS)
foreach(LIB IN LISTS HDF5_LIBRARIES)
    get_filename_component(LIBDIR ${LIB} DIRECTORY)
    list(APPEND HDF5_LIBRARY_DIRS ${LIBDIR})
endforeach(LIB)
list(REMOVE_DUPLICATES HDF5_LIBRARY_DIRS)
find_library(HDF5_LIBRARY hdf5 HINTS ${HDF5_LIBRARY_DIRS})
# need the library directory for python build (guaranteed to exist since we
# found HDF5 package and must have found the 'hdf5' library)
get_filename_component(HDF5_LIBRARY_DIR ${HDF5_LIBRARY} DIRECTORY)
add_library(digital_rf::hdf5 SHARED IMPORTED)
set_target_properties(digital_rf::hdf5 PROPERTIES
    IMPORTED_LOCATION ${HDF5_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS}
)

enable_testing()

# libdigital_rf
configure_file(include/digital_rf.h include/digital_rf.h COPYONLY)
configure_file(include/digital_rf_version.h.in include/digital_rf_version.h @ONLY)
add_library(digital_rf SHARED lib/rf_write_hdf5.c)
target_include_directories(digital_rf PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/digital_rf>
)
target_link_libraries(digital_rf PUBLIC digital_rf::hdf5)
set_target_properties(digital_rf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY lib
    VERSION ${libdigital_rf_VERSION}
    SOVERSION ${libdigital_rf_VERSION_MAJOR}
    INTERFACE_libdigital_rf_MAJOR_VERSION ${libdigital_rf_VERSION_MAJOR}
    COMPATIBLE_INTERFACE_STRING libdigital_rf_MAJOR_VERSION
)

install(TARGETS digital_rf EXPORT DigitalRFTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/digital_rf.h
    ${CMAKE_CURRENT_BINARY_DIR}/include/digital_rf_version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/digital_rf
)

configure_file(lib/pkgconfig/libdigital_rf.pc.in lib/pkgconfig/libdigital_rf.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/lib/pkgconfig/libdigital_rf.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/digital_rfConfigVersion.cmake"
  VERSION ${digital_rf_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(EXPORT DigitalRFTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/digital_rfTargets.cmake"
  NAMESPACE digital_rf::
)
configure_file(cmake/digital_rfConfig.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/digital_rfConfig.cmake"
  COPYONLY
)

set(ConfigPackageLocation ${CMAKE_INSTALL_LIBDIR}/cmake/digital_rf)
install(EXPORT DigitalRFTargets
  FILE
    digital_rfTargets.cmake
  NAMESPACE
    digital_rf::
  DESTINATION
    ${ConfigPackageLocation}
)
install(
  FILES
    cmake/digital_rfConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/digital_rfConfigVersion.cmake"
  DESTINATION
    ${ConfigPackageLocation}
)

# digital_rf python package
file(GLOB PYTHON_SRCS "python/*.py" "tools/*.py")
find_program(PYTHON "python" REQUIRED)
set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
set(SETUP_PY ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
set(PYBUILD_DEPS
    ${PYTHON_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/py_rf_write_hdf5.c
    digital_rf
    ${SETUP_PY}
)
set(PYBUILD_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build)

configure_file(${SETUP_PY_IN} ${SETUP_PY})
add_custom_command(OUTPUT ${PYBUILD_OUTPUT}
                   COMMAND ${PYTHON} ${SETUP_PY} build
                   COMMAND ${PYTHON} ${SETUP_PY} egg_info
                   COMMAND ${CMAKE_COMMAND} -E touch ${PYBUILD_OUTPUT}
                   DEPENDS ${PYBUILD_DEPS})
add_custom_target(python ALL DEPENDS ${PYBUILD_OUTPUT})
install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --prefix ${CMAKE_INSTALL_PREFIX} --single-version-externally-managed --record ${CMAKE_CURRENT_BINARY_DIR}/install_python_manifest.txt)")

# docs, examples, tests
add_subdirectory(docs)
add_subdirectory(examples EXCLUDE_FROM_ALL)
add_subdirectory(tests)

# gr_drf gnuradio module
option(ENABLE_GNURADIO "Build gr_drf GNU Radio module." ON)
set(GR_REQUIRED_COMPONENTS RUNTIME PMT)
find_package(Gnuradio)
if(GNURADIO_RUNTIME_FOUND AND ENABLE_GNURADIO)
  message("| GNU Radio found, gr_drf module ENABLED.")
  add_subdirectory(gnuradio)
else(GNURADIO_RUNTIME_FOUND AND ENABLE_GNURADIO)
  message("| GNU Radio not found or disabled, gr_drf module will be SKIPPED.")
endif(GNURADIO_RUNTIME_FOUND AND ENABLE_GNURADIO)


# add uninstall target
# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY
)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)
