include(CMakeFindDependencyMacro)
find_dependency(HDF5 REQUIRED)

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

include("${CMAKE_CURRENT_LIST_DIR}/digital_rfTargets.cmake")
