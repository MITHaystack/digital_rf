include(CMakeFindDependencyMacro)
find_dependency(HDF5 REQUIRED)

find_library(HDF5_LIBRARY hdf5 PATHS ${HDF5_LIBRARY_DIRS})
add_library(digital_rf::hdf5 SHARED IMPORTED)
set_target_properties(digital_rf::hdf5 PROPERTIES
    IMPORTED_LOCATION ${HDF5_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS}
)

include("${CMAKE_CURRENT_LIST_DIR}/digital_rfTargets.cmake")
