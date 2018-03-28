include(CMakeFindDependencyMacro)
find_dependency(HDF5 REQUIRED COMPONENTS C)
# HDF5 can have Threads::Threads target, otherwise undefined without Threads
find_dependency(Threads QUIET)

# use imported targets from HDF5_LIBRARIES or take the supplied library
# path and turn it into an imported target if it is an hdf5 library
set(HDF5_LIB_TARGETS)
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
            list(APPEND HDF5_LIB_TARGETS digital_rf::${TGT})
        endif(${TGT} MATCHES ".*hdf5.*")
    endif(TARGET ${LIB})
endforeach(LIB)

if(NOT TARGET digital_rf::digital_rf)
    include("${CMAKE_CURRENT_LIST_DIR}/libdigital_rfTargets.cmake")
endif(NOT TARGET digital_rf::digital_rf)
