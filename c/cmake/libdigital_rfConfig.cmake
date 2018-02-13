include(CMakeFindDependencyMacro)
find_dependency(HDF5 REQUIRED)

# make each of the HDF5_LIBRARIES (HDF5 and what it linked against)
# an imported target
foreach(LIB IN LISTS HDF5_LIBRARIES)
    if(TARGET ${LIB})
        list(APPEND HDF5_LIB_TARGETS ${LIB})
    else(TARGET ${LIB})
        # name of library file with directory and extension stripped
        get_filename_component(LIBNAME ${LIB} NAME_WE)
        # strip leading lib (if it exists) to get target name
        string(REGEX REPLACE "^lib(.*)" "\\1" TGT ${LIBNAME})
        add_library(digital_rf::${TGT} SHARED IMPORTED)
        set_target_properties(digital_rf::${TGT} PROPERTIES
            IMPORTED_LOCATION ${LIB}
        )
        if(HDF5_INCLUDE_DIRS)
            set_property(TARGET digital_rf::${TGT} APPEND PROPERTY
                INTERFACE_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS}
            )
        endif(HDF5_INCLUDE_DIRS)
        list(APPEND HDF5_LIB_TARGETS digital_rf::${TGT})
    endif(TARGET ${LIB})
endforeach(LIB)

include("${CMAKE_CURRENT_LIST_DIR}/libdigital_rfTargets.cmake")
