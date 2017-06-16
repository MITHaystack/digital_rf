########################################################################
# Find dependencies
########################################################################
if(TARGET digital_rf)
    set(DIGITAL_RF_TARGET digital_rf)
else(TARGET digital_rf)
    find_package(digital_rf REQUIRED)
    set(DIGITAL_RF_TARGET digital_rf::digital_rf)
endif(TARGET digital_rf)

########################################################################
# Setup library
########################################################################
include(GrPlatform) #define LIB_SUFFIX

list(APPEND drf_sources
    digital_rf_sink_impl.cc
    )

set(drf_sources "${drf_sources}" PARENT_SCOPE)
if(NOT drf_sources)
	MESSAGE(STATUS "No C++ sources... skipping lib/")
	return()
endif(NOT drf_sources)

add_library(gnuradio-drf SHARED ${drf_sources})
target_link_libraries(gnuradio-drf ${DIGITAL_RF_TARGET} ${Boost_LIBRARIES} ${GNURADIO_ALL_LIBRARIES} -lm)
set_target_properties(gnuradio-drf PROPERTIES DEFINE_SYMBOL "gnuradio_drf_EXPORTS")

if(APPLE)
    set_target_properties(gnuradio-drf PROPERTIES
        INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
    )
endif(APPLE)

########################################################################
# Install built library files
########################################################################
install(TARGETS gnuradio-drf
    LIBRARY DESTINATION lib${LIB_SUFFIX} # .so/.dylib file
    ARCHIVE DESTINATION lib${LIB_SUFFIX} # .lib file
    RUNTIME DESTINATION bin              # .dll file
)
