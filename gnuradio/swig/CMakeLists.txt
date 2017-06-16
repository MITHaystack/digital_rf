########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT drf_sources)
	MESSAGE(STATUS "No C++ sources... skipping swig/")
	return()
endif(NOT drf_sources)

########################################################################
# Include swig generation macros
########################################################################
find_package(SWIG REQUIRED)
find_package(PythonLibs 2 REQUIRED)
include(GrSwig)
include(GrPython)

########################################################################
# Find numpy and add to includes
########################################################################
find_package(NumPy REQUIRED)
list(APPEND GR_SWIG_INCLUDE_DIRS ${NUMPY_INCLUDE_DIRS})

########################################################################
# Setup swig generation
########################################################################
foreach(incdir ${GNURADIO_RUNTIME_INCLUDE_DIRS})
    list(APPEND GR_SWIG_INCLUDE_DIRS ${incdir}/gnuradio/swig)
endforeach(incdir)

set(GR_SWIG_LIBRARIES gnuradio-drf)

GR_SWIG_MAKE(drf_swig drf_swig.i)

########################################################################
# Install the build swig module
########################################################################
GR_SWIG_INSTALL(TARGETS drf_swig DESTINATION ${GR_PYTHON_DIR}/gr_drf)

########################################################################
# Install swig .i files for development
########################################################################
install(
    FILES
    drf_swig.i
    DESTINATION ${GR_INCLUDE_DIR}/gr_drf/swig
)
