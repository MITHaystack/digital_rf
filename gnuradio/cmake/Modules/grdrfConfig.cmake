INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_GRDRF gr_drf)

FIND_PATH(
    GRDRF_INCLUDE_DIRS
    NAMES gr_drf/api.h
    HINTS $ENV{GRDRF_DIR}/include
        ${PC_GRDRF_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GRDRF_LIBRARIES
    NAMES gnuradio-drf
    HINTS $ENV{GRDRF_DIR}/lib
        ${PC_GRDRF_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GRDRF DEFAULT_MSG GRDRF_LIBRARIES GRDRF_INCLUDE_DIRS)
MARK_AS_ADVANCED(GRDRF_LIBRARIES GRDRF_INCLUDE_DIRS)

