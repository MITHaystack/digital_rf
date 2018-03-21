cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

# get the package version from digital_rf.prj
file(READ digital_rf.prj.in DIGITAL_RF_PRJIN)
string(REGEX REPLACE
    ".*<param.version>([0-9]+\\.[0-9]+\\.[0-9]+)[\\.a-z0-9]*</param.version>.*"
    "\\1"
    digital_rf_matlab_VERSION ${DIGITAL_RF_PRJIN}
)

project(digital_rf_matlab LANGUAGES NONE VERSION ${digital_rf_matlab_VERSION})

option(ENABLE_MATLAB "Enable target for Digital RF Toolbox for MATLAB." ON)

if(${ENABLE_MATLAB})
    if(WIN32 OR APPLE)
        find_package(Matlab COMPONENTS MAIN_PROGRAM)
    else(WIN32 OR APPLE)
        # shortcut to just find binary (which is what FindMatlab does anyway)
        # so we don't wait on calling matlab to get its version
        find_program(Matlab_MAIN_PROGRAM
            matlab
            HINTS ${Matlab_ROOT_DIR} ${Matlab_ROOT_DIR}/bin
        )
        if(Matlab_MAIN_PROGRAM)
            set(Matlab_FOUND TRUE)
        else(Matlab_MAIN_PROGRAM)
            set(Matlab_FOUND FALSE)
        endif(Matlab_MAIN_PROGRAM)
    endif(WIN32 OR APPLE)
    if(NOT ${Matlab_FOUND})
        set(ENABLE_MATLAB OFF CACHE BOOL "MATLAB not found" FORCE)
        message(STATUS "\
| MATLAB binary not found, disabling Digital RF MATLAB Toolbox.\
 Set 'Matlab_ROOT_DIR' to correct binary path to enable.\
"
        )
    endif(NOT ${Matlab_FOUND})
endif(${ENABLE_MATLAB})

# re-check ENABLE_MATLAB because it could be disabled by not finding binary
if(${ENABLE_MATLAB})
    set(DIGITAL_RF_PRJ ${CMAKE_CURRENT_BINARY_DIR}/digital_rf.prj)
    set(DIGITAL_RF_MLTBX ${CMAKE_CURRENT_BINARY_DIR}/digital_rf.mltbx)
    set(DIGITAL_RF_TOOLBOX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/digital_rf_toolbox)
    file(GLOB MATLAB_SRCS ${DIGITAL_RF_TOOLBOX_ROOT}/*)
    foreach(SRCFILE ${MATLAB_SRCS})
        list(APPEND MATLAB_SRC_ITEM_XML "<file>${SRCFILE}</file>\n")
    endforeach(SRCFILE)
    string(CONCAT DIGITAL_RF_TOOLBOX_ROOTFILES ${MATLAB_SRC_ITEM_XML})
    configure_file(digital_rf.prj.in ${DIGITAL_RF_PRJ} @ONLY)

    # command to generate toolbox
    add_custom_command(
        OUTPUT ${DIGITAL_RF_MLTBX}
        COMMAND ${Matlab_MAIN_PROGRAM} -nodisplay -r "\
    try, matlab.addons.toolbox.packageToolbox('${DIGITAL_RF_PRJ}'); \
    catch ME, warning(getReport(ME)); end; quit"
        DEPENDS ${MATLAB_SRCS} ${DIGITAL_RF_PRJ}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM
    )
    add_custom_target(digital_rf_toolbox ALL
        DEPENDS ${DIGITAL_RF_MLTBX}
    )
endif(${ENABLE_MATLAB})
