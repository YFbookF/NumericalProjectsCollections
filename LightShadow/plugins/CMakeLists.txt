cmake_minimum_required (VERSION 3.1)
project (layered_plugins)

# Require C++11 options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories ("$ENV{MITSUBA_DIR}/include")
include_directories ("$ENV{MITSUBA_DIR}/dependencies/include")

# Set special defines flags
add_definitions( -DSINGLE_PRECISION -DSPECTRUM_SAMPLES=3 )

# Link flags
find_library(MTS_CORE   NAMES mitsuba-core
                        HINTS "$ENV{MITSUBA_DIR}"
                        PATH_SUFFIXES "Mitsuba.app/Contents/Frameworks" "dist")
find_library(MTS_RENDER NAMES mitsuba-render
                        HINTS "$ENV{MITSUBA_DIR}"
                        PATH_SUFFIXES "Mitsuba.app/Contents/Frameworks" "dist")
find_library(MTS_HW     NAMES mitsuba-hw
                        HINTS "$ENV{MITSUBA_DIR}"
                        PATH_SUFFIXES "Mitsuba.app/Contents/Frameworks" "dist")

function(add_plugin name sources)
	add_library(${name} SHARED ${sources})
   set_target_properties(${name} PROPERTIES PREFIX "")
   target_link_libraries(${name} ${ALTA_LIB} ${MTS_CORE} ${MTS_RENDER} ${MTS_HW})
   install(TARGETS ${name} DESTINATION "$ENV{MITSUBA_DIR}/dist/plugins")
endfunction(add_plugin)

add_plugin("fresnel_approx" "fresnel_approx.cpp")
add_plugin("fresnel_reparam" "fresnel_reparam.cpp")
