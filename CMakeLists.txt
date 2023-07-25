cmake_minimum_required(VERSION 3.3...3.8)

# proj

project(ArInPlay
    VERSION 0.0.0
    DESCRIPTION "artifical intelligence playground"
    ) # project

# lang

enable_language(CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# make

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON" CACHE
    STRING "generate \"compile_commands.json\" file for coding tools"
    FORCE) # CMAKE_EXPORT_COMPILE_COMMANDS

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE
        STRING "choose the type of build"
        FORCE) # CMAKE_BUILD_TYPE
endif(NOT CMAKE_BUILD_TYPE)
if(NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE
        PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
        ) # CMAKE_CONFIGURATION_TYPES
endif(NOT CMAKE_CONFIGURATION_TYPES)

set(vMainName "${PROJECT_NAME}")
set(vMainFile "${CMAKE_BINARY_DIR}/${vMainName}")
add_executable(${vMainName} ./Main.cxx)

# libs (TODO)

# test

include(CTest)
enable_testing()

set(vTestName "tHello")
add_test(NAME ${vTestName}
    COMMAND ${vMainFile}
    ) # add_test
set_tests_properties(${vTestName} PROPERTIES PASS_REGULAR_EXPRESSION
    "HelloWorld"
    ) # set_tests_properties

# hide

mark_as_advanced(FORCE CMAKE_INSTALL_PREFIX)

mark_as_advanced(FORCE CMAKE_OSX_ARCHITECTURES)
mark_as_advanced(FORCE CMAKE_OSX_DEPLOYMENT_TARGET)
mark_as_advanced(FORCE CMAKE_OSX_SYSROOT)
