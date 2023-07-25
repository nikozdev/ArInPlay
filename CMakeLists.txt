cmake_minimum_required(VERSION 3.3...3.8)

# proj

project(ArInPlay
    VERSION 0.0.1
    DESCRIPTION "artifical intelligence playground"
    ) # project

# lang

enable_language(CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# make

if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE
    STRING "generate \"compile_commands.json\" file for coding tools"
    FORCE) # CMAKE_EXPORT_COMPILE_COMMANDS
    set_property(CACHE CMAKE_EXPORT_COMPILE_COMMANDS
        PROPERTY STRINGS ON OFF
        ) # CMAKE_CONFIGURATION_TYPES
endif(NOT CMAKE_EXPORT_COMPILE_COMMANDS)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE
        STRING "choose the type of build"
        FORCE) # CMAKE_BUILD_TYPE
    set_property(CACHE CMAKE_BUILD_TYPE
        PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
        ) # CMAKE_CONFIGURATION_TYPES
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)

# exec

set(vExecName "${PROJECT_NAME}")
set(vExecPath "${CMAKE_BINARY_DIR}/${vExecName}")
add_executable(${vExecName} ./Main.cxx)

# libs

find_package(fmt REQUIRED)
target_link_libraries(${vExecName} fmt::fmt)
mark_as_advanced(FORCE fmt_DIR)

find_package(Boost REQUIRED COMPONENTS filesystem)
target_link_libraries(${vExecName} Boost::filesystem)
mark_as_advanced(FORCE Boost_FILESYSTEM_LIBRARY_RELEA)
mark_as_advanced(FORCE Boost_INCLUDE_DIR)
mark_as_advanced(FORCE Boost_DIR)

# test

include(CTest)
enable_testing()

set(vTestName "tFdirPathMain")
add_test(NAME ${vTestName}
    COMMAND ${vExecPath} ${vTestName}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ) # add_test
set_tests_properties(${vTestName} PROPERTIES PASS_REGULAR_EXPRESSION
    "\\[${vTestName}\\]=\\(${CMAKE_SOURCE_DIR}\\)"
    ) # set_tests_properties

# hide

mark_as_advanced(FORCE CMAKE_INSTALL_PREFIX)

mark_as_advanced(FORCE CMAKE_OSX_ARCHITECTURES)
mark_as_advanced(FORCE CMAKE_OSX_DEPLOYMENT_TARGET)
mark_as_advanced(FORCE CMAKE_OSX_SYSROOT)
