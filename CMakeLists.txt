cmake_minimum_required(VERSION 3.3...3.8)

# proj

set(vNameOfProj "ArInPlay")

project(${vNameOfProj}
    VERSION 0.1.1
    DESCRIPTION "artifical intelligence playground"
    ) # project

# lang

enable_language(CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# file

set(vPathToInternal "${CMAKE_SOURCE_DIR}")
set(vPathToResource "${vPathToInternal}/res")

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

set(vNameOfExec ${vNameOfProj})
set(vPathToExec ${CMAKE_BINARY_DIR}/${vNameOfExec})

add_executable(${vNameOfExec} ./Main.cxx)
target_compile_definitions(${vNameOfExec} PRIVATE
    dNameOfProj="${vNameOfExec}"
    dPathToInternal="${vPathToInternal}"
    dPathToResource="${vPathToResource}"
    ) # target_compile_definitions

# libs

find_package(fmt REQUIRED)
target_link_libraries(${vNameOfExec} PRIVATE fmt::fmt)
mark_as_advanced(FORCE fmt_DIR)

find_package(Boost COMPONENTS filesystem REQUIRED)
target_link_libraries(${vNameOfExec} PRIVATE
    Boost::filesystem
    ) # link Boost
mark_as_advanced(FORCE Boost_FILESYSTEM_LIBRARY_RELEA)
mark_as_advanced(FORCE Boost_INCLUDE_DIR)
mark_as_advanced(FORCE Boost_DIR)

find_package(SFML COMPONENTS system window graphics audio REQUIRED)
target_link_libraries(${vNameOfExec} PRIVATE sfml-graphics)
mark_as_advanced(FORCE SFML_DOC_DIR)
mark_as_advanced(FORCE SFML_DIR)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(${vNameOfExec} PRIVATE Eigen3::Eigen)
mark_as_advanced(FORCE Eigen3_DIR)

# test

include(CTest)
enable_testing()

function(fFilterOfText vText)
    string(REPLACE "[" "\\[" ${vText} ${${vText}})
    string(REPLACE "]" "\\]" ${vText} ${${vText}})
    string(REPLACE "(" "\\(" ${vText} ${${vText}})
    string(REPLACE ")" "\\)" ${vText} ${${vText}})
    string(REPLACE "{" "\\{" ${vText} ${${vText}})
    string(REPLACE "}" "\\}" ${vText} ${${vText}})
    set(${vText} ${${vText}} PARENT_SCOPE)
endfunction(fFilterOfText)

set(vNameOfTest "tFileSystem")
add_test(NAME ${vNameOfTest}
    COMMAND ${vPathToExec} ${vNameOfTest}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ) # add_test
set(vTextOfTest "")
string(APPEND vTextOfTest "[${vNameOfTest}]=(.+")
string(APPEND vTextOfTest "[dPathToInternal]=(${vPathToInternal}).+")
string(APPEND vTextOfTest "[${vPathToInternal}]=(1).+")
string(APPEND vTextOfTest "[dPathToResource]=(${vPathToResource}).+")
string(APPEND vTextOfTest "[${vPathToResource}]=(1).+")
string(APPEND vTextOfTest ")=[${vNameOfTest}]\n")
fFilterOfText(vTextOfTest)
set_tests_properties(${vNameOfTest} PROPERTIES
    PASS_REGULAR_EXPRESSION ${vTextOfTest}
    ) # set_tests_properties

set(vNameOfTest "tTextFormat")
add_test(NAME ${vNameOfTest}
    COMMAND ${vPathToExec} ${vNameOfTest}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ) # add_test
set(vTextOfTest "")
string(APPEND vTextOfTest "[${vNameOfTest}]=(.+")
string(APPEND vTextOfTest "[.+(\"{:.2f}\", M_PI)]=(3.14).+")
string(APPEND vTextOfTest "[.+(\"{:.02f}\", M_PI)]=(3.14).+")
string(APPEND vTextOfTest ")=[${vNameOfTest}]\n")
fFilterOfText(vTextOfTest)
set_tests_properties(${vNameOfTest} PROPERTIES
    PASS_REGULAR_EXPRESSION ${vTextOfTest}
    ) # set_tests_properties

set(vNameOfTest "tMakerOfNetwork")
add_test(NAME ${vNameOfTest}
    COMMAND ${vPathToExec} ${vNameOfTest}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ) # add_test
set(vTextOfTest "")
string(APPEND vTextOfTest "[${vNameOfTest}]=(.+")
string(APPEND vTextOfTest ")=[${vNameOfTest}]\n")
fFilterOfText(vTextOfTest)
set_tests_properties(${vNameOfTest} PROPERTIES
    PASS_REGULAR_EXPRESSION ${vTextOfTest}
    ) # set_tests_properties

# hide

mark_as_advanced(FORCE CMAKE_INSTALL_PREFIX)

mark_as_advanced(FORCE CMAKE_OSX_ARCHITECTURES)
mark_as_advanced(FORCE CMAKE_OSX_DEPLOYMENT_TARGET)
mark_as_advanced(FORCE CMAKE_OSX_SYSROOT)
