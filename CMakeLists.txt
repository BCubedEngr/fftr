########################################################################
# Project setup
########################################################################

cmake_minimum_required(VERSION 3.16)
# Only specify the version in one place inside of the conanfile.
file(READ Version VERSION_NUMBER)
string(STRIP "${VERSION_NUMBER}" VERSION_NUMBER)
message(STATUS "Reading version from Version file : version = ${VERSION_NUMBER}")

project(fftr LANGUAGES CXX VERSION ${VERSION_NUMBER})

SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_C_STANDARD 17)

enable_testing()

include(GNUInstallDirs)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, e.g. Debug, Release, RelWithDebInfo, MinSizeRel." FORCE)
  # Set the property on the cache entry so that it shows up in the CMake GUI
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lower )

if(build_type_lower MATCHES debug)
    message(STATUS "Building with coverage enabled")
    SET(CMAKE_CXX_FLAGS "-g -O0 -Wall -fprofile-arcs -ftest-coverage")
    SET(CMAKE_C_FLAGS "-g -O0 -Wall -fprofile-arcs -ftest-coverage")
    SET(CMAKE_EXE_LINGER_FLAGS "-g -O0 -Wall -fprofile-arcs -ftest-coverage")
endif(build_type_lower MATCHES debug)

SET(PROJECT_WARNINGS
    -Werror # Convert Warnings to errors
    -Wall
    -Wextra # reasonable and standard
    -Wshadow # warn the user if a variable declaration shadows one from a parent context
    -Wnon-virtual-dtor # warn the user if a class with virtual functions has a non-virtual destructor. This helps
                       # catch hard to track down memory errors
    -Wold-style-cast # warn for c-style casts
    -Wcast-align # warn for potential performance problem casts
    -Wunused # warn on anything being unused
    -Woverloaded-virtual # warn if you overload (not override) a virtual function
    -Wpedantic # warn if non-standard C++ is used
    -Wsign-conversion # warn on sign conversions
    -Wnull-dereference # warn if a null dereference is detected
    -Wformat=1 # warn on security issues around functions that format output (ie printf)
    -Wno-unknown-pragmas # ignore IDE, GCC/CLANG specific pragmas
    -Wimplicit-fallthrough # Warns when case statements fall-through.)
    -Wmisleading-indentation # warn if indentation implies blocks where blocks do not exist
    -Wduplicated-cond # warn if if / else chain has duplicated conditions
    -Wduplicated-branches # warn if if / else branches have duplicated code
    -Wlogical-op # warn about logical operations being used where bitwise were probably wanted
    -Wuseless-cast # warn if you perform a cast to the same type
    -Wno-maybe-uninitialized # false positives if asan is enabled: https://gcc.gnu.org/bugzilla//show_bug.cgi?id=1056h6
    -fconcepts-diagnostics-depth=3
)

# Make sure our local CMake Modules path comes first
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(VERSION_PATCH git)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

cmake_policy(SET CMP0011 NEW)

# Enable generation of compile_commands.json for code completion engines
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

########################################################################
# Setup doxygen option
########################################################################
if(DOXYGEN_FOUND)
    option(ENABLE_DOXYGEN "Build docs using Doxygen" ON)
else(DOXYGEN_FOUND)
    option(ENABLE_DOXYGEN "Build docs using Doxygen" OFF)
endif(DOXYGEN_FOUND)


########################################################################
# Find deps
########################################################################

########################################################################
# Create Rust Library
########################################################################
if(${BUILD_SHARED_LIBS})
    SET(LIB_FILE libfftr.so)
else()
    SET(LIB_FILE libfftr.a)
endif()

# Always build the release version because this is just a wrapper around rustfft.
# We will build a debug version of our test code if requested.
find_program(CARGO cargo REQUIRED
             # I don't know anywhere else it would be installed
             HINTS $ENV{HOME}/.cargo/bin)
if(${build_type_lower} STREQUAL release)
    SET(RUST_PROFILE release)
else()
    SET(RUST_PROFILE dev)
endif()
add_custom_target(${PROJECT_NAME}_rustlib ALL
                   BYPRODUCTS ${LIB_FILE}
                   COMMAND ${CARGO} build --profile ${RUST_PROFILE}
                   SOURCES Cargo.toml src/lib.rs
                   )
install(FILES target/${build_type_lower}/${LIB_FILE}
        DESTINATION lib)


########################################################################
# Create Target
########################################################################
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
        INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_directories(${PROJECT_NAME}
        INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/target/${build_type_lower}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_LIBDIR}>)


target_link_libraries(${PROJECT_NAME} INTERFACE ${LIB_FILE})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_rustlib)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}_Targets)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

########################################################################
# Add subdirectories
########################################################################

add_subdirectory(include/fftr)

add_subdirectory(cpp_test)

########################################################################
# Setup CMake Config files
########################################################################
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

file(WRITE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    "include(CMakeFindDependencyMacro)\n
    include(\${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}Targets.cmake)\n"
)

install(EXPORT ${PROJECT_NAME}_Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME})

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
    DESTINATION lib/cmake/${PROJECT_NAME})
