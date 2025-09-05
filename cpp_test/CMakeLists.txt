cmake_minimum_required(VERSION 3.17)

find_package(GTest REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED fftw3f)
# fftw system install doesn't include cmake files
#find_package(FFTW3f REQUIRED)

SET(unittest_files fft_test.cc)

foreach(test_file IN ITEMS ${unittest_files})
    get_filename_component(name ${test_file} NAME_WE)
    add_executable(${name} ${test_file})
    target_link_libraries(${name} GTest::gtest fftr::fftr ${FFTW3F_LIBRARIES})
    target_compile_options(${name} PRIVATE ${PROJECT_WARNINGS})
    add_test(
    	NAME ${name}
        # Pass in a directory for where the test files are.
    	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${name})
endforeach()
