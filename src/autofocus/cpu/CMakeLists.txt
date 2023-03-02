cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(autofocus_cpu LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Eigen3 REQUIRED)
#find_package(FFTW)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/../../../../cmake" CACHE STRING "Modules for CMake" FORCE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -DMATLAB_MEX_FILE -g -O0")

include_directories(
        ../include
        ${FFTW_INCLUDE_DIRS}
        #    ${MATIO_INCLUDE_DIRS}
)

add_executable(autofocus_cpu
        autofocus_cpu.cpp
        ../../core/uncc_sar_matio.cpp)
target_link_libraries(autofocus_cpu ${FFTW_FLOAT_LIB} ${FFTW_DOUBLE_LIB} matio Eigen3::Eigen)
#set_target_properties(cpuBackProjection_test PROPERTIES COMPILE_DEFINITIONS "NO_MATLAB=1")


