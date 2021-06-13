cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CUDABackProjectionKernel LANGUAGES CXX CUDA)

set( CMAKE_VERBOSE_MAKEFILE ON)
set( Matlab_ROOT_DIR "/usr/local/bin/matlab/R2021a")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake" CACHE STRING "Modules for CMake" FORCE)

set(MATLAB_PROJECT_DIR "${CMAKE_CURRENT_LIST_DIR}")

add_definitions(/DMATLAB_MEX_FILE) #define matlab macros
add_definitions(/DMX_COMPAT_64)
add_definitions(/DMATLAB_DEFAULT_RELEASE=R2020a)
add_definitions(/DUSE_MEX_CMD)

find_package(Matlab REQUIRED)
find_package(CUDA REQUIRED)

add_subdirectory(cpuBackProjection)

if(MATLAB_FOUND AND CUDA_FOUND)
    message(STATUS "CUDA Found, CUDA MEX will be compiled.")
    add_subdirectory(mexcuda)
else(MATLAB_FOUND AND CUDA_FOUND)
    message("MATLAB or CUDA not found...nothing will be built.")
endif(MATLAB_FOUND AND CUDA_FOUND)

#file(GLOB MATLAB_CODE
#  "../*.m"
#)
#file(COPY ${MATLAB_CODE} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})





