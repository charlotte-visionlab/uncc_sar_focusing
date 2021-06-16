cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CUDABackProjectionKernel LANGUAGES CXX)

set( CMAKE_VERBOSE_MAKEFILE ON)

add_definitions(/DMATLAB_MEX_FILE) #define matlab macros
add_definitions(/DMX_COMPAT_64)
add_definitions(/DMATLAB_DEFAULT_RELEASE=R2019b)
add_definitions(/DUSE_MEX_CMD)

find_package(Matlab REQUIRED)
find_package(CUDA)

add_subdirectory(cpuBackProjection)

if(MATLAB_FOUND AND CUDA_FOUND)
    message(STATUS "CUDA Found, CUDA MEX will be compiled.")
    add_subdirectory(mexcuda)
else(MATLAB_FOUND AND CUDA_FOUND)
    message("MATLAB or CUDA not found... CUDA accelerated GPU code will not be built.")
endif(MATLAB_FOUND AND CUDA_FOUND)

#file(GLOB MATLAB_CODE
#  "../*.m"
#)
#file(COPY ${MATLAB_CODE} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})





