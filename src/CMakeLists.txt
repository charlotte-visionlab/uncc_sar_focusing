cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

add_definitions(/DMATLAB_MEX_FILE) #define matlab macros
add_definitions(/DMX_COMPAT_64)
add_definitions(/DMATLAB_DEFAULT_RELEASE=R2019b)
add_definitions(/DUSE_MEX_CMD)

find_package(Matlab)
find_package(CUDA)

add_subdirectory(cpuBackProjection)

if(MATLAB_FOUND AND CUDA_FOUND)
    message(STATUS "CUDA Found, CUDA MEX will be compiled.")
    add_subdirectory(gpuBackProjection)
else(MATLAB_FOUND AND CUDA_FOUND)
    message("MATLAB or CUDA not found... CUDA accelerated GPU code will not be built.")
endif(MATLAB_FOUND AND CUDA_FOUND)