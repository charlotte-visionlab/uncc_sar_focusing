cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

find_package(Matlab)
find_package(CUDA)
find_package(FFTW)

if (MATLAB_FOUND)
    add_definitions(/DMATLAB_MEX_FILE) #define matlab macros
    add_definitions(/DMX_COMPAT_64)
    add_definitions(/DMATLAB_DEFAULT_RELEASE=R2019b)
    add_definitions(/DUSE_MEX_CMD)
endif (MATLAB_FOUND)

if (FFTW_FOUND)
    add_subdirectory(cpuBackProjection)
endif (FFTW_FOUND)

if (CUDA_FOUND)
    add_subdirectory(gpuBackProjection)
#    add_subdirectory(gpu)
endif (CUDA_FOUND)

add_subdirectory(examples)