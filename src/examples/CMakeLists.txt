cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cpuBackProjection LANGUAGES CXX CUDA)

set( CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -DMATLAB_MEX_FILE -g -O0")

if (FFTW_FOUND)
    include_directories(
        ../include
        ${FFTW_INCLUDE_DIRS}
    #    ${MATIO_INCLUDE_DIRS}
    )

    add_executable(example_cpu
        example_cpu.cpp)
    target_link_libraries(example_cpu ${FFTW_FLOAT_LIB} ${FFTW_DOUBLE_LIB} matio lib_uncc_sar_focusing_cpu)

endif (FFTW_FOUND)

if (CUDA_FOUND)
    set( CUDA_ARCH_FLAGS "-gencode=arch=compute_35,code=sm_35 \
    -gencode=arch=compute_50,code=sm_50 \
    -gencode=arch=compute_60,code=sm_60 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_80,code=sm_80")

    include_directories(
        ../include
        ${CUDA_INCLUDE_DIRS}
        /usr/local/cuda/samples/common/inc
        /usr/local/cuda/targets/x86_64-linux/include/
    )

    set(CUDA_NVCC_FLAGS -Xcompiler -no-pie -fPIC ${CUDA_ARCH_FLAGS} --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)

    add_executable(example_gpu
        example_gpu.cu)
    target_link_libraries(example_gpu -no-pie ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} matio lib_uncc_sar_focusing_gpu)

endif (CUDA_FOUND)



