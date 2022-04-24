cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cpuBackProjection LANGUAGES CXX CUDA)

set( CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -DMATLAB_MEX_FILE -g -O0")

if (FFTW_FOUND)
    include_directories(
        ../include
    #    ${FFTW_INCLUDE_DIRS}
    #    ${MATIO_INCLUDE_DIRS}
    )

    add_executable(example_cpu
        example_cpu.cpp)
    target_link_libraries(example_cpu matio lib_uncc_sar_focusing_cpu ${FFTW_FLOAT_LIB} ${FFTW_DOUBLE_LIB} matio)

endif (FFTW_FOUND)

if (CUDA_FOUND)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_DEFAULT_CUDA_ARCHITECTURES})
    endif()

    include_directories(
        ../include
        ${CUDA_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/
    )

    set(CUDA_NVCC_FLAGS -Xcompiler -no-pie -fPIC ${CUDA_ARCH_FLAGS} --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)

    add_executable(example_gpu
        example_gpu.cpp)
    target_link_libraries(example_gpu -no-pie matio lib_uncc_sar_focusing_gpu ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} matio)

endif (CUDA_FOUND)



