cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cpuBackProjection LANGUAGES CXX CUDA)

set( CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -DMATLAB_MEX_FILE -g -O0")

if (FFTW_FOUND)
    include_directories(
        ../include
    )

    add_executable(example_cpu
        example_cpu.cpp)
    target_link_libraries(example_cpu lib_uncc_sar_focusing_cpu matio ${FFTW_FLOAT_LIB} ${FFTW_DOUBLE_LIB})

endif (FFTW_FOUND)

if (CUDA_FOUND)
    include_directories(
        ../include
    )

    add_executable(example_gpu
        example_gpu.cpp)
    target_link_libraries(example_gpu lib_uncc_sar_focusing_gpu matio ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})

endif (CUDA_FOUND)



