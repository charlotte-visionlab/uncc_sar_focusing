cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(autofocus_gpu LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -DMATLAB_MEX_FILE -g -O0")

set(CMAKE_DEFAULT_CUDA_ARCHITECTURES 50 52 60 61 75 86)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_DEFAULT_CUDA_ARCHITECTURES})
endif()

if (CUDA_FOUND)
    include_directories(
            ${CUDA_INCLUDE_DIRS}
            ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
            ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/
    )
    add_executable(autofocus_gpu
            autofocus_gpu.cu)
    target_compile_options(autofocus_gpu PUBLIC --extended-lambda)
    target_link_libraries(autofocus_gpu lib_uncc_sar_focusing_gpu matio ${CUDA_CUFFT_LIBRARIES})
endif (CUDA_FOUND)
