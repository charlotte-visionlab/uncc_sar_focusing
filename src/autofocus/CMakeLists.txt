cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(autofocus_gpu LANGUAGES CXX CUDA)

find_package(Eigen3 REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -DMATLAB_MEX_FILE -g -O0")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std c++17")

set(CMAKE_DEFAULT_CUDA_ARCHITECTURES 50 52 60 61 75 86)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_DEFAULT_CUDA_ARCHITECTURES})
endif()

if (CUDA_FOUND)
    include_directories(
            ${CUDA_INCLUDE_DIRS}
            ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
            ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/
            /home/cbeam18/.local/lib/python3.8/site-packages/tensorflow/include
    )
    add_executable(autofocus_gpu
            autofocus_gpu.cu)
    target_compile_options(autofocus_gpu PUBLIC --extended-lambda -Xcudafe --diag_suppress=20012)
    target_link_libraries(autofocus_gpu lib_uncc_sar_focusing_gpu matio ${CUDA_CUFFT_LIBRARIES} ${FFTW_FLOAT_LIB} ${FFTW_DOUBLE_LIB} Eigen3::Eigen)

    add_executable(autofocus_gpu_tensorflow
            autofocus_gpu_tensorflow.cu)
    target_compile_options(autofocus_gpu_tensorflow PUBLIC --extended-lambda -Xcudafe --diag_suppress=20012)
    target_link_libraries(autofocus_gpu_tensorflow lib_uncc_sar_focusing_gpu matio ${CUDA_CUFFT_LIBRARIES} ${FFTW_FLOAT_LIB} ${FFTW_DOUBLE_LIB} Eigen3::Eigen /home/cbeam18/.local/lib/python3.8/site-packages/tensorflow/libtensorflow_cc.so.2 /home/cbeam18/.local/lib/python3.8/site-packages/tensorflow/libtensorflow_framework.so.2)
endif (CUDA_FOUND)

add_subdirectory(cpu)
