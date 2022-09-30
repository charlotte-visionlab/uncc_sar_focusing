cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cuda_sar_focusing LANGUAGES CXX CUDA)

set( CMAKE_VERBOSE_MAKEFILE ON)

#set(CUDA_NVCC_FLAGS -Xcompiler -fPIC -use_fast_math -gencode=arch=compute_20,code="sm_20,compute_20" --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)
set(CUDA_NVCC_FLAGS -Xcompiler -no-pie -fPIC --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)

include_directories(
    ../../include
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/
)

add_executable(uncc_sar_focusing_gpu
    cuda_sar_focusing_main.cu
    ../../core/uncc_sar_matio.cpp)
target_link_libraries(uncc_sar_focusing_gpu -no-pie ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} matio)
#set_target_properties(gpuBackProjection_test PROPERTIES COMPILE_DEFINITIONS "NO_MATLAB=1")

add_library(lib_uncc_sar_focusing_gpu SHARED 
    libcuda_sar_focusing.cu 
    ../../core/uncc_sar_matio.cpp
    ../../core/charlotte_sar_api.cpp)
set_target_properties(lib_uncc_sar_focusing_gpu
        PROPERTIES OUTPUT_NAME uncc_sar_focusing_gpu)

if (Matlab_FOUND)

    if (CUDA_FOUND)
        message(STATUS "CUDA and MATLAB Found, CUDA MEX will be compiled.")
    else (CUDA_FOUND)
        message("MATLAB or CUDA not found... CUDA accelerated MATLAB GPU code will not be built.")
    endif (CUDA_FOUND)

    include_directories(
        ${Matlab_INCLUDE_DIRS}
        ${Matlab_ROOT_DIR}/toolbox/parallel/gpu/extern/include
        ${Matlab_ROOT_DIR}/sys/cuda/glnxa64/cuda/include
    )

    add_library(${PROJECT_NAME} SHARED cuda_sar_focusing_mex.cu ${MATLAB_PROJECT_DIR}/Matlabdef.def)
    target_link_libraries(${PROJECT_NAME} ${MATLAB_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
    set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)

    # 32-bit or 64-bit mex
    if(WIN32)
        if (CMAKE_CL_64)
          set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX .mexw64)
        else(CMAKE_CL_64)
          set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX .mexw32)
        endif(CMAKE_CL_64)
    else(WIN32)
        if (CMAKE_SIZEOF_VOID_P MATCHES "8")
          set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX .mexa64 PREFIX "")
        else(CMAKE_SIZEOF_VOID_P MATCHES "8")
          set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX .mexglx PREFIX "")
        endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
    endif(WIN32)

endif (Matlab_FOUND)
