cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cudaBackProjection LANGUAGES CXX CUDA)

set( CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA)

#set(CUDA_NVCC_FLAGS -Xcompiler -fPIC -use_fast_math -gencode=arch=compute_20,code="sm_20,compute_20" --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)
set(CUDA_NVCC_FLAGS -Xcompiler -fPIC ${CUDA_ARCH_FLAGS} --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    /usr/local/cuda/samples/common/inc
    /usr/local/cuda/targets/x86_64-linux/include/
    ${Matlab_INCLUDE_DIRS}
    ${Matlab_ROOT_DIR}/toolbox/parallel/gpu/extern/include
    ${Matlab_ROOT_DIR}/sys/cuda/glnxa64/cuda/include
)

cuda_compile(${PROJECT_NAME} ${PROJECT_NAME}.cu SHARED)
add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}} ${PROJECT_NAME}.cu ${MATLAB_PROJECT_DIR}/Matlabdef.def)
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
