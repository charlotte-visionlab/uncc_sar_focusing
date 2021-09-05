cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cudaAdd LANGUAGES CXX CUDA)

set( CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA)

#set(CUDA_NVCC_FLAGS -Xcompiler -fPIC -use_fast_math -gencode=arch=compute_20,code="sm_20,compute_20" --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)
set(CUDA_NVCC_FLAGS -Xcompiler -fPIC -use_fast_math ${CUDA_ARCH_FLAGS} --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)
if (Matlab_INCLUDE_DIRS)
    include_directories(
        ${Matlab_INCLUDE_DIRS}
    )

    cuda_compile(${PROJECT_NAME} ${PROJECT_NAME}.cu SHARED)
    add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}} ${PROJECT_NAME}.cu ${MATLAB_PROJECT_DIR}/Matlabdef.def)
    target_link_libraries(${PROJECT_NAME} ${MATLAB_LIBRARIES} ${CUDA_LIBRARIES})
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
endif (Matlab_INCLUDE_DIRS)
