# compile cuda mex for shader models 2.0 as default, support double
set(CU_FILE cudaAdd)
set( CMAKE_VERBOSE_MAKEFILE ON)

# set up cuda & matlab libraries
set(CU_FILE_NAME ${CU_FILE}.cu)

set(CUDA_NVCC_FLAGS -Xcompiler -fPIC -use_fast_math ${CUDA_ARCH_FLAGS} --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)
include_directories(${Matlab_INCLUDE_DIRS})
cuda_compile(${CU_FILE} ${CU_FILE_NAME} SHARED)
add_library(${CU_FILE} SHARED ${${CU_FILE}} ${CU_FILE_NAME} ${MATLAB_PROJECT_DIR}/Matlabdef.def)
target_link_libraries(${CU_FILE} ${MATLAB_LIBRARIES} ${CUDA_LIBRARIES})
set_target_properties(${CU_FILE} PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
#set_target_properties(${CU_FILE} PROPERTIES COMPILE_FLAGS "-fvisibility=default")

# 32-bit or 64-bit mex
if(WIN32)
    if (CMAKE_CL_64)
      set_target_properties(${CU_FILE} PROPERTIES SUFFIX .mexw64)
    else(CMAKE_CL_64)
      set_target_properties(${CU_FILE} PROPERTIES SUFFIX .mexw32)
    endif(CMAKE_CL_64)
else(WIN32)
    if (CMAKE_SIZEOF_VOID_P MATCHES "8")
      set_target_properties(${CU_FILE} PROPERTIES SUFFIX .mexa64 PREFIX "")
    else(CMAKE_SIZEOF_VOID_P MATCHES "8")
      set_target_properties(${CU_FILE} PROPERTIES SUFFIX .mexglx PREFIX "")
    endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
endif(WIN32)

# install to /bin by default
install(TARGETS ${CU_FILE} DESTINATION ../bin)

