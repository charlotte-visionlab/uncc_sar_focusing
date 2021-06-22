cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cpuBackProjection LANGUAGES CXX)

set( CMAKE_VERBOSE_MAKEFILE ON)

find_package(FFTW REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -DMATLAB_MEX_FILE -g -O0")

include_directories(
    ../include
    ${Matlab_INCLUDE_DIRS}
    ${FFTW_INCLUDE_DIRS}
#    ${MATIO_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED 
    cpuBackProjection_mex.cpp
    cpuBackProjection_fft.cpp
    ${MATLAB_PROJECT_DIR}/Matlabdef.def)
target_link_libraries(${PROJECT_NAME} ${MATLAB_LIBRARIES} ${FFTW_FLOAT_LIB} ${FFTW_DOUBLE_LIB})

add_executable(cpuBackProjection_test
    cpuBackProjection_main.cpp
    cpuBackProjection_fft.cpp)
target_link_libraries(cpuBackProjection_test ${FFTW_FLOAT_LIB} ${FFTW_DOUBLE_LIB} matio)
#set_target_properties(cpuBackProjection_test PROPERTIES COMPILE_DEFINITIONS "NO_MATLAB=1")

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


