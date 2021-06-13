set( CMAKE_VERBOSE_MAKEFILE ON)

find_package(FFTW REQUIRED)

# compile cuda mex for shader models 2.0 as default, support double
set(CPP_FILE cpuBackProjection)

# set up cuda & matlab libraries
set(CPP_FILE_NAME ${CPP_FILE}.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -DMATLAB_MEX_FILE -g -O0")

include_directories(
    ${Matlab_INCLUDE_DIRS}
    ${FFTW_INCLUDE_DIRS}
)

add_library(${CPP_FILE} SHARED ${${CPP_FILE}} ${CPP_FILE_NAME} ${MATLAB_PROJECT_DIR}/Matlabdef.def)
target_link_libraries(${CPP_FILE} ${MATLAB_LIBRARIES} ${FFTW_LIBRARIES})
set_target_properties(${CPP_FILE} PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)

# 32-bit or 64-bit mex
if(WIN32)
    if (CMAKE_CL_64)
      set_target_properties(${CPP_FILE} PROPERTIES SUFFIX .mexw64)
    else(CMAKE_CL_64)
      set_target_properties(${CPP_FILE} PROPERTIES SUFFIX .mexw32)
    endif(CMAKE_CL_64)
else(WIN32)
    if (CMAKE_SIZEOF_VOID_P MATCHES "8")
      set_target_properties(${CPP_FILE} PROPERTIES SUFFIX .mexa64 PREFIX "")
    else(CMAKE_SIZEOF_VOID_P MATCHES "8")
      set_target_properties(${CPP_FILE} PROPERTIES SUFFIX .mexglx PREFIX "")
    endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
endif(WIN32)

# install to /bin by default
install(TARGETS ${CU_FILE} DESTINATION ../bin)

