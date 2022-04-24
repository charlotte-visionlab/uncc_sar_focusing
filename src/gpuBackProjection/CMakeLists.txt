
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_DEFAULT_CUDA_ARCHITECTURES})
endif()

add_subdirectory(cudaAdd)
add_subdirectory(cuda_sar_focusing)

