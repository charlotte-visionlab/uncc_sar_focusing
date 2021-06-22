
set( CUDA_ARCH_FLAGS "-gencode=arch=compute_35,code=sm_35 \
-gencode=arch=compute_50,code=sm_50 \
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_70,code=sm_70 \
-gencode=arch=compute_80,code=sm_80")

add_subdirectory(cudaAdd)
add_subdirectory(cuda_sar_focusing)

