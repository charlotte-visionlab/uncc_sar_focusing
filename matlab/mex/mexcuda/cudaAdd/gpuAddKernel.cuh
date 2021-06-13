/************************************************************************
 Sample CUDA MEX kernel code written by Fang Liu (leoliuf@gmail.com).
 ************************************************************************/

#ifndef _ADD_KERNEL_GPU_H_
#define _ADD_KERNEL_GPU_H_

#define mexPrintf printf

__global__ void
gpuAddKernel(double *d_A, double *d_B, double *d_C, mwSignedIndex Am, mwSignedIndex An) {
    /* index */
        unsigned int tid = blockIdx.x * blockDim.y + threadIdx.y; /* thread id in matrix*/
    /* strip */
        unsigned int strip = gridDim.x * blockDim.y;


//    int i = threadIdx.x;
//    int j = threadIdx.y;
//    int index = j * An + i;
//    if (index < Am * An) {
//        printf("filled position (%d,%d)\n", j, i);
//        d_C[index] = d_A[index] + d_B[index];
//    }
        while (1) {
            if (tid < Am * An) {
                printf("filled position (%d)\n",tid);
                d_C[tid] = d_A[tid] + d_B[tid];
            } else {
                break;
            }
            tid += strip;
        }
}

#endif