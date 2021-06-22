/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   cufftShift_1D_IP.cu
 * Author: arwillis
 *
 * Created on June 21, 2021, 9:26 AM
 */

#ifndef CUFFTSHIFT_1D_IP_CU
#define CUFFTSHIFT_1D_IP_CU

/*********************************************************************
 * Copyright © 2011-2014,
 * Marwan Abdellah: <abdellah.marwan@gmail.com>
 *
 * This library (cufftShift) is free software; you can redistribute it
 * and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 ********************************************************************/
 
#ifndef CUFFTSHIFT_1D_SINGLE_ARRAY_CU
#define CUFFTSHIFT_1D_SINGLE_ARRAY_CU

#include <cuda.h>
#include <cutil_inline.h>

unsigned int ulog2 (unsigned int u)
{
    unsigned int s, t;

    t = (u > 0xffff) << 4; u >>= t;
    s = (u > 0xff  ) << 3; u >>= s, t |= s;
    s = (u > 0xf   ) << 2; u >>= s, t |= s;
    s = (u > 0x3   ) << 1; u >>= s, t |= s;

    return (t | (u >> 1));
}

template <typename T>
void cufftShift_1D(T* data, int NX)
{
    int threadsPerBlock_X = (NX > 1) ? std::min(ulog2((unsigned int) NX), 1024) : 1;   
    grid = dim3((N / threadsPerBlock_X), 1, 1);
    block = dim3(threadsPerBlock_X, 1, 1);;
    cufftShift_1D_kernel <<< grid, block >>> (data, NX);
}

template <typename T>
__global__
void cufftShift_1D_kernel(T* data, int NX)
{
    int threadIdxX = threadIdx.x;
    int blockDimX = blockDim.x;
    int blockIdxX = blockIdx.x;

    int index = ((blockIdxX * blockDimX) + threadIdxX);
    if (index < NX/2)
    {
        // Save the first value
        T regTemp = data[index];

        // Swap the first element
        data[index] = (T) data[index + (NX / 2)];

        // Swap the second one
        data[index + (NX / 2)] = (T) regTemp;
    }
}

#endif // CUFFTSHIFT_1D_SINGLE_ARRAY_CU

#endif /* CUFFTSHIFT_1D_IP_CU */

