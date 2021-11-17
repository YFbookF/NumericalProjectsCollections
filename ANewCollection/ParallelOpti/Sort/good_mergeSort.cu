#include <stdio.h>
#include <iostream>
#include <limits>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
typedef unsigned int uint;
#define BLOCK_DIM 64
template<typename T>
//https://github.com/longzeyilang/sort_cuda/blob/main/sort.cu
__global__ void kernelSortBlock(const uint size, T* g_data)
{
    __shared__ T s_data[BLOCK_DIM << 1];
    const uint idx = threadIdx.x << 1;
    const uint offset = (blockDim.x * blockIdx.x) << 1;
    const uint range = (offset + (BLOCK_DIM << 1) > size ? (size - offset) : (BLOCK_DIM << 1));
    uint fst_idx = idx;
    uint snd_idx = idx + 1;

    if (offset + snd_idx > size)
        return;

    s_data[fst_idx] = g_data[offset + fst_idx];
    s_data[snd_idx] = g_data[offset + snd_idx];

    __syncthreads();

    bool shift = 1;
    T fst_val, snd_val;
    for (uint k = 0; k <= range; k++) {
        shift = !shift;
        fst_idx = idx + shift;
        snd_idx = fst_idx + 1;
        if (snd_idx < range) {
            fst_val = s_data[fst_idx];
            snd_val = s_data[snd_idx];
            if (snd_val < fst_val) {
                s_data[fst_idx] = snd_val;
                s_data[snd_idx] = fst_val;
            }
        }
        __syncthreads();
    }

    g_data[offset + idx] = s_data[idx];
    g_data[offset + idx + 1] = s_data[idx + 1];
}

template<typename T>
__global__ void kernelMergeBlocks(const uint size, const uint total_size, const T* g_in, T* g_out)
{
    const uint g_idx = (blockDim.x * blockIdx.x + threadIdx.x) * size;
    if (g_idx >= total_size)
        return;

    uint lhs_idx = g_idx;
    uint rhs_idx = g_idx + (size >> 1);
    uint out_idx = g_idx;

    const uint lhs_lmt = (rhs_idx > total_size ? total_size : rhs_idx);
    const uint rhs_lmt = (rhs_idx + (size >> 1) > total_size ? total_size : rhs_idx + (size >> 1));

    while (lhs_idx < lhs_lmt && rhs_idx < rhs_lmt) {
        const uint lhs_val = g_in[lhs_idx];
        const uint rhs_val = g_in[rhs_idx];
        if (lhs_val > rhs_val) {
            g_out[out_idx] = rhs_val;
            rhs_idx++;
        }
        else {
            g_out[out_idx] = lhs_val;
            lhs_idx++;
        }
        out_idx++;
    }
    while (lhs_idx < lhs_lmt) {
        g_out[out_idx] = g_in[lhs_idx];
        out_idx++;
        lhs_idx++;
    }
    while (rhs_idx < rhs_lmt) {
        g_out[out_idx] = g_in[rhs_idx];
        out_idx++;
        rhs_idx++;
    }
}

//cuda sort
template<typename T> void cudaSort(uint size, T* data)
{
    T* d_data, * d_tmp;
    cudaMalloc(&d_data, (size + 1) * sizeof(T));
    cudaMalloc(&d_tmp, (size + 1) * sizeof(T));
    cudaMemcpy(d_data, data, size * sizeof(T), cudaMemcpyHostToDevice);

    uint sort_blocks_no = (((size + BLOCK_DIM - 1) / BLOCK_DIM) + 1) >> 1;
    printf("\n sort_block = %d\n", sort_blocks_no);
    kernelSortBlock << <sort_blocks_no, BLOCK_DIM >> > (size, d_data);
    cudaDeviceSynchronize();

    uint merge_size = BLOCK_DIM << 2;
    const uint twice_block_dim = BLOCK_DIM << 1;

    while (sort_blocks_no > 1) {
        const uint merge_blocks_no = (sort_blocks_no + twice_block_dim - 1) / twice_block_dim;

        kernelMergeBlocks << <merge_blocks_no, BLOCK_DIM >> > (merge_size, size, d_data, d_tmp);
        cudaDeviceSynchronize();

        sort_blocks_no = (sort_blocks_no + 1) >> 1;
        merge_size <<= 1;
        std::swap(d_data, d_tmp);
    }

    cudaMemcpy(data, d_data, size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_tmp);
}


//sort 
// dev_points: src array to be sorted
// sorted_index: output index
//num_points : the size to be sorted
__global__ void rank_kernel(float* dev_points, int* sorted_index, const int num_points) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_points) return;
    int count = 0;
    float aid = dev_points[id];
    for (int i = 0; i < num_points; i++) {
        float ai = dev_points[i];
        if (ai < aid || (i < id && ai == aid)) count++;
    }
    sorted_index[id] = count;  //index
}

float arand(float i)
{
    return (sinf(i * 0.67) + cosf(i * 0.54)) * 0.25 + 0.5;
}


#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
void main() {
    const int num_points = 10;
    float A[num_points] = { 0 };
    int index[num_points] = { 0 };
    for (int i = 0; i < num_points; i++)
    {
        A[i] = arand((float)i);
     
        printf("%f ", A[i]);
    }
    cudaSort(num_points, A);
    printf("\n after sort \n");
    for (int i = 0; i < num_points; i++)
    {
        printf("%f ", A[i]);
    }
    /*
    float* dev_points_ptr = nullptr;
    cudaMalloc((void**)&dev_points_ptr, num_points * sizeof(float));  // x,y,z,intensity
    cudaMemcpy(dev_points_ptr, A, num_points * sizeof(float), cudaMemcpyHostToDevice);

    int* sorted_index = nullptr;
    cudaMalloc((void**)&sorted_index, num_points * sizeof(int));  // x,y,z,intensity
    int num_block = DIVUP(num_points, 128);
    int m_num_threads_ = 128;
    rank_kernel << <num_block, m_num_threads_ >> > (dev_points_ptr, sorted_index, num_points);
    printf("\n after sort \n");
    cudaMemcpy(index, sorted_index, num_points * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_points; i++)
    {
        printf("%f ", A[index[i]]);
    }
    cudaFree(dev_points_ptr);
    cudaFree(sorted_index);
    */
}