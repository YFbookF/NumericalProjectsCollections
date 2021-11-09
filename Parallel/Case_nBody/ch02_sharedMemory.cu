
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>

/*
Copyright (c) 2011-2012, Archaea Software, LLC.
使用共享内存版本
N = 8192时，unroll 32 时间为80.87ms
i 遍历网格，j遍历线程块，k遍历线程
*/
template <typename T>
__host__ __device__ void bodyBodyInteraction(
    T* fx, T* fy, T* fz,
    T x0, T y0, T z0,
    T x1, T y1, T z1, T mass1,
    T softeningSquared)
{
    T dx = x1 - x0;
    T dy = y1 - y0;
    T dz = z1 - z0;

    T distSqr = dx * dx + dy * dy + dz * dz;
    distSqr += softeningSquared;

    //
    // rsqrtf() maps to SFU instruction - to support
    // double, this has to be changed.
    //
    T invDist = rsqrtf(distSqr);

    T invDistCube = invDist * invDist * invDist;
    T s = mass1 * invDistCube;

    *fx = dx * s;
    *fy = dy * s;
    *fz = dz * s;
}


__global__ void
ComputeNBodyGravitation_Shared(
    float* force,
    float* posMass,
    float softeningSquared,
    size_t N)
{
    extern __shared__ float4 shPosMass[];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < N;
        i += blockDim.x * gridDim.x)
    {
        float acc[3] = { 0 };
        float4 myPosMass = ((float4*)posMass)[i];
#pragma unroll 32
        for (int j = 0; j < N; j += blockDim.x) {
			//既然共享内存仅限于同一个线程块，我们就采用一个分块的策略，
			//不同天体计算各自的加速度之间是有数据重用的，先将天体数据预先加载到共享内存，
			//同一个线程块的线程访问这个共享内存做相应的数值计算。但是又引入一个问题，共享内存是有限的，
			//所以对于一个线程块的共享内存，我们不是一次性地将所有天体数据从全局内存加载到共享内存，
			//而是分批次地加载、计算。如下图3所示，设一个线程块有pp个线程，那么NN个天体的引力计算被分到N/pN/p个线程块，
			//每个线程块分配p个天体数据大小的共享内存，分N/PN/P批次串行计算。
			//每次计算前，在同一个线程块内，各个线程相应地从全局内存加载各自对应的天体数据（通过线程id和线程块id、size计算索引）到共享内存，
			//加载完毕之后各自迭代pp次计算当前批次的天体引力，如下进行下去。
            shPosMass[threadIdx.x] = ((float4*)posMass)[j + threadIdx.x];
            __syncthreads();
            for (size_t k = 0; k < blockDim.x; k++) {
                float fx, fy, fz;
                float4 bodyPosMass = shPosMass[k];

                bodyBodyInteraction<float>(
                    &fx, &fy, &fz,
                    myPosMass.x, myPosMass.y, myPosMass.z,
                    bodyPosMass.x,
                    bodyPosMass.y,
                    bodyPosMass.z,
                    bodyPosMass.w,
                    softeningSquared);
                acc[0] += fx;
                acc[1] += fy;
                acc[2] += fz;
            }
            __syncthreads();
        }
        force[3 * i + 0] = acc[0];
        force[3 * i + 1] = acc[1];
        force[3 * i + 2] = acc[2];
    }
}
float
ComputeGravitation_GPU_Shared(
    float* force,
    float* posMass,
    float softeningSquared,
    size_t N
)
{
    cudaError_t status;
    cudaEvent_t evStart = 0, evStop = 0;
    float ms = 0.0;
    size_t bodiesLeft = N;

    void* p;
    cudaGetSymbolAddress(&p, g_constantBodies);

    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventRecord(evStart, NULL);
    ComputeNBodyGravitation_Shared << <300, 256, 256 * sizeof(float4) >> > (
        force,
        posMass,
        softeningSquared,
        N);
    cudaEventRecord(evStop, NULL);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, evStart, evStop);
    printf(" time %fms\n", ms);
Error:
    cudaEventDestroy(evStop);
    cudaEventDestroy(evStart);
    return ms;
}

int main()
{

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return -1;
    }

    int bodyNum = 8192;
    float* host_force = (float*)std::malloc(bodyNum * sizeof(float) * 4);
    float* host_posMass = (float*)std::malloc(bodyNum * sizeof(float) * 4);
    for (int i = 0; i < bodyNum * 4; i++)
    {
        host_force[i] = 0;
        host_posMass[i] = i;
    }
    float* dev_force = 0;
    float* dev_posMass = 0;
    cudaMalloc((void**)&dev_force, bodyNum * 4 * sizeof(float));
    cudaMalloc((void**)&dev_posMass, bodyNum * 4 * sizeof(float));
    cudaMemcpy(dev_force, host_force, bodyNum * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_posMass, host_posMass, bodyNum * 4 * sizeof(float), cudaMemcpyHostToDevice);

    ComputeGravitation_GPU_Shared(dev_force, dev_posMass, 1.0, bodyNum);
    cudaMemcpy(host_force, dev_force, bodyNum * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < bodyNum * 4; i++)
    {
        //printf("force %d = %f\n", i, host_force[i]);
    }
    cudaFree(dev_force);
    cudaFree(dev_posMass);
    return 0;
}

