
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>

/*
使用原子操作版本
N = 8192时，91ms
很难说哪个好，因为都差不多
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


template<typename T>
__global__ void
ComputeNBodyGravitation_Atomic(T* force, T* posMass, size_t N, T softeningSquared)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < N;
        i += blockDim.x * gridDim.x)
    {
        float4 me = ((float4*)posMass)[i];
        T acc[3] = { 0.0f, 0.0f, 0.0f };
        T myX = me.x;
        T myY = me.y;
        T myZ = me.z;
        for (int j = 0; j < i; j++) {
            float4 body = ((float4*)posMass)[j];

            T fx, fy, fz;
            bodyBodyInteraction(
                &fx, &fy, &fz,
                myX, myY, myZ,
                body.x, body.y, body.z, body.w,
                softeningSquared);

            acc[0] += fx;
            acc[1] += fy;
            acc[2] += fz;

            float* f = &force[3 * j + 0];
            atomicAdd(f + 0, -fx);
            atomicAdd(f + 1, -fy);
            atomicAdd(f + 2, -fz);

        }

        atomicAdd(&force[3 * i + 0], acc[0]);
        atomicAdd(&force[3 * i + 1], acc[1]);
        atomicAdd(&force[3 * i + 2], acc[2]);
    }
}
float
ComputeGravitation_GPU_Atomic(
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

    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventRecord(evStart, NULL);
    ComputeNBodyGravitation_Atomic<float> << <300, 256 >> > (force, posMass, N, softeningSquared);
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

    ComputeGravitation_GPU_Atomic(dev_force, dev_posMass, 1.0, bodyNum);
    cudaMemcpy(host_force, dev_force, bodyNum * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < bodyNum * 4; i++)
    {
        //printf("force %d = %f\n", i, host_force[i]);
    }
    cudaFree(dev_force);
    cudaFree(dev_posMass);
    return 0;
}

