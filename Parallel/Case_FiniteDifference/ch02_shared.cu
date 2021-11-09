#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#define TPB 4
#define RAD 1
__global__ void ddKernel(float* d_out, const float* d_in, int size, float h) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    printf("blockdimx = %d,blockidxx = %d,threadidxx = %d,i = %d\n",
        blockDim.x, blockIdx.x, threadIdx.x, i);
    const int s_idx = threadIdx.x + RAD;
    extern __shared__ float s_in[];
    printf("d_in[%d] = %f\n", i - RAD, d_in[i - RAD]);
    if (threadIdx.x < RAD)
    {
        s_in[s_idx - RAD] = d_in[i - RAD];//跨线程块，d_in[-1] = 0
        s_in[s_idx + blockDim.x] = d_in[i + blockDim.x];
    }
    __syncthreads();
    d_out[i] = (s_in[s_idx - 1] - 2.f * s_in[s_idx] + s_in[s_idx + 1]) / (h * h);
}

void ddParallel(float* out, const float* in, int n, float h) {
    float* d_in = 0, * d_out = 0;

    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, in, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    float etime;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    const size_t smemSize = (TPB + 2 * RAD) * sizeof(float);
    ddKernel << <(n + TPB - 1) / TPB, TPB ,smemSize>> > (d_out, d_in, n, h);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&etime, startEvent, stopEvent);
    printf(" time %f\n", etime);

    cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}


int main() {
  const float PI = 3.1415927;
  const int N = 16;
  const float h = 2 * PI / N;
  
  float x[N] = { 0.0 };
  float u[N] = { 0.0 };
  float result_parallel[N] = { 0.0 };

  for (int i = 0; i < N; ++i) {
    x[i] = 2 * PI*i / N;
    u[i] = i;
  }

  ddParallel(result_parallel, u, N, h);

  
}