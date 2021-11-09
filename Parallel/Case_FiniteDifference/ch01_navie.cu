#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#define TPB 64
//https://github.com/myurtoglu/cudaforengineers
__global__ void ddKernel(float* d_out, const float* d_in, int size, float h) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= size - 1 || i == 0) return;
    d_out[i] = (d_in[i - 1] - 2.f * d_in[i] + d_in[i + 1]) / (h * h);
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

    ddKernel << <(n + TPB - 1) / TPB, TPB >> > (d_out, d_in, n, h);

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
  const int N = 8192;
  const float h = 2 * PI / N;
  
  float x[N] = { 0.0 };
  float u[N] = { 0.0 };
  float result_parallel[N] = { 0.0 };

  for (int i = 0; i < N; ++i) {
    x[i] = 2 * PI*i / N;
    u[i] = sinf(x[i]);
  }

  ddParallel(result_parallel, u, N, h);

  
}