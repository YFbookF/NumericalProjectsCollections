#include <cuda_runtime.h>
#include <stdio.h>
__global__ void test_GlobalMemorykernel(float* darray, float* gangle)
{
    int index;
    //calculate each thread global index
    index = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll 10
    for (int loop = 0; loop < 360; loop++)
        darray[index] = darray[index] + gangle[loop];
    return;
}


//declare constant memory
__constant__ float cangle[360];
__global__ void test_ConstantMemorykernel(float* darray)
{
    int index;
    //calculate each thread global index
    index = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll 10
    for (int loop = 0; loop < 360; loop++)
        darray[index] = darray[index] + cangle[loop];
    return;
}


int main(int argc, char** argv)
{
    int size = 3200;
    float* darray;
    float* dhangle;
    float hangle[360];
    int kernel = 1;
    if (argc > 1) (kernel = atoi(argv[1]));
    if (kernel > 2) kernel = 2;
    if (kernel < 1) kernel = 1;

    //initialize angle array on host
    for (int loop = 0; loop < 360; loop++)
        hangle[loop] = acos(-1.0f) * loop / 180.0f;

    if (kernel == 1) //global memory
    {

        //allocate device memory
        cudaMalloc((void**)&darray, sizeof(float) * size);

        //initialize allocated memory
        cudaMemset(darray, 0, sizeof(float) * size);

        cudaMalloc((void**)&dhangle, sizeof(float) * 360);

        //copy host angle data to global memory
        cudaMemcpy(dhangle, hangle, sizeof(float) * 360, cudaMemcpyHostToDevice);
        cudaEvent_t startEvent, stopEvent;
        float etime;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent, 0);
        test_GlobalMemorykernel << <  size / 64, 64 >> > (darray, dhangle);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&etime, startEvent, stopEvent);
        printf(" time %f\n", etime);
        //free device memory
        cudaFree(darray);
        cudaFree(dhangle);
    }
    if (kernel == 2)
    {

        //allocate device memory
        cudaMalloc((void**)&darray, sizeof(float) * size);

        //initialize allocated memory
        cudaMemset(darray, 0, sizeof(float) * size);

        //copy host angle data to constant memory
        cudaMemcpyToSymbol(cangle, hangle, sizeof(float) * 360);
        cudaEvent_t startEvent, stopEvent;
        float etime;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        cudaEventRecord(startEvent, 0);
        test_ConstantMemorykernel << <  size / 64, 64 >> > (darray);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&etime, startEvent, stopEvent);
        printf(" time %f\n", etime);
        //free device memory
        cudaFree(darray);
    }

    return 0;
}

