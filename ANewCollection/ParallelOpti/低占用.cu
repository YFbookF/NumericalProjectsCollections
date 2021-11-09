#include <omp.h>
#include<iostream>
using namespace std;
#include <cmath>

__device__ float d_a[32], d_d[32], d_e[32], d_f[32];

#define NUM_ITERATIONS (1024 * 1024)
#define ILP4
#ifdef  ILP4
// 指令级并行
#define OP_COUNT 4*2*NUM_ITERATIONS

__global__ void kernel(float a, float b, float c)
{
	register float d = a, e = a, f = a;
#pragma unroll 16
	for (int i = 0; i < NUM_ITERATIONS; i++)
	{
		a = a * b + c;
		d = d * b + c;
		e = e * b + c;
		f = f * b + c;
	}
	// 向全局内存写入，以使上述工作不会被编译器优化
	d_a[threadIdx.x] = a;
	d_d[threadIdx.x] = d;
	d_e[threadIdx.x] = e;
	d_f[threadIdx.x] = f;

}
#else
// 线程级并行
#define OP_COUNT 1*2*NUM_ITERATIONS
__global__ void kernel(float a, float b, float c)
{
#pragma unroll 16
	for (int i = 0; i < NUM_ITERATIONS; i++)
	{
		a = a * b + c;
	}
	// 向全局内存写入，以使上述工作不会被编译器优化
	d_a[threadIdx.x] = a;

}

#endif 
int main()
{
	for (int nThreads = 32; nThreads <= 1024; nThreads += 32)
	{
		double start = omp_get_wtime();
		kernel << <1, nThreads >> > (1., 2., 3.);
		if (cudaGetLastError() != cudaSuccess)
		{
			cerr << "L Error" << endl;
			return (1);
		}
		cudaThreadSynchronize();
		double end = omp_get_wtime();
		cout << "warp" << ceil(nThreads / 32) << " "
			<< nThreads << " " << (nThreads * (OP_COUNT / 1.e9) / (end - start))
			<< "Gflops" << endl;
	}
	return (0);
}