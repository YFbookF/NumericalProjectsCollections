#include <iostream>

using namespace std;

__global__ void gmem_add(int* a, int n, unsigned int* counter, int* result)
{
	bool finishSum;
	if (threadIdx.x == 0)
	{
		// 基于threadIdx.x进入延时变量
		register int  delay = blockIdx.x * 1000000;
		while (delay > 0)delay--;
		//将blockIdx.x 写入全局内存
		a[blockIdx.x] = blockIdx.x;
		__threadfence();
	}
	// 使用原子加来找到最后完成的流多处理器，记数从0开始
	if (threadIdx.x == 0)
	{
		unsigned int ticket = atomicInc(counter, gridDim.x);
		finishSum = (ticket == gridDim.x - 1);
	}
	if (finishSum)
	{
		register int sum = a[0];
#pragma unroll
		for (int i = 1; i < n; i++)sum += a[i];
		result[0] = sum;
	}
	counter = 0;
}

#define N_BLOCKS 32
int main(int argc, char* argv[])
{
	int* d_a, * d_result;
	unsigned int* d_counter;
	cudaMalloc(&d_a, sizeof(int) * N_BLOCKS);
	cudaMalloc(&d_result, sizeof(int));
	cudaMalloc(&d_counter, sizeof(unsigned int));
	int zero = 0;
	cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);
	gmem_add << <N_BLOCKS, 64 >> > (d_a, N_BLOCKS, d_counter, d_result);
	int h_a[N_BLOCKS], h_result;
	cudaMemcpy(h_a, d_a, sizeof(int) * N_BLOCKS, cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
	int sum = 0;
	for (int i = 0; i < N_BLOCKS; i++)sum += h_a[i];
	cout << "should be " << sum << " got " << h_result << endl;

}