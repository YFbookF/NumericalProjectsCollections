先看最简单版本的reduction，所执行的操作就是将

```
// Bandwidth: (((2^27) + 1) unsigned ints * 4 bytes/unsigned int)/(173.444 * 10^-3 s)
//  3.095 GB/s = 21.493% -> bad kernel memory bandwidth
__global__ void reduce0(unsigned int* g_odata, unsigned int* g_idata, unsigned int len) {
	extern __shared__ unsigned int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = 0;
	
	if (i < len)sdata[tid] = g_idata[i];
	__syncthreads();

	// do reduction in shared mem
	// Interleaved addressing, which causes huge thread divergence
	//  because threads are active/inactive according to their thread IDs
	//  being powers of two. The if conditional here is guaranteed to diverge
	//  threads within a warp.
	for (unsigned int s = 1; s < 2048; s <<= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) 
		g_odata[blockIdx.x] = sdata[0];
}
```

https://face2ai.com/CUDA-F-3-2-%E7%90%86%E8%A7%A3%E7%BA%BF%E7%A8%8B%E6%9D%9F%E6%89%A7%E8%A1%8C%E7%9A%84%E6%9C%AC%E8%B4%A8-P1/