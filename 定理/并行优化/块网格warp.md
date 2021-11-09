**注意:** 当一个blcok被分配给一个SM后，他就只能在这个SM上执行了，不可能重新分配到其他SM上了，多个线程块可以被分配到同一个SM上。

```
#include<stdio.h>
__global__ void hello_world(void)
{
	unsigned int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;
	printf("GPU: blockDim.x = %d,blockIdx.x = %d,threadIdx.x = %d,griddim.x = %d\n", blockDim.x, blockIdx.x, threadIdx.x,gridDim.x);
int main(int argc, char** argv)
{
	printf("CPU: Hello world!\n");
	hello_world << <8, 3 >> > ();
	cudaDeviceReset();//if no this line ,it can not output hello world from gpu
	return 0;
}
```

```
CPU: Hello world!
GPU: blockDim.x = 3,blockIdx.x = 5,threadIdx.x = 0,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 5,threadIdx.x = 1,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 5,threadIdx.x = 2,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 7,threadIdx.x = 0,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 7,threadIdx.x = 1,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 7,threadIdx.x = 2,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 1,threadIdx.x = 0,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 1,threadIdx.x = 1,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 1,threadIdx.x = 2,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 3,threadIdx.x = 0,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 3,threadIdx.x = 1,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 3,threadIdx.x = 2,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 6,threadIdx.x = 0,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 6,threadIdx.x = 1,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 6,threadIdx.x = 2,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 4,threadIdx.x = 0,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 4,threadIdx.x = 1,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 4,threadIdx.x = 2,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 2,threadIdx.x = 0,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 2,threadIdx.x = 1,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 2,threadIdx.x = 2,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 0,threadIdx.x = 0,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 0,threadIdx.x = 1,griddim.x = 8
GPU: blockDim.x = 3,blockIdx.x = 0,threadIdx.x = 2,griddim.x = 8
```

```
#include<stdio.h>
#include<helper_cuda.h>
#include <cuda_runtime.h>
__global__ void hello_world(void)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	printf("thread_id(%d,%d) block_id(%d,%d) coordinate(%d,%d)"
		"global\n", threadIdx.x, threadIdx.y,
		blockIdx.x, blockIdx.y, ix, iy);
}
int main(int argc, char** argv)
{
	printf("CPU: Hello world!\n");
	dim3 block(4, 2);
	dim3 grid(3,3);
	hello_world << <grid, block >> > ();
	cudaDeviceReset();//if no this line ,it can not output hello world from gpu
	return 0;
}
```

```
CPU: Hello world!
thread_id(0,0) block_id(2,1) coordinate(8,2)global
thread_id(1,0) block_id(2,1) coordinate(9,2)global
thread_id(2,0) block_id(2,1) coordinate(10,2)global
thread_id(3,0) block_id(2,1) coordinate(11,2)global
thread_id(0,1) block_id(2,1) coordinate(8,3)global
thread_id(1,1) block_id(2,1) coordinate(9,3)global
thread_id(2,1) block_id(2,1) coordinate(10,3)global
thread_id(3,1) block_id(2,1) coordinate(11,3)global
thread_id(0,0) block_id(1,2) coordinate(4,4)global
thread_id(1,0) block_id(1,2) coordinate(5,4)global
thread_id(2,0) block_id(1,2) coordinate(6,4)global
thread_id(3,0) block_id(1,2) coordinate(7,4)global
thread_id(0,1) block_id(1,2) coordinate(4,5)global
thread_id(1,1) block_id(1,2) coordinate(5,5)global
```

https://face2ai.com/CUDA-F-4-1-%E5%86%85%E5%AD%98%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0/

寄存器对于每个线程是私有的，寄存器通常保存被频繁使用的私有变量，注意这里的变量也一定不能使共有的，不然的话彼此之间不可见，就会导致大家同时改变一个变量而互相不知道，寄存器变量的声明周期和核函数一致，从开始运行到运行结束，执行完毕后，寄存器就不能访问了。
寄存器是SM中的稀缺资源，Fermi架构中每个线程最多63个寄存器。Kepler结构扩展到255个寄存器，一个线程如果使用更少的寄存器，那么就会有更多的常驻线程块，SM上并发的线程块越多，效率越高，性能和使用率也就越高。
那么问题就来了，如果一个线程里面的变量太多，以至于寄存器完全不够呢？这时候寄存器发生溢出，本地内存就会过来帮忙存储多出来的变量，这种情况会对效率产生非常负面的影响，所以，不到万不得已，一定要避免此种情况发生。

=============高性能CUDA应用设计与开发

因为流处理器同一时刻仅能对一个Warp发射一条指令，所以在Kernel中，没有必要再使用大于Warp尺寸的线程。因此有以下代码

```
example6.6 functionReduce.h
if(threadIdx.x < WARP_SIZE)
{
#pragma unroll
	for(int i = threadIdx.x;i < (N_THREADS-WARP_SIZE);i += WARP_SIZE)
	{
		myVal = fcn1(myVal,(T)smem1);
		smem[threadIdx.x] = myVal;
	}
}
```

===============并行算法设计与性能优化

CPU上的线程在运行时会占据一个完整的执行单元，拥有自己的指令指针，因此应尽量使每个线程处理的任务不相同。

而GPU上的多个线程共享一套执行单元，因此应尽量使得一套执行单元上工作的多个线程与其它线程不相关、

===============高性能CUDA应用设计与开发

p73 每个流多处理器可以被调度执行一个或多个线程块。由于线程块内所有的线程都在同一流处理器上执行，GPGPU设计者在流多处理器内部提供称为共享内存的高速内存实现数据共享。

5.5.1

### 寄存器

===============高性能CUDA应用设计与开发 5.5.1

寄存器是GPU上速度最快的存储器，在GPU内存中只有它可以达到足够的带宽和足够低的时延来满足GPU的峰值性能

### 局域内存

===============高性能CUDA应用设计与开发 5.5.2

对自动变量的操作会访问到局域内存。所谓自动变量就是在设备端代码中申请的，不包含device, shared, constant限定符的变量。通常也会放在寄存器中

### 共享内存

===============高性能CUDA应用设计与开发 5.5.4

每个流多处理器可以使用的共享内存为16KB或48KB

### 常量内存

===============高性能CUDA应用设计与开发 5.5.5

常量内存是用来向所有设备线程广播只读数据的理想方式。仅有64KB。当同一WARP中的所有线程都读取同一地址的数据时，常量内存仅需2个时钟就能向流多处理器内的所有WARP广播32位数据。若线程间并发访问常量内存不同地址的数据，则采用串行方式工作。

但是如果数据存在在全局内存中，有const标识符，且与线程ID无关，则也是常量内存相同的效果。

### WarpShuffle

===============The CUDA handbook

The SM 3.0 instruction set added the warp shuffle instruction, which enables
registers to be exchanged within the 32 threads of a warp.   

### 虚拟寻址

CUDA专家手册p18

虚拟寻址能使一个连续的虚拟地址空间映射到物理内存并不连续的页。

GPU并不支持请求式调页，所以被CUDA分配的每一字节虚拟内存都必须对应一个字节的物理内存。

### 统一寻址

CUDA专家手册p23

当 unified virtual addressing 生效时，CUDA会从相同的虚拟地址空间为CPU和GPU分配内存。

### 缓存数据大小

Dataset size
The size of the dataset makes a huge difference as to how a problem can be handled. These fall into
a number of categories on a typical CPU implementation:
• Dataset within L1 cache (~16 KB to 32 KB)
• Dataset within L2 cache (~256 KB to 1 MB)
• Dataset within L3 cache (~512 K to 16 MB)
• Dataset within host memory on one machine (~1 GB to 128 GB)
• Dataset within host-persistent storage (~500 GB to ~20 TB)
• Dataset distributed among many machines (>20 TB)
With a GPU the list looks slightly different:
• Dataset within L1 cache (~16 KB to 48 KB)1
• Dataset within L2 cache (~512 KB to 1536 MB)  

• Dataset within GPU memory (~512 K to 6 GB)
• Dataset within host memory on one machine (~1 GB to 128 GB)
• Dataset within host-persistent storage (~500 GB to ~20 TB)
• Dataset distributed among many machines (>20 TB)  

所以要在CPU上加速，好办法之一就是上多核。16核带来16倍的加速

### 计算密集型

计算密集型的和函数，不是内存友好型，因为核函数在内存访问时花费了很多时间
