当执行矩阵乘法时，需要执行下面的流程

![image-20210927215954037](D:\图形学书籍\系列流体文章\gif\image-20210927215954037.png)

有两种方式，显然前者不如后者。

![image-20210927220052409](D:\图形学书籍\系列流体文章\gif\image-20210927220052409.png)

![image-20210927220102122](D:\图形学书籍\系列流体文章\gif\image-20210927220102122.png)

为了提高每瓦特的计算性能，必须Heterogeneousness ，也就是高频率CPU执行串行，而低频率CPU执行并行部分。

For example, twodimensional array A[i][j] in C language has a direction of continues access in the j-direction. On the other hand, two-dimensional array A(i, j) in Fortran language has a direction of continues access in the i-direction.  

For example, in the following code in Fortran:
do i=1, n
A(1, i) = b(i) * c(i)
enddo  

确实是这样，有差别，大约15%？python 没区别，内部做了什么优化？

```
#include <iostream>
#include <time.h>　
using namespace std;
//https://blog.csdn.net/mxclxp/article/details/7991127
int main()
{
	const int msize = 200;
	int arr[msize][msize];
	clock_t start, end;
	start = clock();
	for (int k = 0; k < 10000;k++)
	{
		for (int i = 0; i < msize; i++)
		{
			for (int j = 0; j < msize; j++)
			{
				arr[j][i] = 100; // 1153ms
				//arr[i][j] = 100; // 986ms
			}
		}
	}
	end = clock();
	cout << "time = " << double(end - start)*1000 / CLOCKS_PER_SEC << "ms" << endl;
	return 0;
```

但是它说middleProductForm有最好性能，我表示很怀疑，循环也需要时间啊，InnerProduct2000ms，MiddleProductForm300ms。

```
// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <time.h>　
using namespace std;
//https://blog.csdn.net/mxclxp/article/details/7991127
const int msize = 200;
const int looptime = 100;
int C[msize][msize];
int A[msize][msize], B[msize][msize];

void InnerProductForm()
{
	int At = 0;
	for (int n = 0; n < looptime; n++)
	{
		for (int i = 0; i < msize; i++)
		{
			for (int j = 0; j < msize; j++)
			{
				At = 0;
				for (int k = 0; k < msize; k++)
				{
				// 如果改成
				// C[i][j] = C[i][j] + A[i][k] * B[k][j];
				// 就跑不过了
					At = At + A[i][k] * B[k][j];
				}
				C[i][j] = At;
			}
		}
	}
}

void MiddleProductForm()
{
	int At = 0;
	for (int n = 0; n < looptime; n++)
	{
		for (int i = 0; i < msize; i++)
		{
			for (int k = 0; k < msize; k++)
			{
				At = A[i][k];
				for (int j = 0; j < msize; j++)
				{
					C[i][j] = C[i][j] + At * B[k][j];
				}
			}
		}
	}
}

int main()
{
	

	clock_t start, end;
	
	for (int i = 0; i < msize; i++)
	{
		for (int j = 0; j < msize; j++)
		{
			A[i][j] = B[i][j] = 1;
			C[i][j] = 0;
		}
	}
	start = clock();
	MiddleProductForm();
	end = clock();
	cout << "time = " << double(end - start)*1000 / CLOCKS_PER_SEC << "ms" << endl;
	start = clock();
	InnerProductForm();
	end = clock();
	cout << "time = " << double(end - start) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
	return 0;
}

```

![image-20210927223801659](D:\图形学书籍\系列流体文章\gif\image-20210927223801659.png)

![image-20210927224722511](D:\图形学书籍\系列流体文章\gif\image-20210927224722511.png)

按连续顺序访问的话fortran的话，line0只会被放到内存中一次。但现在A(4,4)，也就是line0会被循环放到的内存中四次，这和没有缓存是没有区别的。

unroll确实有效果，分别是200ms和300ms，大约10%？

```
#include <iostream>
#include <time.h>　
using namespace std;
//https://blog.csdn.net/mxclxp/article/details/7991127
const int msize = 20;
const int looptime = 10000;
const int fsize = msize / 2;
int C[msize][msize];
int A[msize][msize], B[msize][msize];

void Pro1()
{
	for (int n = 0; n < looptime; n++)
	{
		for (int i = 0; i < fsize; i++)
		{
			for (int j = 0; j < fsize; j++)
			{
				for (int k = 0; k < msize; k++)
				{
					C[i][j] = C[i][j]  + A[i][k] * B[k][j];
					C[i+1][j] = C[i+1][j] + A[i+1][k] * B[k][j];
					C[i][j+1] = C[i][j+1] + A[i][k] * B[k][j+1];
					C[i+1][j+1] = C[i+1][j+1] + A[i+1][k] * B[k][j+1];
				}
			}
		}
	}
}

void Pro2()
{
	for (int n = 0; n < looptime; n++)
	{
		for (int i = 0; i < msize; i++)
		{
			for (int j = 0; j < msize; j++)
			{
				for (int k = 0; k < msize; k++)
				{
					C[i][j] = C[i][j] + A[i][k] * B[k][j];
				}
			}
		}
	}
}

int main()
{
	

	clock_t start, end;
	
	for (int i = 0; i < msize; i++)
	{
		for (int j = 0; j < msize; j++)
		{
			A[i][j] = B[i][j] = 1;
			C[i][j] = 0;
		}
	}
	start = clock();
	Pro1();
	end = clock();
	cout << "time = " << double(end - start)*1000 / CLOCKS_PER_SEC << "ms" << endl;
	start = clock();
	Pro2();
	end = clock();
	cout << "time = " << double(end - start) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
	return 0;
}
```

Since whole matrix–matrix multiplications are divided by multiple small matrix–matrix multiplications, the number of
cache miss–hits can be reduced.   

block 乘法确实有效，但最好的ibl很难确定。大概能有5%的加速，并且不能和unloop混合使用，因为它们本来就是一样的

```
const int msize = 256;
const int looptime = 100;
const int lsize = 8;
int C[msize][msize];
int A[msize][msize], B[msize][msize];

void Pro1()
{
	int At = 0;
	for (int n = 0; n < looptime; n++)
	{
		for (int i = 0; i < msize; i+= lsize)
		{
			for (int j = 0; j < msize; j+= lsize)
			{
				At = 0;
				for (int k = 0; k < msize; k+=lsize)
				{
					for (int i0 = i; i0 < i + lsize; i0++)
					{
						for (int j0 = j; j0 < j + lsize; j0++)
						{
							for (int k0 = k; k0 < k + lsize; k0++)
							{
								At = At + A[i0][k0] * B[k0][j0];
							}
						}
					}
				}
				C[i][j] = At;
			}
		}
	}
}

void Pro2()
{
	int At = 0;
	for (int n = 0; n < looptime; n++)
	{
		for (int i = 0; i < msize; i++)
		{
			for (int j = 0; j < msize; j++)
			{
				At = 0;
				for (int k = 0; k < msize; k++)
				{
					At = At + A[i][k] * B[k][j];
				}
				C[i][j] = At;
			}
		}
	}
}

```

The best size of ibl in Fig. 1.6 depends on computer architectures where the
total amount of lines on cache affects the best size of ibl.   

测试2，说明三维向量效率最高，但是并行和openmp和cuda和eigen还有待商榷加速效果约是0%~10%

```
const int msize = 128;
const int looptime = 100;
const int lsize = 8;
int C[msize][msize][msize];
int D[msize * msize * msize];
//msize = 128 looptime = 100 538ms 612ms 601ms
//msize = 512 looptime = 10 3206ms 3331ms 3332ms
//msize = 16 looptime = 100000 997ms 1043ms 1023ms
//msize = 4 looptime = 10000000 1931ms 1426ms 1443ms
void Pro1()
{
	int idx = 0;
	for (int n = 0; n < looptime; n++)
	{
		idx = 0;
		for (int i = 0; i < msize; i++)
		{
			for (int j = 0; j < msize; j++)
			{
				for (int k = 0; k < msize; k++)
				{
					D[idx] = 1;
					idx += 1;
				}
			}
		}
	}
}

void Pro2()
{
	for (int n = 0; n < looptime; n++)
	{
		for (int i = 0; i < msize; i++)
		{
			for (int j = 0; j < msize; j++)
			{
				for (int k = 0; k < msize; k++)
				{
					C[i][j][k] = 1;
				}
			}
		}
	}
}

void Pro3()
{
	for (int n = 0; n < looptime; n++)
	{
		for (int i = 0; i < msize * msize * msize; i++)
		{
			D[i] = 1;
		}
	}
}
```

以及减少寄存器使用量，如下，分别是380ms和440ms

```
void Pro1()
{
	int a = 1, b = 1, c = 1, d = 1, e = 1, f = 1, g = 1;
	for (int n = 0; n < looptime; n++)
	{
		for (int i = 0; i < msize; i++)
		{
			for (int j = 0; j < msize; j++)
			{
				for (int k = 0; k < msize; k++)
				{
					c = a + b + e;
					d = a + b + f;
				}
			}
		}
	}
}

void Pro2()
{
	int a = 1, b = 1, c = 1, d = 1,e = 1, f = 1, g = 1;
	for (int n = 0; n < looptime; n++)
	{
		for (int i = 0; i < msize; i++)
		{
			for (int j = 0; j < msize; j++)
			{
				for (int k = 0; k < msize; k++)
				{
					g = a + b;
					c = g + e;
					d = g + f;
				}
			}
		}
	}
}

```

![image-20210928093608729](D:\图形学书籍\系列流体文章\gif\image-20210928093608729.png)

乘除法，几乎三倍的差距！

读写顺序，差距80%！

```
// 2297ms 1546ms 1318ms

#include <iostream>
#include <time.h>　
using namespace std;
//https://blog.csdn.net/mxclxp/article/details/7991127
const int msize = 128;
const int looptime = 100;
const int lsize = 8;
int A[msize][msize][msize];
int B[msize][msize][msize];
int C[msize][msize][msize];
int D[msize][msize][msize];
int E[msize][msize][msize];
int F[msize][msize][msize];

void Pro1()
{
	int ta, tb, tc, td;
	for (int n = 0; n < looptime; n++)
	{
		for (int i = 0; i < msize; i++)
		{
			for (int j = 0; j < msize; j++)
			{
				for (int k = 0; k < msize; k++)
				{
					ta = A[i][j][k];
					tb = B[i][j][k];
					E[i][j][k] = ta + tb;
					tc = C[i][j][k];
					td = D[i][j][k];
					F[i][j][k] = tc + td;

				}
			}
		}
	}
}

void Pro2()
{
	int ta, tb, tc, td;
	for (int n = 0; n < looptime; n++)
	{
		for (int i = 0; i < msize; i++)
		{
			for (int j = 0; j < msize; j++)
			{
				for (int k = 0; k < msize; k++)
				{
					ta = A[i][j][k]; 
					tb = B[i][j][k];
					tc = C[i][j][k];
					td = D[i][j][k];
					E[i][j][k] = ta + tb;
					F[i][j][k] = tc + td;

				}
			}
		}
	}
}

void Pro3()
{
	int ta, tb, tc, td;
	for (int n = 0; n < looptime; n++)
	{
		for (int i = 0; i < msize; i++)
		{
			for (int j = 0; j < msize; j++)
			{
				for (int k = 0; k < msize; k++)
				{
					E[i][j][k] = A[i][j][k] + B[i][j][k];
					F[i][j][k] = C[i][j][k] + D[i][j][k];

				}
			}
		}
	}
}

int main()
{
	

	clock_t start, end;
	
	for (int i = 0; i < msize; i++)
	{
		for (int j = 0; j < msize; j++)
		{
			for (int k = 0; k < msize; k++)
			{
				A[i][j][k] = 0;
				B[i][j][k] = 0;
				C[i][j][k] = 0;
			}
		}
	}
	start = clock();
	Pro1();
	end = clock();
	cout << "time = " << double(end - start)*1000 / CLOCKS_PER_SEC << "ms" << endl;
	start = clock();
	Pro2();
	end = clock();
	cout << "time = " << double(end - start) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
	start = clock();
	Pro3();
	end = clock();
	cout << "time = " << double(end - start) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
	return 0;
}


```

As explained this in the section of parallel do construct, default scheduling in
OpenMP is dividing loop length by the number of threads to allocate computations with the equal division to each thread. This type of scheduling works well when computations to be allocated in loop length can be divided equally. However, general loops cannot be assured of such “equal division” for computations. If each computation for loop iteration causes any imbalance, such kind of scheduling (shown below) will not work. As a conclusion, the efficiency of thread parallelizatio decreases. This situation is called “load imbalance.  

. In the architecture of ccNUMA, allocated arrays are set to
the nearest core when the first access for the arrays is performed.   The following is an initialization loop for the array A and B. This allows us to
set the array A and B to the nearest memory  

```
!$omp parallel do private( j )
do i=1, 100
do j=1, 100
A( i ) = 0.0d0
B( i , j ) =0.0d0
enddo
enddo
```

ScaLAPACK,mpi并行化后的lapack