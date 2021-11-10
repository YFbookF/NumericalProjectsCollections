===========基于CUDA的并行程序设计
感觉也没什么特殊之处啊
template<typename T> __global__ void MeanFilterCUDA(T* pInput, T* pOutput,
	int nKernelSize, int nWidth, int nHeight)

{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int Pos = j * nWidth + i;
	if ((0 < i) && (i < nWidth - 1) && (j > 0) && (j < nHeight - 1))
	{
		float temp = pInput[Pos - 1];
		temp += pInput[Pos];
		temp += pInput[Pos + 1];
		temp += pInput[Pos - nWidth - 1];
		temp += pInput[Pos - nWidth];
		temp += pInput[Pos - nWidth + 1];
		temp += pInput[Pos + nWidth - 1];
		temp += pInput[Pos + nWidth];
		temp += pInput[Pos + nWidth + 1];
		pOutput[Pos] = (T)(temp / nKernelSize);
	}
	else
	{
		pOutput[Pos] = pInput[Pos];
	}
}

dim3 block(256, 1, 1);
dim3 grid((int)((nWidth + 255) / 256.0), nHeight, 1);
MeanFilterCUDA<<<grid,block,0>>>()