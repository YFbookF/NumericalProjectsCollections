
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
const int threshold = 400;
const int m = 100;
__global__ void selection_sort(int* a, int left, int right)
{
	int temp;
	for (int i = left; i < right; i++)
		for (int j = i + 1; j <= right; j++)
			if (a[i] > a[j])
			{
				temp = a[i];
				a[i] = a[j];
				a[j] = temp;
			}
}


__global__ void partition(int* a, int left, int right, int pivot, int* al, int* ah)
{
	int l, h;
	int diff = (right - left + 1) / m;
	int k1 = threadIdx.x * diff + left;
	int k2 = k1 + diff - 1;
	if (threadIdx.x == m - 1)
		k2 = right;
	l = h = k1;
	for (int i = k1; i <= k2; i++)
	{
		al[i] = ah[i] = -999;
	}
	for (int i = k1; i <= k2; i++)
	{
		if (a[i] < pivot)
		{
			al[l++] = a[i];
		}
		else
		{
			if (a[i] > pivot)
			{
				ah[h++] = a[i];
			}
		}
	}
}


const int n = 16;
void quicksort(int* a, const int left, const int right)
{
	if (right - left <= threshold)
	{
		int* ad;
		cudaMalloc((void**)&ad, (right - left + 1) * sizeof(int));
		cudaMemcpy(ad, a, (right - left + 1) * sizeof(int), cudaMemcpyHostToDevice);
		selection_sort << <1, 1 >> > (ad, left, right);
		cudaMemcpy(a, ad, (right - left + 1) * sizeof(int), cudaMemcpyDeviceToHost);
		return;
	}
	int pivot = a[left];
	int* al, * ah;
	int* ad;
	cudaMalloc((void**)&ad, (right - left + 1) * sizeof(int));
	cudaMalloc((void**)&al, (right - left + 1) * sizeof(int));
	cudaMalloc((void**)&ah, (right - left + 1) * sizeof(int));
	cudaMemcpy(ad, a, (right - left + 1) * sizeof(int), cudaMemcpyHostToDevice);
	partition << <1, m >> > (ad, left, right, pivot, al, ah);
	int al_h[n], ah_h[n];
	cudaMemcpy(al_h, al, (right - left + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(ah_h, ah, (right - left + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	int i = 0, k = 0;
	while (i < right - left + 1)
	{
		while (al_h[i] == -999 && i < right - left + 1)
			i++;
		while (al_h[i] != -999 && i < right - left + 1)
		{
			al_h[k++] = al_h[i++];
		}
	}
	quicksort(al_h, 0, k - 1);
	int p = left;
	int x = 0;
	while (x < k)
	{
		a[p++] = al_h[x++];
	}
	a[p] = pivot;
	i = 0;
	k = 0;
	while (i < right - left + 1)
	{
		while (ah_h[i] == -999 && i < right - left + 1)
			i++;
		while (ah_h[i] != -999 && i < right - left + 1)
		{
			ah_h[k++] = ah_h[i++];
		}
	}
	quicksort(ah_h, 0, k - 1);
	i = 0;
	p++;
	while (i < k)
	{
		a[p++] = ah_h[i++];
	}
}


int main()
{
	//scanf("%d", &n);
	int a[n];

	float start, end, total_time_taken;


	int x, flag;
	for (unsigned i = 0; i < n; i++)
	{
		x = rand() % n;

		flag = 0;
		for (int j = 0; j < i; j++)
		{
			if (a[j] == x)
			{
				i--;
				flag = 1;
				break;
			}
		}
		if (flag == 0)
			a[i] = x;
	}
	printf("before sort:");
	for (int i = 0; i < n; i++)
		printf("%d  ", a[i]);
	quicksort(a, 0, n - 1);
	printf("\n after sort:");
	for(int i=0;i<n;i++)
		printf("%d  ",a[i]);
}