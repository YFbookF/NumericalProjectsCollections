
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>
#include "device_functions.h"
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <time.h>
#include <windows.h>
#include < time.h >
#include <iostream>
//https://github.com/mvkgta26/Merge-Sort-Multiple-Iterations/blob/main/kernel.cu
//外加pdf https://github.com/mvkgta26/Merge-Sort-Multiple-Iterations/blob/main/DESCRIPTION-MERGE-SORT.pdf
// 大意就是先用冒泡排序，搞定每个warp
// 然后merge的时候，就直接搜索引，好脑残，这样真的快吗？

//-------------------------------------------------------CPU TIMER LIBRARY-------------------------------------------------------

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  116444736000000000Ui64 // CORRECT
#else
#define DELTA_EPOCH_IN_MICROSECS  116444736000000000ULL // CORRECT
#endif

struct timezone
{
	int  tz_minuteswest; /* minutes W of Greenwich */
	int  tz_dsttime;     /* type of dst correction */
};

// Definition of a gettimeofday function

int gettimeofday(struct timeval* tv, struct timezone* tz)
{
	// Define a structure to receive the current Windows filetime
	FILETIME ft;

	// Initialize the present time to 0 and the timezone to UTC
	unsigned __int64 tmpres = 0;
	static int tzflag = 0;

	if (NULL != tv)
	{
		GetSystemTimeAsFileTime(&ft);

		// The GetSystemTimeAsFileTime returns the number of 100 nanosecond 
		// intervals since Jan 1, 1601 in a structure. Copy the high bits to 
		// the 64 bit tmpres, shift it left by 32 then or in the low 32 bits.
		tmpres |= ft.dwHighDateTime;
		tmpres <<= 32;
		tmpres |= ft.dwLowDateTime;

		// Convert to microseconds by dividing by 10
		tmpres /= 10;

		// The Unix epoch starts on Jan 1 1970.  Need to subtract the difference 
		// in seconds from Jan 1 1601.
		tmpres -= DELTA_EPOCH_IN_MICROSECS;

		// Finally change microseconds to seconds and place in the seconds value. 
		// The modulus picks up the microseconds.
		tv->tv_sec = (long)(tmpres / 1000000UL);
		tv->tv_usec = (long)(tmpres % 1000000UL);
	}

	if (NULL != tz)
	{
		if (!tzflag)
		{
			_tzset();
			tzflag++;
		}

		// Adjust for the timezone west of Greenwich
		tz->tz_minuteswest = _timezone / 60;
		tz->tz_dsttime = _daylight;
	}

	return 0;
}

//--------------------------------------------------------GPU TIMER LIBRARY--------------------------------------------------------------------

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

//-----------------------------------------------------------------------------------------------------------------------------

using namespace std;



//Device function which bubble sorts a specific section of arr[]. Section [start:end]. Both inclusive. 
__device__ void bubblesort(int* arr, int start, int end)
{
	int n = end - start + 1; //Length of array section from start to end

	int i, k, flag, temp;
	for (k = 1; k < (n - 1) + 1; k++)
	{
		flag = 0;
		for (i = start; i < (end - k + 1); i++)
		{
			if (arr[i] > arr[i + 1])
			{
				temp = arr[i];    //
				arr[i] = arr[i + 1];  //  Swapping A[i+1] and A[i]
				arr[i + 1] = temp;  //
			}
		}
	}
}



//Entire block for the entire array. Each thread takes care of bubble-sorting an individual section of size : *section_length
__global__ void section_sort(int* nums, int section_size)   //(, int n)
{
	//The thread with thread index = idx will take care of nums[] from : [( section_size * idx ) to ( section_size * (idx + 1) - 1 )]
	//For example: idx = 1 and section_size = 20, then, thread with idx = 1 will take care of nums[ 20: 39 ]

	int idx = threadIdx.x;



	//Bubble sort nums[] from index  [ ( section_size * idx ) : ( section_size * (idx + 1) - 1 ) ]
	bubblesort(nums, section_size * idx, (section_size * (idx + 1) - 1));

}



//----------------------------------------------------------------------------------------------------------------------------

//Device Function: Takes a number target and searches the array arr[] ( an array of size n), and returns the index such that : nums[index] <= target
__device__ int bin_search(int* arr, int target, int n)        // n = size of arr2
{

	//Corner Cases : When the target is out of boundary of the range of values in the array
	if (target < arr[0])		return -1;
	if (target > arr[n - 1])    return n - 1;


	//f, l, mid
	int left = 0;
	int right = n;
	int mid = (left + right) / 2;

	while (left <= right)
	{
		int mid = (left + right) / 2;    //Calculate mid

		if (arr[mid] == target)
		{
			return mid - 1;   //Return index where nums[] == target
		}

		//All elements to right of mid are greater than target
		else if (arr[mid] > target)
		{
			//If nums[mid-1] < target < nums[mid]      ( Meaning target lies between nums[mid-1] and nums[mid] ==> (mid-1) is the required index)
			if (arr[mid - 1] < target)
			{
				return (mid - 1);
			}

			else			//Change the right border
			{
				right = mid - 1;
			}

		}

		//All elements to left of mid are lesser than target
		else if (arr[mid] < target)
		{
			//If nums[mid] < target < nums[mid+1]            ( Meaning target lies between nums[mid] and nums[mid+1] )
			if ((arr[mid + 1] > target))
			{
				return (mid);
			}


			else		//Change the left border
			{
				left = mid + 1;
			}

		}
	}

	return -1;
}


//Merges 2 sorted array, by using a GPU kernel call to parallely produce scatter addresses:
// Each Thread of block will parallely produce scatter addresses for its element. Block is divided into 2 sections. 
// Scatter address for All elements of both sections are parallelly produced.
// THe block is responsible for merging both of its sections
// Finally, the block in array is sorted according to the scatter addresses
	// *arr  = array pointer
	// *section_size = The length of the both 2 subarrays into which arr[] is split
	// *d_out_temp = Where array output is stored
__global__ void merge(int* arr, int section_length, int* d_out_temp)
{
	//int section_length = *section_size;
	int superset_length = section_length * 2;   //Block will be 2 * (size of 1 section). Because 2 sections are merged
	int idx = threadIdx.x;
	int b_idx = blockIdx.x;

	//Length of arr1[] and arr2[] are section size
	int len1 = section_length;
	int len2 = section_length;

	//-----Select *arr1 and *arr2 and *d_out_curr------------------------

	int* arr1 = arr + (b_idx * superset_length);
	int* arr2 = arr1 + (section_length);
	int* d_out_curr = d_out_temp + (b_idx * superset_length);   //Determine d_out_curr[], the output array for current merge 

	//Dynamically allocated shared memory array. 
	// scat_ad[] from index [0 to n1-1] is for arr1[].
	//scat_ad[] from index [n1 to n2-1] is for arr2[]

	 //Create a shared memory of size n1+n2 to accomodate the scatter-addresses corresonding to each element in arr1[] and arr2[] 
	extern __shared__ int scat_ad[];

	//--------------------------------These threads are responsible for arr1[]-------------------------------------------------------
	if (idx <= len1 - 1)
	{
		int idx1 = idx;     //Number of elements in arr1[] that are lesser than arr1[idx]. idx1 = index of current element in arr1[]

		int target = arr1[idx1];    //Target is current element in arr1[]

		//--------------Find idx2----------------------------------------Binary Search Part------------------------------
		int idx2 = bin_search(arr2, target, len2) + 1;    //Number of elements in arr2[] that are lesser than arr1[idx].....

		//Calculate and store the scatter address in array
		//scat_arr1[idx] = idx1 + idx2;     //If there are 2 elements before a number in output array, its index will be 2

		scat_ad[idx] = idx1 + idx2;    //Scatter address correspinding to arr1[idx] = idx1 + idx2 
	}


	//--------------------------------------These threads are responsible for arr2[]--------------------------------------------
	else if (idx >= len1)
	{
		//Number of elements in arr2[] that are lesser than arr2[idx]. 
		//idx1 = index of current element in arr2[]
		//(idx-len1) because threads with index n1 to n2-1 are responsible for arr2[] index [0: n2-1] 
		int idx1 = idx - len1;

		int target = arr2[idx1];    //Target is current element in arr1[] 

		//--------------Find idx2-----------------------------Binary Search Part---------------------------
		int idx2 = bin_search(arr1, target, len1) + 1;    //Number of elements in arr1[] that are lesser than arr2[idx].  +1 bcos we want appropriate position for current element

		//Calculate and store the scatter address in array
		//scat_arr1[idx] = idx1 + idx2;     //If there are 2 elements before a number in output array, its index will be 2
		scat_ad[idx] = idx1 + idx2;    //Scatter address corresponding to arr2[idx - len1] = idx1 + idx2 

	}

	__syncthreads();   //Barrier to ensure that all threads have finished writing scat_ad[].------------------Not necessary

	//-------------Store the output in respective position in d_out_temp[] using scatter address so that they are in sorted order-----------------------------------
	/*
	if (idx < len1)
	{
		d_out_curr[scat_ad[idx]] = arr1[idx];
	}
	else if (idx >= len1)
	{
		//d_out_curr[scat_ad[idx]] = arr2[idx - len1];
		d_out_curr[scat_ad[idx]] = arr1[idx];
	}
	*/
	d_out_curr[scat_ad[idx]] = arr1[idx];

	__syncthreads();
	//--------------------------------------Copy sorted elements back to array-----------------------------------------------------


	arr1[idx] = d_out_curr[idx];

	//printf( "%d ", arr1[idx] );
}



//Makes kernel call to merge 2 sorted array:
	//
void merge_sort()
{
	GpuTimer timer;

	//4 sections of 5 elements size
	//int h_arr[] = { 120,119,118,117,116,	115,114,113,112,111,	110,109,108,107,106,	105,104,103,102,101 };
	//int h_arr[] = { 596, 703, 277, 228, 548, 515, 213, 880, 391, 364, 224, 623, 845, 152, 454, 987, 854, 257, 402, 990, 996, 819, 756, 735, 460, 87, 693, 268, 92, 14, 860, 68, 996, 934, 478, 855, 209, 293, 171, 285 };
	int h_arr[40] = { 100,99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61 };
	//int h_arr[80] = { 100,99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21 };


	//int h_arr[64] = { 100,99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37 };


	int n = sizeof(h_arr) / sizeof(int);     //n = Total size of host array

	int div_num = 4;   //How many parts the array is initially split.
	int section_size = n / div_num;				// section_size = Size of each section after splitting arr[] into div_num parts (Stored in Host)


	//-----------------------------------Create input and output arrays in GPU---------------------------------------------
	int* d_arr, * d_out_temp;																// *d_out2;
	cudaMalloc((void**)&d_arr, n * sizeof(int));
	cudaMemcpy((void*)d_arr, (void*)h_arr, n * sizeof(int), cudaMemcpyHostToDevice);   //d_arr[] is input array in device

	cudaMalloc((void**)&d_out_temp, n * sizeof(int));			//d_out_temp[] is temporarily used to store sorted block elements



	timer.Start();

	//------------------------------Stage-1: KERNEL CALL: Bubble Sort Each Section of section_size elements------------------------------------

	section_sort << <1, div_num >> > (d_arr, section_size);    //Call div_num threads: Each thread bubble-sorts a sub-section of n/div_num elements in the array.


	/*

	//---------------Stage-2 : KERNEL CALL: Perform 2 Parallel Merges on 2 Groups of 2 Sections (Each Section Of Size n/4)-----------------------------------------------

		//Make kernel call to 2 blocks of n/2 threads each. Each thread is responsible for 1 element of its block. 3rd parameter n/2 is for shared memory size
		//Imagine The entire arr[] is divided into 2 blocks of n/2 size each. Each block is divided into sections of section_size

	div_num = div_num / 2;				//Initially : Number of supersets will be Half of Total Number of Divisons (Here: 8/2 = 4)
	merge <<< div_num, n/div_num, n/div_num >>> ( d_arr, section_size, d_out_temp);
	 //Number of Threads = Size of Superset (Group of 2 sections)
	 //NUmber of blocks = Number of supersets

	//---------------------------Stage-3 : KERNEL CALL: Perform 1 Merge On 2 Sections (Each of Size n/2)-------------------------------------------------------------------

		//Make kernel call to 1 blocks of n threads . Each thread is responsible for 1 element of its block. 3rd parameter n is for shared memory size
		//Entire arr[] is 1 block.

	div_num = div_num / 2;  //Number of supersets will be halved (Here : 4/2= 2)
	section_size = section_size * 2;    //Size of each section will double
	merge <<< div_num, n/div_num, n/div_num >>> ( d_arr, section_size, d_out_temp);      //Call kernel with INPUT: d_out_temp, and Output = d_out2[].... Section Length = 10

	div_num = div_num / 2;  //Number of supersets will be halved (Here : 2/2 = 1)
	section_size = section_size * 2;    //Size of each section will double
	merge <<< div_num, n/div_num, n/div_num >>> (d_arr, section_size, d_out_temp);      //Call kernel with INPUT: d_out_temp, and Output = d_out2[].... Section Length = 10


	*/

	//-----------------------------VERY IMPORTANT NOTE------------------------
	//NOTE: SUPERSET = GROUP OF 2 SECTIONS. WHEN WE MERGE A SUPERSET, WE MERGE THE 2 SECTIONS OF THE SUPERSET TO PRODUCE A SORTED SUPERSET

	//Initially, section_size = n / div_num
	int superset_num = div_num / 2; //Is the total number of supersets in the array, each of which are merged by a separate block. Initially, number of supersets = half of total number of divisions/sections in array
	while (superset_num >= 1)
	{

		/*
		//---------------Stage-suoerset_num : KERNEL CALL: Perform 2 Parallel Merges on 2 Groups of 2 Sections (Each Section Of Size n/4)-----------------------------------------------
		*/
		//Make kernel call to 2 blocks of n/2 threads each. Each thread is responsible for 1 element of its block. 3rd parameter n/2 is for shared memory size
		//Imagine The entire arr[] is divided into 2 blocks of n/2 size each. Each block is divided into sections of section_size

		//Number of Threads = Size of Superset 
		//Number of blocks = Number of supersets
		merge << < superset_num, n / superset_num, n / superset_num >> > (d_arr, section_size, d_out_temp);

		//UPDATE : superset_num ( halved ) and section_size (doubled)
		superset_num = superset_num / 2;
		section_size = section_size * 2;

	}


	timer.Stop();

	double time_elapsed = timer.Elapsed();

	//---------------------------Copy Final Sorted Output From Device into a Host Array h_out[]------------------------------------------
	int* output_array = (int*)malloc(n * sizeof(int));

	cudaMemcpy((void*)output_array, (void*)d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);



	for (int i = 0; i < n; i++)
	{
		printf("%d ", output_array[i]);

		//if (i == 9) cout << endl;
	}

	printf("\n Time Elapsed : %g ms", time_elapsed);

}



void main()
{
	merge_sort();
}