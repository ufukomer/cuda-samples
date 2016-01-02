#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <ctime>

#define BLOCK_DIM 4
#define ARRAY_SIZE 12

__global__ void maxValue(int *a, int *d)
{
	__shared__ int sdata[BLOCK_DIM]; 

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = a[i];

	for (unsigned int s = BLOCK_DIM / 2; s >= 1; s = s / 2)
	{
		if (tid < s)
		{
			if (sdata[tid] < sdata[tid + s])
			{
				sdata[tid] = sdata[tid + s];
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		d[blockIdx.x] = sdata[0];
	}
}

int main()
{
	srand(time(NULL));
	const int N = ARRAY_SIZE;

	int d[ARRAY_SIZE / BLOCK_DIM], a[12] = { 1, 3, 21, 55, 2, 5, 6, 8, 87, 6, 5, 0 };

	int *dev_a, *dev_d;

	cudaMalloc((void **)&dev_a, N * sizeof(int));
	cudaMalloc((void **)&dev_d, BLOCK_DIM * sizeof(int));

	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);

	maxValue <<< (ARRAY_SIZE + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM >>>(dev_a, dev_d);

	cudaDeviceSynchronize();

	cudaMemcpy(d, dev_d, ARRAY_SIZE / BLOCK_DIM * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < ARRAY_SIZE / BLOCK_DIM; ++i)
		printf("Max: %d\n", d[i]);

	cudaFree(dev_a);
	cudaFree(dev_d);

	return 0;
}