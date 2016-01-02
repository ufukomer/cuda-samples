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

__global__ void reduction(int *d_in, int *d_out)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;

	// Do reduction in global mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			d_in[i] += d_in[i + s];
		}
		__syncthreads(); // Make sure all adds at one stage are done!
	}

	if (tid == 0)
	{
		d_out[blockIdx.x] = d_in[i];
	}
}

int main()
{
	const int N = ARRAY_SIZE;
	srand(time(NULL));

	int d[ARRAY_SIZE + BLOCK_DIM], a[12] = { 1, 3, 21, 55, 2, 5, 6, 8, 87, 6, 5, 0 };

	int *dev_a, *dev_d;

	cudaMalloc((void **)&dev_a, N * sizeof(int));
	cudaMalloc((void **)&dev_d, BLOCK_DIM * sizeof(int));

	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);

	reduction <<< (ARRAY_SIZE + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM >>>(dev_a, dev_d);

	cudaDeviceSynchronize();

	cudaMemcpy(d, dev_d, ARRAY_SIZE / BLOCK_DIM * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < ARRAY_SIZE / BLOCK_DIM; ++i)
		printf("d[%d]: %d\n", i, d[i]);

	cudaFree(dev_a);
	cudaFree(dev_d);

	printf("");

	return 0;

	return 0;
}