#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 16	// Threads per block
#define SIZE 65			// Array size

__global__ void vectorAdd(int *a, int *b, int *c, int n)
{
	// blockIdx.x is block index
	// threadIdx.x is thread index
	// blockDim.x corresponds to threads per block

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// Avoid accessing beyond the end of the arrays
	if (i < n)
		c[i] = a[i] + b[i];

	// Parallel threads
	// c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];

	// Parallel blocks
	// c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

int main()
{
	int block_size = SIZE / THREADS_PER_BLOCK;
	int *a, *b, *c;		// Host arrays
	int *d_a, *d_b, *d_c;	// Device arrays

	// Allocate the memory on the CPU
	a = (int *)malloc(SIZE * sizeof(int));
	b = (int *)malloc(SIZE * sizeof(int));
	c = (int *)malloc(SIZE * sizeof(int));

	// Allocate the memory on the GPU
	cudaMalloc(&d_a, SIZE * sizeof(int));
	cudaMalloc(&d_b, SIZE * sizeof(int));
	cudaMalloc(&d_c, SIZE * sizeof(int));

	for (int i = 0; i < SIZE; ++i)
	{
		a[i] = i;
		b[i] = i + 2;
	}

	// Copy Host array to Device array
	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	// Make a call to GPU kernel
	vectorAdd <<< block_size, THREADS_PER_BLOCK >>>(d_a, d_b, d_c, SIZE);

	// Copy result back to Host array from Device array
	cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < SIZE; ++i)
		printf("c[%d] = %d\n", i, c[i]);

	// Free the Host array memory
	free(a);
	free(b);
	free(c);

	// Free the Device array memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
