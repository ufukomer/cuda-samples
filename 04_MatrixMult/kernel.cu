#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 

#define BLOCK_DIM 16

// Column & Row size
const int N = 2;
const int SIZE = N * N;

__global__ void matrixMult(int *c, int *a, int *b, int n)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int k, sum = 0;

	if (row < n && col < n)
	{
		for (k = 0; k < n; k++)
			sum += a[row * n + k] * b[k * n + col];

		c[row * n + col] = sum;
	}
}

int main()
{
	srand(time(NULL));

	int a[N][N], b[N][N], *c;
	int *dev_a, *dev_b, *dev_c;

	c = (int *)malloc(SIZE * sizeof(int));

	cudaMalloc((void **)&dev_a, SIZE * sizeof(int));
	cudaMalloc((void **)&dev_b, SIZE * sizeof(int));
	cudaMalloc((void **)&dev_c, SIZE * sizeof(int));

	for (int row = 0; row < N; ++row)
		for (int col = 0; col < N; ++col)
		{
			a[row][col] = rand() % 10;
			b[row][col] = rand() % 10;
		}

	cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimBlock(1, 1);
	dim3 dimGrid(N, N);

	matrixMult <<< dimGrid, dimBlock >>>(dev_c, dev_a, dev_b, N);

	cudaMemcpy(c, dev_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < SIZE; ++i)
		printf("c[%d] =  %d\n", i, c[i]);

	// Free the Host array memory
	free(c);

	// Free the Device array memory
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}