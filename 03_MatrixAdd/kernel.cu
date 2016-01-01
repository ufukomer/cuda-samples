#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 

// Column & Row size
const int N = 16;
const int SIZE = N * N;

__global__ void matrixAdd(int *c, const int *a, const int *b, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int i = row + n * col;

	if (row < n && col < n)
	{
		c[i] = a[i] + b[i];
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

	dim3 dimBlock(N, N);
	dim3 dimGrid((int)ceil(N / dimBlock.x), (int)ceil(N / dimBlock.y));

	matrixAdd <<< dimGrid, dimBlock >>>(dev_c, dev_a, dev_b, N);

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
