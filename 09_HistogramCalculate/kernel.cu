#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include <iostream>

#define ARRAY_SIZE 1024
#define BLOCK_DIM 1024

using namespace std;

__global__ void fill_histrogram(int *dev_out, int *dev_in)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	atomicAdd(&dev_out[dev_in[i]], 1);
}

int main()
{
	int a[ARRAY_SIZE], b[100];
	int *dev_in, *dev_out;

	srand(time(NULL));

	cudaMalloc((void **)&dev_in, ARRAY_SIZE * sizeof(int));
	cudaMalloc((void **)&dev_out, 100 * sizeof(int));

	for (int i = 0; i < ARRAY_SIZE; ++i)
	{
		a[i] = rand() % 100;
	}

	cudaMemcpy(dev_in, a, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	fill_histrogram <<< ARRAY_SIZE / BLOCK_DIM, BLOCK_DIM >>>(dev_out, dev_in);

	cudaMemcpy(b, dev_out, 100 * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 100; ++i)
	{
		cout << "Out[" << i << "]: " << b[i] << endl;
	}

	cudaFree(dev_in);
	cudaFree(dev_out);

	return 0;
}