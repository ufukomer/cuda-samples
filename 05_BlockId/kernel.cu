#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void hello()
{
	printf("Hello world! I'm a thread in block %d\n", blockIdx.x);
}

int main()
{
	// 16! different output
	hello <<< NUM_BLOCKS, BLOCK_WIDTH >>> ();

	// Wait for the kernal execution
	cudaDeviceSynchronize();

	printf("That's all!\n");

	return 0;
}