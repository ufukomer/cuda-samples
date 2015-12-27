#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void reduction(float *d_out, float *d_in)
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
    // TODO: Write main function

    return 0;
}