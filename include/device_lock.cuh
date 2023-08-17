#ifndef DEVICE_LOCK
#define DEVICE_LOCK

#include <cuda_runtime.h>
#include "intellisense_cuda.h"

struct DSpinlock
{
	int locked;

	__device__ __inline__ DSpinlock() : locked(0) {}

	__device__ __inline__ void acquire()
	{
		while ( atomicCAS(&locked, 0, 1) != 0 ) {}
	}

	__device__ __inline__ void release()
	{
		atomicExch(&locked, 0);
	}
};

#endif