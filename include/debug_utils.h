#pragma once

#ifndef DEBUG_UTILS
#define DEBUG_UTILS

#include <thrust/device_vector.h>

template<typename  T>
void checkDeviceVector(thrust::device_vector<T>& src)
{
	std::vector<T> dst(src.size());
	thrust::copy(src.begin(), src.end(), dst.begin());
}

#endif
