#include "bvh.cuh"
#include "device_lock.cuh"
#include "intellisense_cuda.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "debug_utils.h"

__device__ __inline__ int findUpperContZero(int64_t num)
{
	return __clzll(num);
}

CUDA_CALLABLE inline void expandBits(int64_t& bits)
{
	bits = (bits | bits << 32) & 0x1f00000000ffff;
	bits = (bits | bits << 16) & 0x1f0000ff0000ff;
	bits = (bits | bits << 8) & 0x100f00f00f00f00f;
	bits = (bits | bits << 4) & 0x10c30c30c30c30c3;
	bits = (bits | bits << 2) & 0x1249249249249249;
}

CUDA_CALLABLE inline int64_t floatTo21Int(float x)
{
	int32_t ix = *(int32_t*)(&x);
	int32_t exponent = (ix >> 23) & 0x000000ff;
	int32_t mantissa = (ix & 0x00ffffff) | 0x00800000;
	exponent -= 127;
	uint32_t signBit = ix & 0x80000000;
	signBit >>= 30;
	int32_t sign = -1 * ((int32_t)signBit - 1);
	int32_t value;
	if (exponent >= 8) {
		value = INT32_MAX;
	}
	else
	{
		if (exponent >= 0) value = mantissa << exponent;
		else value = mantissa >> (exponent * -1);
	}
	value *= sign;
	value += INT32_MAX;
	uint32_t bits = *(uint32_t*)(&value);
	int64_t result = (bits & 0xfffff800) >> 11;
	return result;
}

// use 21 bit per coordinate
// TODO: transform from float, should be more precise
CUDA_CALLABLE inline int64_t getMortonCode(Vec3 point)
{
	/*int64_t ix = ((*(int32_t*)(&point.x)) >> 2) & 0x1ffffflu;
	int64_t iy = ((*(int32_t*)(&point.y)) >> 2) & 0x1ffffflu;
	int64_t iz = ((*(int32_t*)(&point.z)) >> 2) & 0x1ffffflu;*/
	int64_t ix = floatTo21Int(point.x);
	int64_t iy = floatTo21Int(point.y);
	int64_t iz = floatTo21Int(point.z);
	expandBits(ix);
	expandBits(iy);
	expandBits(iz);

	int64_t code = ix | (iy << 1) | (iz << 2);
	return code;
}

__device__ inline int getTheOtherEnd(int64_t const* keys, int32_t idx, int32_t size, int& dir, int& lmax)
{

	// get direction of the range
	int64_t left;
	if (idx == 0) left = -1;
	else left = keys[idx - 1];
	int64_t right = keys[idx + 1];
	int64_t self = keys[idx];
	int leftCommon = findUpperContZero(left ^ self);
	int rightCommon = findUpperContZero(right ^ self);
	dir = leftCommon > rightCommon ? -1 : 1;

	// compute upper bound for the length of the range
	int minRange = min(leftCommon, rightCommon);
	lmax = 2;
	int theOtherEnd = idx + dir * lmax;
	while (theOtherEnd >= 0 && theOtherEnd < size &&
		(findUpperContZero(self ^ keys[theOtherEnd]) > minRange))
	{
		lmax <<= 1;
		theOtherEnd = idx + dir * lmax;
	}

	// find the other end using binary search
	int range = 0;
	for (int step = lmax >> 1; step > 0; step >>= 1)
	{
		theOtherEnd = idx + (range + step) * dir;
		if ( theOtherEnd < 0 || theOtherEnd >= size ) continue;
		if (findUpperContZero(self ^ keys[theOtherEnd]) > minRange) range += step;
	}

	theOtherEnd = idx + range * dir;
	return theOtherEnd;
}

__device__ inline int findSplitPosition(int64_t const* keys, int idx, int otherEnd, int dir, int lmax)
{
	int left, right;

	if (dir == -1) { left = otherEnd; right = idx; }
	else { left = idx; right = otherEnd; }

	int delta = findUpperContZero(keys[left] ^ keys[right]);
	int split = 0;

	for (int t = lmax >> 1; t > 0; t >>= 1)
	{
		int pos = idx + dir * (split + t);
		if ( pos < left || pos > right ) continue;
		if (findUpperContZero(keys[idx] ^ keys[pos]) > delta) split += t;
	}
	split = idx + split * dir;

	return split;
}

struct NodeState
{
	CUDA_CALLABLE NodeState(): lock(), writeTimes(0) {}
	DSpinlock lock;
	int writeTimes;
};

__global__ void initNodes(Vec3* vertices, uint32_t* indices, int64_t* keys, BVHNode* lNodes, 
	BVHNode* iNodes, NodeState* states, uint32_t size)
{
	int32_t tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	int32_t bid = blockIdx.x + blockIdx.y * gridDim.x;
	int32_t tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;

	if (tidGlobal >= size) return;

	__shared__ BBox boxes[256];

	boxes[tidLocal] = BBox(vertices[indices[3 * tidGlobal]], vertices[indices[3 * tidGlobal + 1]],
		vertices[indices[3 * tidGlobal + 2]]);
	Vec3 centroid = boxes[tidLocal].center();
	keys[tidGlobal] = getMortonCode(centroid);
	lNodes[tidGlobal].box = boxes[tidLocal];
	lNodes[tidGlobal].info.leaf.fid = tidGlobal;
	if (tidGlobal < size - 1) {
		iNodes[tidGlobal].box = BBox(Vec3(REAL_MAX), Vec3(-REAL_MAX));
		states[tidGlobal].writeTimes = 0;
	}
}

__global__ void computeNodeRange(BVHNode* iNodes, BVHNode* lNodes, int64_t* keys, int32_t nFaces)
{
	int tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;
	if (tidGlobal >= nFaces) return;

	if (tidGlobal < nFaces - 1)
	{
		int dir = 0;
		int lmax = 0;
		int otherEnd = getTheOtherEnd(keys, tidGlobal, nFaces, dir, lmax);
		int splitPos = findSplitPosition(keys, tidGlobal, otherEnd, dir, lmax);

		if ( dir == 1 )
		{
			if ( max(tidGlobal, otherEnd) == splitPos + 1 )
			{
				// right child is leaf node
				lNodes[splitPos + 1].parent = tidGlobal;
				iNodes[tidGlobal].info.intern.rightChild = splitPos + nFaces;
			}
			else
			{
				iNodes[splitPos + 1].parent = tidGlobal;
				iNodes[tidGlobal].info.intern.rightChild = splitPos + 1;
			}

			if ( min(tidGlobal, otherEnd) == splitPos )
			{
				// left child is leaf node
				lNodes[splitPos].parent = tidGlobal;
				iNodes[tidGlobal].info.intern.leftChild = splitPos + nFaces - 1;
			}
			else
			{
				iNodes[splitPos].parent = tidGlobal;
				iNodes[tidGlobal].info.intern.leftChild = splitPos;
			}
		}
		else
		{
			if (min(tidGlobal, otherEnd) == splitPos - 1)
			{
				// left child is leaf node
				lNodes[splitPos - 1].parent = tidGlobal;
				iNodes[tidGlobal].info.intern.leftChild = splitPos + nFaces - 2;
			}
			else
			{
				iNodes[splitPos - 1].parent = tidGlobal;
				iNodes[tidGlobal].info.intern.leftChild = splitPos - 1;
			}

			if (max(tidGlobal, otherEnd) == splitPos)
			{
				// right child is leaf node
				lNodes[splitPos].parent = tidGlobal;
				iNodes[tidGlobal].info.intern.rightChild = splitPos + nFaces - 1;
			}
			else
			{
				iNodes[splitPos].parent = tidGlobal;
				iNodes[tidGlobal].info.intern.rightChild = splitPos;
			}
		}
	}
}

__global__ void computeBBox(BVHNode* iNodes, BVHNode* lNodes, NodeState* states, int32_t size)
{
	int tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;
	if (tidGlobal >= size) return;

	int currentIdx = tidGlobal;

	int parentIdx = lNodes[currentIdx].parent;
	bool cont = false;
	// Dead lock?
	states[parentIdx].lock.acquire();

	iNodes[parentIdx].box.enclose(lNodes[currentIdx].box);
	states[parentIdx].writeTimes += 1;
	cont = states[parentIdx].writeTimes == 2;

	states[parentIdx].lock.release();

	while (cont)
	{
		currentIdx = parentIdx;
		parentIdx = iNodes[currentIdx].parent;
		states[parentIdx].lock.acquire();

		iNodes[parentIdx].box.enclose(iNodes[currentIdx].box);
		states[parentIdx].writeTimes += 1;
		cont = states[parentIdx].writeTimes == 2 && currentIdx != 0;

		states[parentIdx].lock.release();
	}
}

void BVH::construct(thrust::device_vector<Vec3>& vertices, thrust::device_vector<uint32_t>& indices)
{
	auto nFace = indices.size() / 3;
	auto nNodes = 2 * nFace - 1;

	if (m_nodes.size() != nNodes) m_nodes.resize(nNodes);
	if (m_keys.size() != nFace) m_keys.resize(nFace);

	thrust::device_vector<NodeState> states(nFace - 1);

	auto iNodesPtr = thrust::raw_pointer_cast(m_nodes.data());
	auto keysPtr = thrust::raw_pointer_cast(m_keys.data());
	auto lNodesPtr = iNodesPtr + nFace - 1;
	auto verticesPtr = thrust::raw_pointer_cast(vertices.data());
	auto indicesPtr = thrust::raw_pointer_cast(indices.data());
	auto statePtr = thrust::raw_pointer_cast(states.data());

	// constructing BVH
	dim3 blkSize = 256;
	dim3 gridSize = (nFace + 255) / 256;
	initNodes KERNEL_DIM(gridSize, blkSize) (verticesPtr, indicesPtr, keysPtr, lNodesPtr, iNodesPtr, statePtr, nFace);
	CUDA_CHECK(cudaDeviceSynchronize());
	thrust::sort_by_key(m_keys.begin(), m_keys.end(), m_nodes.begin() + nFace - 1);
	computeNodeRange KERNEL_DIM(gridSize, blkSize) (iNodesPtr, lNodesPtr, keysPtr, nFace);
	CUDA_CHECK(cudaDeviceSynchronize());
	computeBBox KERNEL_DIM(gridSize, blkSize) (iNodesPtr, lNodesPtr, statePtr, nFace);
	CUDA_CHECK(cudaDeviceSynchronize());
}
