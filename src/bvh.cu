#include "bvh.cuh"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

CUDA_CALLABLE inline void expandBits(int64_t& bits)
{
	bits = (bits | bits << 32) & 0x1f00000000ffff;
	bits = (bits | bits << 16) & 0x1f0000ff0000ff;
	bits = (bits | bits << 8) & 0x100f00f00f00f00f;
	bits = (bits | bits << 4) & 0x10c30c30c30c30c3;
	bits = (bits | bits << 2) & 0x1249249249249249;
}

// use 21 bit per coordinate, the range of coordinates should be limited in [0.0, 1.0]
CUDA_CALLABLE inline int64_t getMortonCode(Vec3& point)
{
	int64_t ix = (*reinterpret_cast<int32_t*>(&point.x) >> 2) & 0x1ffffflu;
	int64_t iy = (*reinterpret_cast<int32_t*>(&point.y) >> 2) & 0x1ffffflu;
	int64_t iz = (*reinterpret_cast<int32_t*>(&point.z) >> 2) & 0x1ffffflu;

	expandBits(ix);
	expandBits(iy);
	expandBits(iz);

	int64_t code = ix | (iy << 1) | (iz << 2);
	return code;
}

__device__ inline int getOtherEnd(int64_t const* keys, int32_t idx, int32_t size)
{
	if (idx == 0) return size - 1;

	// get direction of the range
	int64_t left = keys[idx - 1];
	int64_t right = keys[idx + 1];
	int64_t self = keys[idx];
	int leftDelta = __clzll(left ^ self);
	int rightDelta = __clzll(right ^ self);
	int dir = leftDelta > rightDelta ? -1 : 1;

	// compute upper bound for the length of the range
	int minRange = min(leftDelta, rightDelta);
	int lmax = 2;
	int64_t otherEnd = idx + dir * lmax;
	while (otherEnd >= 0 && otherEnd < size && (__clzll(self ^ keys[otherEnd]) > minRange))
	{
		lmax <<= 1;
		otherEnd = idx + dir * lmax;
	}

	// find the other end using binary search
	int range = 0;
	for (int step = lmax >> 1; step > 0; step >>= 1)
	{
		otherEnd = idx + (range + step) * dir;
		if (otherEnd >= 0 && otherEnd < size) continue;
		if (__clzll(self ^ keys[otherEnd]) > minRange) range += step;
	}

	otherEnd = idx + range * dir;
	return otherEnd;
}

__device__ inline int findSplitPosition(int64_t const* keys, int idx, int otherEnd)
{
	int left, right, dir;

	if (otherEnd < idx) { left = otherEnd; right = idx; dir = -1; }
	else { left = idx; right = otherEnd; dir = 1; }

	int delta = __clzll(keys[left] ^ keys[right]);
	int range = right - left;
	int split = 0;

	for (int t = range >> 1; t > 0; t >>= 1)
	{
		int pos = idx + dir * (split + t);
		if (__clzll(keys[idx] ^ keys[pos]) > delta) split += t;
	}
	split = idx + split * dir + min(dir, 0);

	return split;
}

__global__ void initNodes(Vec3* vertices, uint32_t* indices, int64_t* keys, BVHNode* lNodes, BVHNode* iNodes, NodeState* states, uint32_t size)
{
	uint32_t tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	uint32_t bid = blockIdx.x + blockIdx.y * gridDim.x;
	uint32_t tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;

	if (tidGlobal >= size) return;

	__shared__ BBox boxes[256];

	boxes[tidLocal] = BBox(vertices[3 * tidGlobal], vertices[3 * tidGlobal + 1], vertices[3 * tidGlobal + 2]);
	Vec3 centroid = boxes[tidLocal].center();
	keys[tidGlobal] = getMortonCode(centroid);
	lNodes[tidGlobal].box = boxes[tidLocal];
	if (tidGlobal < size - 1) {
		states[tidGlobal].lock = 0;
		iNodes[tidGlobal].box = BBox(Vec3(REAL_MAX), Vec3(-REAL_MAX));
		states[tidGlobal].writeTimes = 0;
	}
}

// thrust::sort_by_keys

__global__ void computeNodeRange(BVHNode* iNodes, BVHNode* lNodes, int64_t* keys, int32_t size)
{
	int tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;

	if (tidGlobal < size - 1)
	{
		int otherEnd = getOtherEnd(keys, tidGlobal, size);
		int splitPos = findSplitPosition(keys, tidGlobal, otherEnd);
		
		if (min(tidGlobal, otherEnd) == splitPos)
		{
			iNodes[tidGlobal].info.intern.leftChild = splitPos;
			lNodes[splitPos].parent = tidGlobal;
		}
		else
		{
			iNodes[tidGlobal].info.intern.leftChild = splitPos;
			iNodes[splitPos].parent = tidGlobal;
		}


		if (max(tidGlobal, otherEnd) == splitPos + 1)
		{
			iNodes[tidGlobal].info.intern.rightChild = splitPos + 1;
			lNodes[splitPos + 1].parent = tidGlobal;
		}
		else
		{
			iNodes[tidGlobal].info.intern.rightChild = splitPos + 1;
			iNodes[splitPos + 1].parent = tidGlobal;
		}
	}
}

__global__ void computeBBox(BVHNode* iNodes, BVHNode* lNodes, NodeState* states, int32_t size)
{
	int tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;
	if (tidGlobal >= size - 1) return;

	int currentIdx = tidGlobal;

	int parentIdx = lNodes[currentIdx].parent;
	bool cont = false;
	while (atomicCAS(&states[parentIdx].lock, 0, 1) != 0) {}

	iNodes[parentIdx].box.enclose(lNodes[currentIdx].box);
	states[parentIdx].writeTimes += 1;
	cont = states[parentIdx].writeTimes == 2;

	atomicExch(&states[parentIdx].lock, 0);

	while (cont)
	{
		currentIdx = parentIdx;
		parentIdx = iNodes[currentIdx].parent;
		while (atomicCAS(&states[parentIdx].lock, 0, 1) != 0) {}

		iNodes[parentIdx].box.enclose(iNodes[currentIdx].box);
		states[parentIdx].writeTimes += 1;
		cont = states[parentIdx].writeTimes == 2 && currentIdx != 0;

		atomicExch(&states[parentIdx].lock, 0);
	}
}

void BVH::construct(thrust::device_vector<Vec3>& vertices, thrust::device_vector<uint32_t>& indices)
{
	auto size = indices.size() / 3;

	if (m_nodes.size() != 2 * size - 1) m_nodes.resize(size - 1);
	if (m_keys.size() != size) m_keys.resize(size);

	thrust::device_vector<NodeState> states(size - 1);

	auto iNodesPtr = thrust::raw_pointer_cast(m_nodes.data());
	auto keysPtr = thrust::raw_pointer_cast(m_keys.data());
	auto lNodesPtr = iNodesPtr + size - 1;
	auto verticesPtr = thrust::raw_pointer_cast(vertices.data());
	auto indicesPtr = thrust::raw_pointer_cast(indices.data());
	auto statePtr = thrust::raw_pointer_cast(states.data());

	// constructing BVH
	dim3 blkSize = 256;
	dim3 gridSize = (size + 255) / 256;
	initNodes KERNEL_DIM(gridSize, blkSize) (verticesPtr, indicesPtr, keysPtr, lNodesPtr, iNodesPtr, statePtr, size);
	thrust::sort_by_key(m_keys.begin(), m_keys.end(), m_nodes.begin() + size - 1);
	computeNodeRange KERNEL_DIM(gridSize, blkSize) (iNodesPtr, lNodesPtr, keysPtr, size);
	computeBBox(iNodesPtr, lNodesPtr, statePtr, size);
}
