#pragma once

#ifndef BVH_H
#define BVH_H

#include <thrust/device_vector.h>
#include "primitives.h"
#include <cstdint>
#include "intellisense_cuda.h"
#include <thrust/detail/mpl/math.h>

// AABB
struct BBox
{
	CUDA_CALLABLE inline BBox() : m_min(REAL_MAX), m_max(-REAL_MAX) {}
	CUDA_CALLABLE inline BBox(const Vec3& min, const Vec3& max) : m_min(min), m_max(max) {}
	CUDA_CALLABLE inline BBox(Vec3& v0, Vec3& v1, Vec3& v2) :
		m_min(hmin(hmin(v0, v1), v2)), m_max(hmax(hmax(v0, v1), v2)) {}
	CUDA_CALLABLE inline void enclose(const BBox& box)
	{
		m_min = hmin(m_min, box.m_min);
		m_max = hmax(m_max, box.m_max);
	}

	CUDA_CALLABLE inline bool empty() const { return m_min.x > m_max.x || m_min.y > m_max.y || m_min.z > m_max.z; }
	CUDA_CALLABLE inline Vec3 center() const { return (m_min + m_max) * 0.5f; }
	CUDA_CALLABLE inline Real surfaceArea() const
	{
		if (empty()) return 0.0f;
		Vec3 extent = m_max - m_min;
		return 2.0f * (extent.x * extent.z + extent.x * extent.y + extent.y * extent.z);
	}

	Vec3 m_min, m_max;
};

struct NodeState
{
	int lock;
	int writeTimes;
};

struct Intern
{
	int leftChild;
	int rightChild;
};

struct Leaf
{
	int fid;
	int placeHolder;
};

union NodeType
{
	Intern intern;
	Leaf leaf;
};

struct BVHNode
{
	CUDA_CALLABLE inline BVHNode() : parent(0), info({0, 0}), box(Vec3(REAL_MAX), Vec3(-REAL_MAX)) {}
	uint32_t parent;
	NodeType info;
	BBox box;
};

struct BVH
{
	BVH() = default;
	BVH(size_t size) : m_nodes(size - 1), m_keys(size) {}
	thrust::device_vector<BVHNode> m_nodes;
	thrust::device_vector<int64_t> m_keys;

	void construct(thrust::device_vector<Vec3>& vertices, thrust::device_vector<uint32_t>& indices);
};

#endif