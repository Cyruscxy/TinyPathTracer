#pragma once

#include "intellisense_cuda.h"
#include "ray.h"
#include "sampler.h"
#include "bvh.cuh"
#include "geometry_queries.h"
#include "mesh.cuh"
#include <cuda_runtime.h>
#include <cstdint>

constexpr int BLK_DIM = 16;
constexpr int BLK_SIZE = BLK_DIM * BLK_DIM;

__global__ void setup(size_t seed, curandState* state)
{
	size_t tid = threadIdx.x + threadIdx.y * blockDim.x;
	curand_init(seed, tid, 0, &state[tid]);
}

__global__ void sampleRay (Ray* rays, int width, int height, curandState* globalState, Real vFov, Real aspectRatio)
{
	int tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	int bid = blockIdx.x + gridDim.x * blockIdx.y;
	int tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;
	int pixelX = threadIdx.x + blockDim.x * blockIdx.x;
	int pixelY = threadIdx.y + blockDim.y * blockIdx.y;

	if (pixelX >= width || pixelY >= height) return;

	auto localState = globalState + tidLocal;
	auto localRay = rays[pixelX + pixelY * width];

	Vec2 pixelLocation = DeviceSampler::RectUniform(localState);
	Real pdf = DeviceSampler::RectPDF(pixelLocation);
	pixelLocation += Vec2(pixelX, pixelY) ; // screen space

	Vec2 invScale(1.0f / width, 1.0f / height);
	pixelLocation *= invScale; // normalized device space

	Real tan_vFov = tanf(MathConst::Rad * vFov * 0.5f);
	Real sensorHeight = 2.0f * tan_vFov;
	Real sensorWidth = aspectRatio * sensorHeight;
	Vec3 sensorPlaneConner(-0.5f * sensorWidth, -0.5f * sensorHeight, -1.0f);
	Vec3 sensorPlaneRightUp(sensorWidth, sensorHeight, 0.0f);

	localRay.m_direction = sensorPlaneConner + sensorPlaneRightUp * Vec3(pixelLocation, 0.0f);
	localRay.m_direction *= rsqrtf(localRay.m_direction.norm2());
	localRay.m_origin = Vec3(0.0f);
	localRay.m_distBound = Vec2(0.0f, REAL_MAX);
	localRay.m_distance = 0.0f;
}

CUDA_CALLABLE inline
int traverseBVH(Ray& ray, BVHNode* nodes, Vec3* vertices, uint32_t* indices, int size, Real& dist)
{
	int callStacks[64];
	int stackPtr = 1;
	callStacks[0] = 0;
	
	int hitIdx = -1;

	while ( stackPtr > 0 )
	{
		int currentIdx = callStacks[stackPtr - 1];
		stackPtr--;

		if ( currentIdx >= size - 1 ) // is leaf node
		{
			int fid = nodes[currentIdx].info.leaf.fid;
			auto vid0 = indices[3 * fid + 0];
			auto vid1 = indices[3 * fid + 1];
			auto vid2 = indices[3 * fid + 2];
			Real hitDist;
			// Shall we do the ray-triangle test here, or later after all collected ?
			if ( rayHitTriangle(ray, vertices[vid0], vertices[vid1], vertices[vid2], hitDist) )
			{
				if (hitDist < dist) { dist = hitDist; hitIdx = currentIdx; }
			}
		}
		else
		{
			int leftChild = nodes[currentIdx].info.intern.leftChild;
			int rightChild = nodes[currentIdx].info.intern.rightChild;
			if ( rayHitBBox(ray, nodes[leftChild].box) )
			{
				callStacks[stackPtr] = leftChild;
				stackPtr++;
			}
			if (rayHitBBox(ray, nodes[rightChild].box))
			{
				callStacks[stackPtr] = leftChild;
				stackPtr++;
			}
		}
	}

	return hitIdx;
}

// worry about warp divergence
CUDA_CALLABLE inline
int mtlBinarySearch(int index, MtlInterval* mtlLUT, int objCnt)
{
	// binary search
	int left = 0;
	int right = objCnt - 1;
	while ( right != left )
	{
		int mid = (left + right) >> 1;
		if (index < mtlLUT[mid].begin) right = mid;
		else left = mid;
	}
	return mtlLUT[right].mtlIdx;
}

CUDA_CALLABLE inline
int mtlLinearSearch(int index, MtlInterval* mtlLUT, int objCnt)
{
	int i = 0;
	for ( ; i < objCnt; ++i )
	{
		if (index >= mtlLUT[i].begin) break;
	}
	return mtlLUT[i].mtlIdx;
}

CUDA_CALLABLE inline
Vec3 getBaryCoords(Vec3 v0, Vec3 v1, Vec3 v2, Vec3 pos)
{
	Real gamma = cross(pos - v0, v1 - v0).norm();
	Real alpha = cross(pos - v1, v2 - v1).norm();
	Real beta = cross(pos - v2, v0 - v2).norm();
	Real invSum = 1.0f / (alpha + beta + gamma);
	alpha *= invSum;
	beta *= invSum;
	gamma *= invSum;
	return Vec3( alpha, beta, gamma );
}

CUDA_CALLABLE inline
Vec3 reflect(Vec3 dir, Vec3 normal)
{
	return dir + 2.0f * dot(dir, normal) * normal;
}

CUDA_CALLABLE inline
Vec3 refract(Vec3 dir, Real ior)
{
	auto eta = dir.y < 0.0f ? ior : 1.0f / ior;

	Real cosThetaI = abs(dir.y);
	Real sin2ThetaI = max(1.0f - cosThetaI * cosThetaI, 0.0f);
	Real sin2ThetaT = eta * eta * sin2ThetaI;

	if ( sin2ThetaT >= 1.0f )
	{
		return reflect(dir);
	}

	Real cosThetaT = sqrtf(1.0f - sin2ThetaT);
	return -eta * dir + (dir.y > 0.0f ? 1.0f : -1.0f) * (eta * cosThetaI - cosThetaT) * Vec3(0.0f, 1.0f, 0.0f);
}

CUDA_CALLABLE inline
Real shlickFresnel(Real u)
{
	Real m = min(1.0f, max(0.0f, 1.0f - u));
	Real m2 = m * m;
	return m2 * m2 * m;
}

__device__ __inline__
void generateNewRay(Ray& ray, Vec3 normal, Material* mtl, curandState* state)
{
	// TODO: Do we need to rotate the ray into local space?
	if ( mtl->eta > 0.0f )
	{
		Real nDotRayDir = dot(normal, ray.m_direction);
		Real eta = (nDotRayDir < 0.0f) ? mtl->eta : 1.0f / mtl->eta;
		Real cosThetaI = abs(nDotRayDir);
		Real sin2ThetaI = max(0.0f, 1.0f - cosThetaI * cosThetaI);
		Real sin2ThetaT = eta * eta * sin2ThetaI;

		if ( sin2ThetaT >= 0.0f )
		{
			ray.m_direction = reflect()
		}

		Real cosThetaT = sqrtf(1.0f - sin2ThetaT);
		ray.m_direction = 
	}
}

__global__
void trace(BVHNode* nodes, Vec3* vertices, Vec3* normals, uint32_t* indices, MtlInterval* mtlLUT,
	Material* materials, Ray* rays, curandState* globalState, int width, int height, int size, int depth, int objCnt)
{
	int tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	int bid = blockIdx.x + gridDim.x * blockIdx.y;
	int tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;
	int pixelX = threadIdx.x + blockDim.x * blockIdx.x;
	int pixelY = threadIdx.y + blockDim.y * blockIdx.y;

	if (pixelX >= width || pixelY >= height) return;

	while ( depth > 0 )
	{
		depth--;
		Real dist = REAL_MAX;
		auto localRay = rays[pixelX + pixelY * width];
		auto hitIdx = traverseBVH(localRay, nodes, vertices, indices, size, dist);

		if (hitIdx < 0) { break; }

		int mtlIndex = mtlBinarySearch(hitIdx, mtlLUT, objCnt);

		uint32_t i0 = indices[hitIdx * 3 + 0];
		uint32_t i1 = indices[hitIdx * 3 + 1];
		uint32_t i2 = indices[hitIdx * 3 + 2];
		auto v0 = vertices[i0];
		auto v1 = vertices[i1];
		auto v2 = vertices[i2];
		Vec3 newPos = localRay.m_origin + localRay.m_direction * localRay.m_distance;
		auto baryCoords = getBaryCoords(v0, v1, v2, newPos);

		v0 = normals[i0];
		v1 = normals[i1];
		v2 = normals[i2];
		auto actualNormal = baryCoords.x * v0 + baryCoords.y * v1 + baryCoords.z * v2;
	}

}


