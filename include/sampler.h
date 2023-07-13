#pragma once

#ifndef SAMPLER_H
#define SAMPLER_H

#include "intellisense_cuda.h"
#include <curand_kernel.h>
#include "math/vec.h"

namespace DeviceSampler
{
	__device__ __inline__ Vec2 RectUniform(curandState* state, const Vec2& size = Vec2(1.0f, 1.0f))
	{
		Real u = curand_uniform(state);
		Real v = curand_uniform(state);
		return Vec2(u, v) * size;
	}

	__device__ __inline__ Real RectPDF(const Vec2& at, const Vec2& size = Vec2(1.0f, 1.0f))
	{
		Real f = (at.x < 0.0f || at.x > size.x || at.y < 0.0f || at.y > size.y);
		return f * 1.0f / (size.x * size.y);
	}

	__device__ __inline__ Vec3 TriangleUniform(curandState* state, const Vec3& v0, const Vec3& v1, const Vec3& v2)
	{
		Real u = sqrtf(curand_uniform(state));
		Real v = curand_uniform(state);
		Real a = u * (1.0f - v);
		Real b = u * v;
		return a * v0 + b * v1 + (1.0f - a - b) * v2;
	}

	__device__ __inline__ Real TrianglePDF(const Vec3& at, const Vec3& v0, const Vec3& v1, const Vec3& v2)
	{
		Real inv_a = 2.0f / cross(v1 - v0, v2 - v0).norm();
		Real u = 0.5f * cross(at - v1, at - v2).norm() * inv_a;
		Real v = 0.5f * cross(at - v2, at - v0).norm() * inv_a;
		Real w = 1.0f - u - v;
		if (v < 0.0f || u < 0.0f || w < 0.0f) return 0.0f;
		if (v > 1.0f || u > 1.0f || w > 1.0f) return 0.0f;
		return inv_a;
	}

	__device__ __inline__ Vec3 HemishpereUniform(curandState* state)
	{
		Real Xi1 = curand_uniform(state);
		Real Xi2 = curand_uniform(state);

		Real theta = acosf(Xi1);
		Real phi = 2.0f * MathConst::PI * Xi2;
		Real x = sinf(theta) * cosf(phi);
		Real y = cosf(theta);
		Real z = sinf(theta) * cosf(phi);

		return Vec3(x, y, z);
	}

	__device__ __inline__ Real HemishpereUniformPDF(const Vec3& dir)
	{
		Real f = (dir.y < 0.0f);
		return (1.0f / (2.0f * MathConst::PI)) * f;
	}

	__device__ __inline__ Vec3 HemisphereCosine(curandState* state)
	{
		Real phi = 2.0f * MathConst::PI * curand_uniform(state);
		Real cosTheta = sqrtf(curand_uniform(state));
		Real sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

		Real x = cosf(phi) * sinTheta;
		Real z = sinf(phi) * sinTheta;
		Real y = cosTheta;
		return Vec3(x, y, z);
	}

	__device__ __inline__ Real HemishpereCosinePDF(const Vec3& dir)
	{
		Real f = dir.y < 0.0f;
		return (dir.y / MathConst::PI) * f;
	}
}


#endif