#pragma once

#ifndef RAY_H
#define RAY_H

#include "math/vec.h"

struct Ray
{
	CUDA_CALLABLE inline Ray(): m_origin(0.0f), m_direction(0.0f) {}
	CUDA_CALLABLE inline Ray(const Vec3& o, const Vec3& d) :
	m_origin(o), m_direction(d) {}

	Vec3 m_origin;
	Vec3 m_direction;
};

struct Trace
{
	Real distance;
	int hitIdx;
	Vec3 normal;
};

#endif