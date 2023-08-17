#pragma once

#ifndef RAY_H
#define RAY_H

#include "math/vec.h"

struct Ray
{
	CUDA_CALLABLE inline Ray(): m_origin(0.0f), m_direction(0.0f), 
		m_distBound(0.0f), m_distance(0.0f) {}
	CUDA_CALLABLE inline Ray(const Vec3& o, const Vec3& d, float t) :
	m_origin(o), m_direction(d), m_distance(t), m_distBound(0.0f, REAL_MAX) {}

	Vec3 m_origin;
	Vec3 m_direction;
	Vec2 m_distBound;
	Real m_distance;
};

struct Trace
{
	Real distance;
	int hitIdx;
	Vec3 normal;
};

#endif