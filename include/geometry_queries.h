#pragma once

#ifndef GEOMETRY_QUERIES_H
#define GEOMETRY_QUERIES_H

#include "math/vec.h"
#include "ray.h"
#include "primitives.h"

template<typename T>
CUDA_CALLABLE inline void swap(T& t1, T& t2)
{
	T tmp = t2;
	t2 = t1;
	t1 = tmp;
}

CUDA_CALLABLE inline bool rayHitBBox(Ray& ray, BBox& box)
{
	Vec3 invDirection = ray.m_direction.reciprocal();
	Vec2 tx, ty, tz; // Vec2(min, max) pair
	Vec2 times(REAL_MAX, -REAL_MAX);

	tx.x = (box.m_min.x - ray.m_origin.x) * invDirection.x;
	tx.y = (box.m_max.x - ray.m_origin.x) * invDirection.x;
	if (tx.x > tx.y) swap(tx.x, tx.y);
	if (times.x > tx.y || tx.x > times.y) return false;
	times.x = max(times.x, tx.x);
	times.y = min(times.y, tx.y);

	ty.x = (box.m_min.y - ray.m_origin.y) * invDirection.y;
	ty.y = (box.m_max.y - ray.m_origin.y) * invDirection.y;
	if (ty.x > ty.y) swap(ty.x, ty.y);
	if (times.x > ty.y || ty.x > times.y) return false;
	times.x = max(times.x, ty.x);
	times.y = min(times.y, ty.y);

	tz.x = (box.m_min.z - ray.m_origin.z) * invDirection.z;
	tz.y = (box.m_max.z - ray.m_origin.z) * invDirection.z;
	if (tz.x > tz.y) swap(tz.x, tz.y);
	if (times.x > tz.y || tz.x > times.y) return false;
	times.x = max(times.x, tz.x);
	times.y = min(times.y, tz.y);

	return true;
}

CUDA_CALLABLE inline bool rayHitSphere(Ray& ray, Real raidus, Vec2& times)
{
	// local space
	// solve equation: a * t^2 + b * t + c = 0
	Real a = ray.m_direction.norm2();
	Real b = -2.0f * dot(ray.m_origin, ray.m_direction);
	Real c = ray.m_origin.norm2() - raidus * raidus;

	Real delta = b * b - 4.0f * a * c;
	if (delta <= 0) return false;

	Real tmin = -b - sqrt(delta);
	Real tmax = -b + sqrt(delta);
	if (tmin >= ray.m_distBound.x && tmin <= ray.m_distBound.y) times.x = tmin;
	else if (tmax > ray.m_distBound.y || tmax < ray.m_distBound.x) times.y = tmax;
	else return false;

	return true;
}

CUDA_CALLABLE inline bool rayHitTriangle(Ray& ray, const Vec3& v0, const Vec3& v1, const Vec3& v2, Real& dist)
{
	Vec3 e1 = v1 - v0;
	Vec3 e2 = v2 - v1;
	Vec3 t = ray.m_origin - v0;
	Vec3 p = cross(ray.m_direction, e2);
	Vec3 q = cross(t, e1);

	Real denom = dot(p, e1);
	if (denom == 0.0f) return false;

	Real u = dot(p, t) / denom;
	Real v = dot(q, ray.m_direction) / denom;
	if (u < 0.0f || v < 0.0f || u + v > 1.0f) return false;

	dist = dot(q, e2);
	return true;
}

#endif