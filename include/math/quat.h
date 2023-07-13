#pragma once

#ifndef QUAT_H
#define QUAT_H

#include "math/mat.h"

struct Quat
{
	CUDA_CALLABLE inline Quat() : w(0.0f), v(0.0f) {}
	CUDA_CALLABLE inline Quat(Real _x, Real _y, Real _z, Real _w) : w(_w), v(_x, _y, _z) {}
	CUDA_CALLABLE inline Quat(Real _w, const Vec3& _v) : w(_w), v(_v) {}
	CUDA_CALLABLE inline Quat(const Vec3& angle)
	{
		// the order is Rz * Rx * Ry
		auto cx = cos(0.5f * angle.x * MathConst::Rad);
		auto cy = cos(0.5f * angle.y * MathConst::Rad);
		auto cz = cos(0.5f * angle.z * MathConst::Rad);
		auto sx = sin(0.5f * angle.x * MathConst::Rad);
		auto sy = sin(0.5f * angle.y * MathConst::Rad);
		auto sz = sin(0.5f * angle.z * MathConst::Rad);

		w = cx * cy * cz - sx * sy * sz;
		v.x = sx * cy * cz - cx * sy * sz;
		v.y = cx * sy * cz + sx * cy * sz;
		v.z = sx * sy * cz + cx * cy * sz;
	}
	CUDA_CALLABLE inline Real norm2() const;
	CUDA_CALLABLE inline Real norm() const;
	CUDA_CALLABLE inline Quat inverse() const;
	CUDA_CALLABLE inline static Mat3 RotateFromQuat(const Quat& q);

	Real w;
	Vec3 v;
};

CUDA_CALLABLE inline Quat operator+(const Quat& lhs, const Quat& rhs) { return Quat(lhs.w + rhs.w, lhs.v + rhs.v); }
CUDA_CALLABLE inline Quat operator-(const Quat& lhs, const Quat& rhs) { return Quat(lhs.w - rhs.w, lhs.v - rhs.v); }
CUDA_CALLABLE inline Quat operator*(const Quat& lhs, const Quat& rhs)
{
	Quat res;
	res.w = lhs.w * rhs.w - dot(lhs.v, rhs.v);
	res.v = lhs.w * rhs.v + rhs.w * lhs.v + cross(lhs.v, rhs.v);
	return res;
}
CUDA_CALLABLE inline Quat operator*(const Quat& lhs, Real s) { return Quat(s * lhs.w, s * lhs.v); }
CUDA_CALLABLE inline Quat operator*(Real s, const Quat& rhs) { return Quat(s * rhs.w, s * rhs.v); }
CUDA_CALLABLE inline Real Quat::norm2() const { return w * w + v.norm2(); }
CUDA_CALLABLE inline Real Quat::norm() const { return sqrt(norm2()); }
CUDA_CALLABLE inline Quat Quat::inverse() const { return Quat(w, -v); }

CUDA_CALLABLE inline Mat3 Quat::RotateFromQuat(const Quat& q)
{
	auto x2 = q.v.x * q.v.x;
	auto y2 = q.v.y * q.v.y;
	auto z2 = q.v.z * q.v.z;
	auto xy = q.v.x * q.v.y;
	auto xz = q.v.x * q.v.z;
	auto yz = q.v.y * q.v.z;
	auto wx = q.w * q.v.x;
	auto wy = q.w * q.v.y;
	auto wz = q.w * q.v.z;
	return Mat3 (
		Vec3 ( 1.0f - 2.0f * (y2 - z2), 2.0f * (xy + wz), 2.0f * (xz - wy) ),
		Vec3 { 2.0f * (xy - wz), 1.0f - 2.0f * (x2 - z2), 2.0f * (yz + wz) },
		Vec3 { 2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (x2 - y2) }
	);
}

#endif
