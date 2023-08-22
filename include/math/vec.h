#pragma once

#ifndef Vec_H
#define Vec_H

#include <cstdint>

#ifdef __CUDACC__ 
#define CUDA_CALLABLE __host__ __device__
#include "intellisense_cuda.h"

//template<typename T>
//CUDA_CALLABLE inline T frsqrt(T x)
//{
//	if constexpr (sizeof(T) == sizeof(float))
//	{
//		return rqsrtf(x);
//	}
//	else if constexpr (sizeof(T) == sizeof(double))
//	{
//		return rqsrt(x);
//	}
//}

CUDA_CALLABLE inline float frsqrt(float num)
{
	int32_t i;
	float x2, y;
	const float threeHalfs = 1.5f;

	x2 = num * 0.5f;
	y = num;
	i = *(int32_t*)&y;
	i = 0x5f3759df - (i >> 1);
	y = *(float*)&i;
	y = y * (threeHalfs - (x2 * y * y));
	return y;
}

#else
#define CUDA_CALLABLE 
#include <algorithm>

CUDA_CALLABLE inline float frsqrt(float num)
{
	int32_t i;
	float x2, y;
	const float threeHalfs = 1.5f;

	x2 = num * 0.5f;
	y = num;
	i = *(int32_t*)&y;
	i = 0x5f3759df - (i >> 1);
	y = *(float*)&i;
	y = y * (threeHalfs - (x2 * y * y));
	return y;
}
#endif

typedef float Real;
#define REAL_MAX FLT_MAX	
typedef unsigned int Index;

namespace MathConst
{
constexpr Real PI = 3.141592653589793f;
constexpr Real InvPI = 1.0f / PI;
constexpr Real Rad = PI / 180.f;
constexpr Real Deg = 180.f / PI;
constexpr Real Delta = 1e-4f;
}

template<typename T>
CUDA_CALLABLE inline T max(T x, T y) { return x > y ? x : y; }
template<typename T>
CUDA_CALLABLE inline T min(T x, T y) { return x < y ? x : y; }
template<typename T>
CUDA_CALLABLE inline T clamp(T x, T upperBound, T lowerBound) { return max(min(x, upperBound), lowerBound); }

struct Vec2
{
	CUDA_CALLABLE inline Vec2() : x(0.0f), y(0.0f) {}
	CUDA_CALLABLE inline explicit Vec2(Real _x) : x(_x), y(_x) {}
	CUDA_CALLABLE inline Vec2(Real _x, Real _y) : x(_x), y(_y) {}

	CUDA_CALLABLE inline Real operator[](Index index) const { return (&x)[index]; }
	CUDA_CALLABLE inline Real& operator[](Index index) { return (&x)[index]; }
	CUDA_CALLABLE inline Vec2 reciprocal() const { return Vec2(1.0f / x, 1.0f / y); }
	CUDA_CALLABLE inline Real norm2() const { return x * x + y * y; }
	CUDA_CALLABLE inline Real norm() const { return sqrt(norm2()); }
	CUDA_CALLABLE inline Vec2 habs() const { return Vec2(abs(x), abs(y)); }

	Real x, y;
};

CUDA_CALLABLE inline Vec2 operator+(const Vec2& lhs, const Vec2& rhs) { return Vec2(lhs.x + rhs.x, lhs.y + rhs.y); }
CUDA_CALLABLE inline Vec2 operator-(const Vec2& lhs, const Vec2& rhs) { return Vec2(lhs.x - rhs.x, lhs.y - rhs.y); }
CUDA_CALLABLE inline Vec2 operator-(const Vec2& v) { return Vec2(-v.x, -v.y); }
CUDA_CALLABLE inline Vec2 operator*(Real lhs, const Vec2& rhs) { return Vec2(lhs * rhs.x, lhs * rhs.y); }
CUDA_CALLABLE inline Vec2 operator*(const Vec2& lhs, Real rhs) { return Vec2(lhs.x * rhs, lhs.y * rhs); }
CUDA_CALLABLE inline Vec2 operator*(const Vec2& lhs, const Vec2& rhs) { return Vec2(lhs.x * rhs.x, lhs.y * rhs.y); }
CUDA_CALLABLE inline Vec2 operator/(const Vec2& lhs, Real rhs) { return (1.0f / rhs) * lhs; }

CUDA_CALLABLE inline Vec2& operator+=(Vec2& lhs, const Vec2& rhs) { return lhs = lhs + rhs; }
CUDA_CALLABLE inline Vec2& operator-=(Vec2& lhs, const Vec2& rhs) { return lhs = lhs - rhs; }
CUDA_CALLABLE inline Vec2& operator*=(Vec2& lhs, Real rhs) { return lhs = lhs * rhs; }
CUDA_CALLABLE inline Vec2& operator*=(Vec2& lhs, const Vec2& rhs) { return lhs = lhs * rhs; }
CUDA_CALLABLE inline Vec2& operator/=(Vec2& lhs, Real rhs) { return lhs = lhs / rhs; }

CUDA_CALLABLE inline Real dot(const Vec2& lhs, const Vec2& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y; }
CUDA_CALLABLE inline Real cross(const Vec2& lhs, const Vec2& rhs) { return lhs.x * rhs.y - lhs.y * rhs.x; }
CUDA_CALLABLE inline Vec2 hmax(const Vec2& lhs, const Vec2& rhs) { return Vec2(max(lhs.x, rhs.x), max(lhs.y, rhs.y)); }
CUDA_CALLABLE inline Vec2 hmin(const Vec2& lhs, const Vec2& rhs) { return Vec2(min(lhs.x, rhs.x), min(lhs.y, rhs.y)); }
CUDA_CALLABLE inline Vec2 normalize(const Vec2& v) { return v * frsqrt(v.norm2()); }

struct Vec3
{
	CUDA_CALLABLE inline Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
	CUDA_CALLABLE inline explicit Vec3(Real _x) : x(_x), y(_x), z(_x) {}
	CUDA_CALLABLE inline Vec3(Vec2 v2, Real _z) : x(v2.x), y(v2.y), z(_z) {}
	CUDA_CALLABLE inline Vec3(Real _x, Real _y, Real _z) : x(_x), y(_y), z(_z) {}

	CUDA_CALLABLE inline Real operator[](Index index) const { return (&x)[index]; }
	CUDA_CALLABLE inline Real& operator[](Index index) { return (&x)[index]; }
	CUDA_CALLABLE inline Vec3 reciprocal() const { return Vec3(1.0f / x, 1.0f / y, 1.0f / z); }
	CUDA_CALLABLE inline Real norm2() const { return x * x + y * y + z * z; }
	CUDA_CALLABLE inline Real norm() const { return sqrt(norm2()); }
	CUDA_CALLABLE inline Vec3 habs() const { return Vec3(abs(x), abs(y), abs(z)); }

	Real x, y, z;
};

CUDA_CALLABLE inline Vec3 operator+(const Vec3& lhs, const Vec3& rhs) { return Vec3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
CUDA_CALLABLE inline Vec3 operator-(const Vec3& lhs, const Vec3& rhs) { return Vec3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
CUDA_CALLABLE inline Vec3 operator-(const Vec3& v) { return Vec3(-v.x, -v.y, -v.z); }
CUDA_CALLABLE inline Vec3 operator*(Real lhs, const Vec3& rhs) { return Vec3(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z); }
CUDA_CALLABLE inline Vec3 operator*(const Vec3& lhs, Real rhs) { return Vec3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs); }
CUDA_CALLABLE inline Vec3 operator*(const Vec3& lhs, const Vec3& rhs) { return Vec3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z); }
CUDA_CALLABLE inline Vec3 operator/(const Vec3& lhs, Real rhs) { return (1.0f / rhs) * lhs; }

CUDA_CALLABLE inline Vec3& operator+=(Vec3& lhs, const Vec3& rhs) { return lhs = lhs + rhs; }
CUDA_CALLABLE inline Vec3& operator-=(Vec3& lhs, const Vec3& rhs) { return lhs = lhs - rhs; }
CUDA_CALLABLE inline Vec3& operator*=(Vec3& lhs, Real rhs) { return lhs = lhs * rhs; }
CUDA_CALLABLE inline Vec3& operator*=(Vec3& lhs, const Vec3& rhs) { return lhs = lhs * rhs; }
CUDA_CALLABLE inline Vec3& operator/=(Vec3& lhs, Real rhs) { return lhs = lhs / rhs; }

CUDA_CALLABLE inline Real dot(const Vec3& lhs, const Vec3& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
CUDA_CALLABLE inline Vec3 cross(const Vec3& lhs, const Vec3& rhs) { return Vec3(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x); }
CUDA_CALLABLE inline Vec3 hmax(const Vec3& lhs, const Vec3& rhs) { return Vec3(max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)); }
CUDA_CALLABLE inline Vec3 hmin(const Vec3& lhs, const Vec3& rhs) { return Vec3(min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)); }
CUDA_CALLABLE inline Vec3 normalize(const Vec3& v) { return v * frsqrt(v.norm2()); }

struct Vec4
{
	CUDA_CALLABLE inline Vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
	CUDA_CALLABLE inline explicit Vec4(Real _x) : x(_x), y(_x), z(_x), w(_x) {}
	CUDA_CALLABLE inline Vec4(Vec3 v3, Real _w) : x(v3.x), y(v3.y), z(v3.z), w(_w) {}
	CUDA_CALLABLE inline Vec4(Real _x, Real _y, Real _z, Real _w) : x(_x), y(_y), z(_z), w(_w) {}

	CUDA_CALLABLE inline Real operator[](Index index) const { return (&x)[index]; }
	CUDA_CALLABLE inline Real& operator[](Index index) { return (&x)[index]; }
	CUDA_CALLABLE inline Vec4 reciprocal() const { return Vec4(1.0f / x, 1.0f / y, 1.0f / z, 1.0f / w); }
	CUDA_CALLABLE inline Vec3 xyz() const { return Vec3(x, y, z); }
	CUDA_CALLABLE inline Real norm2() const { return x * x + y * y + z * z + w * w; }
	CUDA_CALLABLE inline Real norm() const { return sqrtf(norm2()); }
	CUDA_CALLABLE inline Vec4 habs() const { return Vec4(abs(x), abs(y), abs(z), abs(w)); }

	Real x, y, z, w;
};

CUDA_CALLABLE inline Vec4 operator+(const Vec4& lhs, const Vec4& rhs) { return Vec4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }
CUDA_CALLABLE inline Vec4 operator-(const Vec4& lhs, const Vec4& rhs) { return Vec4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }
CUDA_CALLABLE inline Vec4 operator-(const Vec4& v) { return Vec4(-v.x, -v.y, -v.z, -v.w); }
CUDA_CALLABLE inline Vec4 operator*(Real lhs, const Vec4& rhs) { return Vec4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w); }
CUDA_CALLABLE inline Vec4 operator*(const Vec4& lhs, Real rhs) { return Vec4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs); }
CUDA_CALLABLE inline Vec4 operator*(const Vec4& lhs, const Vec4& rhs) { return Vec4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w); }
CUDA_CALLABLE inline Vec4 operator/(const Vec4& lhs, Real rhs) { return (1.0f / rhs) * lhs; }

CUDA_CALLABLE inline Vec4& operator+=(Vec4& lhs, const Vec4& rhs) { return lhs = lhs + rhs; }
CUDA_CALLABLE inline Vec4& operator-=(Vec4& lhs, const Vec4& rhs) { return lhs = lhs - rhs; }
CUDA_CALLABLE inline Vec4& operator*=(Vec4& lhs, Real rhs) { return lhs = lhs * rhs; }
CUDA_CALLABLE inline Vec4& operator*=(Vec4& lhs, const Vec4& rhs) { return lhs = lhs * rhs; }
CUDA_CALLABLE inline Vec4& operator/=(Vec4& lhs, Real rhs) { return lhs = lhs / rhs; }

CUDA_CALLABLE inline Real dot(const Vec4& lhs, const Vec4& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w; }
CUDA_CALLABLE inline Vec4 hmax(const Vec4& lhs, const Vec4& rhs) { return Vec4(max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z), max(lhs.w, rhs.w)); }
CUDA_CALLABLE inline Vec4 hmin(const Vec4& lhs, const Vec4& rhs) { return Vec4(min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z), min(lhs.w, rhs.w)); }
CUDA_CALLABLE inline Vec4 normalize(const Vec4& v) { return v * frsqrt(v.norm2()); }

#endif