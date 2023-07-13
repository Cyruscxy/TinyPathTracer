#pragma once

#ifndef MAT_H
#define MAT_H

#include "vec.h"

struct Mat2;
struct Mat3;
struct Mat4;

template<typename T> CUDA_CALLABLE constexpr int getDim();
template<> CUDA_CALLABLE constexpr int getDim<Mat2>() { return 2; }
template<> CUDA_CALLABLE constexpr int getDim<Mat3>() { return 3; }
template<> CUDA_CALLABLE constexpr int getDim<Mat4>() { return 4; }

template<typename T, int N = getDim<T>()>
CUDA_CALLABLE inline T MatrixMultiply(const T& lhs, const T& rhs)
{
	T res(0.0f);
#pragma unroll(N)
	for ( int i = 0; i < N; ++i )
	{
#pragma unroll(N)
		for ( int j = 0; j < N; ++j )
		{
#pragma unroll(N)
			for ( int k = 0; k < N; ++k )
			{
				res[j][i] += lhs[k][i] * rhs[j][k];
			}
		}
	}
	return res;
}

template<typename T_M, typename T_V, int N = getDim<T_M>()>
CUDA_CALLABLE inline T_V MatrixVectorMultiply(const T_M& lhs, const T_V& rhs)
{
	T_V res(0.0f);
#pragma unroll(N)
	for (int i = 0; i < N; ++i)
	{
#pragma unroll(N)
		for (int j = 0; j < N; ++j)
		{
			res[i] += lhs[j][i] * rhs[j];
		}
	}
	return res;
}

template<typename T, int N = getDim<T>()>
CUDA_CALLABLE inline T MatrixTranspose(const T& m)
{
	T res(0.0f);
#pragma unroll(N)
	for (int i = 0; i < N; ++i)
	{
#pragma unroll(N)
		for (int j = 0; j < N; ++j)
		{
			res[j][i] = m[i][j];
		}
	}
	return res;
}

struct Mat2
{
	CUDA_CALLABLE inline Mat2() = default;
	CUDA_CALLABLE inline Mat2(Real r) : cols{ Vec2(r), Vec2(r) } {}
	CUDA_CALLABLE inline Mat2(Vec2 c1, Vec2 c2) : cols{ c1, c2 } {}

	CUDA_CALLABLE inline Vec2 operator[](Index index) const { return cols[index]; }
	CUDA_CALLABLE inline Vec2& operator[](Index index) { return cols[index]; }
	CUDA_CALLABLE inline Mat2 transpose() const;
	CUDA_CALLABLE inline Real determinant() const;
	CUDA_CALLABLE inline Mat2 inverse() const;
	CUDA_CALLABLE inline static Mat2 Identity() { return Mat2(Vec2(1.0f, 0.0f), Vec2(0.0f, 1.0f)); }
	CUDA_CALLABLE inline static Mat2 Scale(Vec2 s);

	// row major
	Vec2 cols[2];
};

CUDA_CALLABLE inline Mat2 operator+(const Mat2& lhs, const Mat2& rhs) { return Mat2(lhs[0] + rhs[0], lhs[1] + rhs[1]); }
CUDA_CALLABLE inline Mat2 operator-(const Mat2& lhs, const Mat2& rhs) { return Mat2(lhs[0] - rhs[0], lhs[1] - rhs[1]); }
CUDA_CALLABLE inline Mat2 operator*(const Mat2& lhs, const Mat2& rhs) { return MatrixMultiply(lhs, rhs); }
CUDA_CALLABLE inline Vec2 operator*(const Mat2& lhs, const Vec2& rhs) { return MatrixVectorMultiply(lhs, rhs); }
CUDA_CALLABLE inline Mat2 operator*(const Mat2& lhs, Real s) { return Mat2(lhs[0] * s, lhs[1] * s); }
CUDA_CALLABLE inline Mat2 operator*(Real s, const Mat2& rhs) { return Mat2(rhs[0] * s, rhs[1] * s); }
CUDA_CALLABLE inline Mat2 operator-(const Mat2& m) { return Mat2(m[0] * -1.0f, m[1] * -1.0f); }
CUDA_CALLABLE inline Mat2& operator+=(Mat2& lhs, const Mat2& rhs) { return lhs = lhs + rhs; }
CUDA_CALLABLE inline Mat2& operator-=(Mat2& lhs, const Mat2& rhs) { return lhs = lhs - rhs; }
CUDA_CALLABLE inline Mat2& operator*=(Mat2& lhs, const Mat2& rhs) { return lhs = lhs * rhs; }

CUDA_CALLABLE inline Mat2 Mat2::transpose() const { return MatrixTranspose(*this); }
CUDA_CALLABLE inline Real Mat2::determinant() const { return cross(cols[0], cols[1]); }
CUDA_CALLABLE inline Mat2 Mat2::inverse() const
{
	auto det = determinant();
	if (det == 0.0f) return *this;
	return (-1.0f / det) * Mat2(Vec2(cols[0][0], -cols[1][0]), Vec2(-cols[0][1], cols[1][1]));
}
CUDA_CALLABLE inline Mat2 Mat2::Scale(Vec2 s)
{
	auto res = Identity();
	res[0][0] = s.x;
	res[1][1] = s.y;
	return res;
}

struct Mat3
{
	CUDA_CALLABLE inline Mat3() = default;
	CUDA_CALLABLE inline Mat3(Real r) : cols{ Vec3(r), Vec3(r), Vec3(r)} {}
	CUDA_CALLABLE inline Mat3(Vec3 c1, Vec3 c2, Vec3 c3) : cols{ c1, c2, c3 } {}

	CUDA_CALLABLE inline Vec3 operator[](Index index) const { return cols[index]; }
	CUDA_CALLABLE inline Vec3& operator[](Index index) { return cols[index]; }
	CUDA_CALLABLE inline Mat3 transpose() const;
	CUDA_CALLABLE inline Real determinant() const;
	CUDA_CALLABLE inline Mat3 inverse() const;
	CUDA_CALLABLE inline static Mat3 Identity() { return Mat3(Vec3(1.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f)); }
	CUDA_CALLABLE inline static Mat3 Scale(Vec3 s);

	Vec3 cols[3];
};

CUDA_CALLABLE inline Mat3 operator+(const Mat3& lhs, const Mat3& rhs) { return Mat3(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]); }
CUDA_CALLABLE inline Mat3 operator-(const Mat3& lhs, const Mat3& rhs) { return Mat3(lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]); }
CUDA_CALLABLE inline Mat3 operator*(const Mat3& lhs, const Mat3& rhs) { return MatrixMultiply(lhs, rhs); }
CUDA_CALLABLE inline Vec3 operator*(const Mat3& lhs, const Vec3& rhs) { return MatrixVectorMultiply(lhs, rhs); }
CUDA_CALLABLE inline Mat3 operator*(const Mat3& lhs, Real s) { return Mat3(lhs[0] * s, lhs[1] * s, lhs[2] * s); }
CUDA_CALLABLE inline Mat3 operator*(Real s, const Mat3& rhs) { return Mat3(rhs[0] * s, rhs[1] * s, rhs[2] * s); }
CUDA_CALLABLE inline Mat3 operator-(const Mat3& m) { return Mat3(m[0] * -1.0f, m[1] * -1.0f, m[2] * -1.0f); }
CUDA_CALLABLE inline Mat3& operator+=(Mat3& lhs, const Mat3& rhs) { return lhs = lhs + rhs; }
CUDA_CALLABLE inline Mat3& operator-=(Mat3& lhs, const Mat3& rhs) { return lhs = lhs - rhs; }
CUDA_CALLABLE inline Mat3& operator*=(Mat3& lhs, const Mat3& rhs) { return lhs = lhs * rhs; }

CUDA_CALLABLE inline Mat3 Mat3::transpose() const { return MatrixTranspose(*this); }
CUDA_CALLABLE inline Real Mat3::determinant() const { return dot(cols[0], cross(cols[1], cols[2])); }
CUDA_CALLABLE inline Mat3 Mat3::inverse() const
{
	auto det = determinant();
	if (det == 0.0f) return *this;
	Mat3 res;
	res[0][0] = cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1];
	res[1][0] = cols[2][0] * cols[1][2] - cols[1][0] * cols[2][2];
	res[2][0] = cols[1][0] * cols[2][1] - cols[2][0] * cols[1][1];
	res[0][1] = cols[2][1] * cols[0][2] - cols[0][1] * cols[2][2];
	res[1][1] = cols[0][0] * cols[2][2] - cols[2][0] * cols[0][2];
	res[2][1] = cols[0][1] * cols[2][0] - cols[0][0] * cols[2][1];
	res[0][2] = cols[0][1] * cols[1][2] - cols[0][2] * cols[1][1];
	res[1][2] = cols[0][2] * cols[1][0] - cols[0][0] * cols[1][2];
	res[2][2] = cols[0][0] * cols[1][1] - cols[0][1] * cols[1][0];
	res *= (-1.0f / det);
	return res;
}
CUDA_CALLABLE inline Mat3 Mat3::Scale(Vec3 s)
{
	auto res = Identity();
	res[0][0] = s.x;
	res[1][1] = s.y;
	res[2][2] = s.z;
	return res;
}

struct Mat4
{
	CUDA_CALLABLE inline Mat4() = default;
	CUDA_CALLABLE inline Mat4(Real r) : cols{ Vec4(r), Vec4(r), Vec4(r), Vec4(r)} {}
	CUDA_CALLABLE inline Mat4(Vec4 c1, Vec4 c2, Vec4 c3, Vec4 c4) : cols{ c1, c2, c3, c4 } {}
	CUDA_CALLABLE inline Mat4(const Mat3& m) : Mat4(Vec4(m[0], 0.0f), Vec4(m[1], 0.0f), Vec4(m[2], 0.0f), Vec4(Vec3(0.0f), 1.0f)) {}

	CUDA_CALLABLE inline Vec4 operator[](Index index) const { return cols[index]; }
	CUDA_CALLABLE inline Vec4& operator[](Index index) { return cols[index]; }
	CUDA_CALLABLE inline Mat4 transpose() const;
	CUDA_CALLABLE inline Mat4 static Identity() { return Mat4(Vec4(1.0f, 0.0f, 0.0f, 0.0f), Vec4(0.0f, 1.0f, 0.0f, 0.0f), Vec4(0.0f, 0.0f, 1.0f, 0.0f), Vec4(0.0f, 0.0f, 0.0f, 1.0f)); }
	CUDA_CALLABLE inline Mat4 static Scale(Vec3 s) { return Mat4(Mat3::Scale(s)); }
	CUDA_CALLABLE inline Mat4 static Scale(Vec4 s);
	CUDA_CALLABLE inline Mat4 static Translate(const Vec3& loc);

	Vec4 cols[4];
};

CUDA_CALLABLE inline Mat4 operator+(const Mat4& lhs, const Mat4& rhs) { return Mat4(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3]); }
CUDA_CALLABLE inline Mat4 operator-(const Mat4& lhs, const Mat4& rhs) { return Mat4(lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3]); }
CUDA_CALLABLE inline Mat4 operator*(const Mat4& lhs, const Mat4& rhs) { return MatrixMultiply(lhs, rhs); }
CUDA_CALLABLE inline Vec4 operator*(const Mat4& lhs, const Vec4& rhs) { return MatrixVectorMultiply(lhs, rhs); }
CUDA_CALLABLE inline Mat4 operator*(const Mat4& lhs, Real s) { return Mat4(lhs[0] * s, lhs[1] * s, lhs[2] * s, lhs[3] * s); }
CUDA_CALLABLE inline Mat4 operator*(Real s, const Mat4& rhs) { return Mat4(rhs[0] * s, rhs[1] * s, rhs[2] * s, rhs[3] * s); }
CUDA_CALLABLE inline Mat4 operator-(const Mat4& m) { return Mat4(m[0] * -1.0f, m[1] * -1.0f, m[2] * -1.0f, m[3] * -1.0f); }
CUDA_CALLABLE inline Mat4& operator+=(Mat4& lhs, const Mat4& rhs) { return lhs = lhs + rhs; }
CUDA_CALLABLE inline Mat4& operator-=(Mat4& lhs, const Mat4& rhs) { return lhs = lhs - rhs; }
CUDA_CALLABLE inline Mat4& operator*=(Mat4& lhs, const Mat4& rhs) { return lhs = lhs * rhs; }

CUDA_CALLABLE inline Mat4 Mat4::transpose() const { return MatrixTranspose(*this); }
CUDA_CALLABLE inline Mat4 Mat4::Translate(const Vec3& loc)
{
	auto res = Mat4::Identity();
	res[3] = Vec4(loc, 1.0f);
	return res;
}
CUDA_CALLABLE inline Mat4 Mat4::Scale(Vec4 s)
{
	auto res = Identity();
#pragma unroll(4)
	for ( int i = 0; i < 4; ++i )
	{
		res[i][i] = s[i];
	}
	return res;
}

#endif
