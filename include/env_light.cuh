#ifndef ENV_LIGHT
#define ENV_LIGHT

#include "texture.h"
#include "intellisense_cuda.h"
#include "sampler.h"
#include <thrust/device_vector.h>

class EnvLight
{
public:
	EnvLight() = default;
	EnvLight(const std::string& file);

	Texture m_radiance;
	thrust::device_vector<Real> m_pdf;
	thrust::device_vector<Real> m_cdf;
	Real m_intensity;
};

CUDA_CALLABLE inline Real EnvLightPDF(Vec3 dir, int width, int height, Real* pdf)
{
	Real cosTheta = dir.y;
	Real theta = acos(cosTheta);
	Real sinTheta = sqrt(1.0f - cosTheta * cosTheta);
	Real phi = acos(dir.x / sinTheta);

	int x = (int)round((phi / (2.0f * MathConst::PI)) * width);
	int y = (int)round((theta / MathConst::PI) * height);

	Real p = pdf[x + y * width];
	return p;
}

CUDA_CALLABLE inline int BinarySearchBound(int x)
{
	int bits = 1;
	while (x > 0)
	{
		x >>= 1;
		bits <<= 1;
	}
	return bits;
}

CUDA_CALLABLE inline int upper_bound(Real* data, Real ub, int size)
{
	int bound = BinarySearchBound(size);
	int t = 0;
	for ( int i = bound >> 1 ; i > 0; i >>= 1 )
	{
		if ( t + i >= size ) continue;
		if (data[t + i] <= ub) t += i;
	}
	return t;
}

__device__ __inline__ Vec3 EnvLightSample(curandState* state, int width, int height, Real* pdf, Real* cdf)
{
	int thetaIdx = upper_bound(cdf, DeviceSampler::Unitform(state), height);
	int phiIdx = upper_bound(pdf + thetaIdx * width, DeviceSampler::Unitform(state) * cdf[thetaIdx], width);

	Real theta = (float)thetaIdx * MathConst::PI / height;
	Real phi = (float)phiIdx * MathConst::PI * 2.0f / width;
	Vec3 dir;
	dir.x = sin(theta) * cos(phi);
	dir.z = sin(theta) * sin(phi);
	dir.y = cos(theta);
	return dir;
}

CUDA_CALLABLE inline Vec2 Vec2UV(const Vec3& dir)
{
	Real u = atan2(dir.z, dir.x) / (2.0f * MathConst::PI);
	if (u < 0.0f) u += 1.0f;
	Real v = 1.0f - acos(clamp(dir.y, 1.0f, -1.0f)) / MathConst::PI;
	return Vec2(u, v);
}

#endif