#ifndef ENV_LIGHT
#define ENV_LIGHT

#include "texture.h"
#include "material.h"
#include "intellisense_cuda.h"
#include "sampler.h"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>


__global__ void initEnvLight(Real* pdf, Real* cdf, Real* data, int height, int width)
{
	int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
	int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
	if (pixelX >= width || pixelY >= height) return;
	int offset = pixelX + pixelY * width;

	Real sum = data[width * height - 1];
	pdf[offset] = data[offset] / sum;
	if (pixelX == width - 1) cdf[pixelY] = data[pixelY];
}

class EnvLight
{
public:
	EnvLight() = default;
	EnvLight(const std::string& file) : m_radiance(file), m_pdf(m_radiance.m_data.size()),
		m_cdf(m_radiance.m_height), m_importance(m_radiance.m_data.size()), m_intensity(1.0f)
	{
		thrust::device_vector<unsigned char> deviceData(m_radiance.m_data.size());
		thrust::copy(m_radiance.m_data.begin(), m_radiance.m_data.end(), deviceData.begin());

		Real* dp_pdf = thrust::raw_pointer_cast(m_pdf.data());
		Real* dp_cdf = thrust::raw_pointer_cast(m_cdf.data());
		Real* dp_importance = thrust::raw_pointer_cast(m_importance.data());

		struct UChar2FLT
		{
			CUDA_CALLABLE Real operator()(unsigned char x) const { return (Real)x; }
		};

		thrust::transform(deviceData.begin(), deviceData.end(), m_importance.begin(), UChar2FLT());
		thrust::inclusive_scan(thrust::device, m_importance.begin(), m_importance.end(), m_importance.begin());

		dim3 grid((m_radiance.m_width + 15) / 16, (m_radiance.m_height + 15) / 16);
		dim3 block(16, 16);
		initEnvLight KERNEL_DIM(grid, block) (dp_pdf, dp_cdf, dp_importance, m_radiance.m_height, m_radiance.m_width);
	}

	Texture m_radiance;
	thrust::device_vector<Real> m_importance;
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

CUDA_CALLABLE int log2Ceil(int x)
{
	int bits = 1;
	while (x > 0)
	{
		x >>= 1;
		bits <<= 1;
	}
	return bits;
}

CUDA_CALLABLE int upper_bound(Real* data, Real ub, int size)
{
	int bound = log2Ceil(size);
	int t = 0;
	for ( int i = bound >> 1 ; i > 0; i >>= 1 )
	{
		if ( t + i >= size ) continue;
		if (data[t + i] <= ub) t += i;
	}
	return t;
}

CUDA_CALLABLE inline Vec3 EnvLightSample(curandState* state, int width, int height, Real* pdf, Real* cdf)
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

#endif