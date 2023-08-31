#include "env_light.cuh"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

CUDA_CALLABLE inline Real luma(unsigned char* pixel)
{
	return 0.2126f * pixel[2] + 0.7152f * pixel[1] + 0.0722f * pixel[0];
}

__global__ void computeEnvLightPDF(Real* pdf, unsigned char* data, int height, int width)
{
	int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
	int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
	if (pixelX >= width || pixelY >= height) return;
	int offset = pixelX + pixelY * width;

	Real theta = (MathConst::PI * pixelY) / height;
	pdf[offset] = luma(&data[3 * offset]) * theta;
}

__global__ void initEnvLight(Real* pdf, Real* cdf, int height, int width)
{
	int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
	int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
	if (pixelX >= width || pixelY >= height) return;
	int offset = pixelX + pixelY * width;

	Real sum = pdf[width * height - 1];
	pdf[offset] = pdf[offset] / sum;
	if (pixelX == width - 1) cdf[pixelY] = pdf[pixelY];
	__syncthreads();
	if (pixelY >= 1)
	{
		pdf[offset] -= cdf[pixelY - 1];
	}
}

EnvLight::EnvLight(const std::string& file) : m_radiance(file), m_pdf(m_radiance.m_data.size()),
m_cdf(m_radiance.m_height), m_intensity(1.0f)
{
	thrust::device_vector<unsigned char> deviceData(m_radiance.m_data.size());
	thrust::copy(m_radiance.m_data.begin(), m_radiance.m_data.end(), deviceData.begin());

	Real* dp_pdf = thrust::raw_pointer_cast(m_pdf.data());
	Real* dp_cdf = thrust::raw_pointer_cast(m_cdf.data());
	unsigned char* dp_data = thrust::raw_pointer_cast(deviceData.data());

	dim3 grid((m_radiance.m_width + 15) / 16, (m_radiance.m_height + 15) / 16);
	dim3 block(16, 16);

	computeEnvLightPDF KERNEL_DIM(grid, block) (dp_pdf, dp_data, m_radiance.m_height, m_radiance.m_width);
	thrust::inclusive_scan(thrust::device, m_pdf.begin(), m_pdf.end(), m_pdf.begin());
	initEnvLight KERNEL_DIM(grid, block) (dp_pdf, dp_cdf, m_radiance.m_height, m_radiance.m_width);
}