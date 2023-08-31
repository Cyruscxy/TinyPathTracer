#include <cmath>
#include "texture.h"
#include "mesh.cuh"

#include "intellisense_cuda.h"

inline void CudaCheck(cudaError status)
{
	if ( status != cudaSuccess )
	{
		throw std::runtime_error("CUDA Error! Error code: " + std::to_string(status));
	}
}

__global__ void textureDownsampling(
	cudaSurfaceObject_t texOut,
	cudaTextureObject_t texIn,
	uint32_t width,
	uint32_t height
) {
	uint32_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;
	if (pixelX >= width || pixelY >= height) return;

	float inv_w = 1.0f / float(width);
	float inv_h = 1.0f / float(height);

	uchar4 rgba = tex2D<uchar4>(texIn, inv_w * (0.5f + pixelX) , inv_h * (0.5f + pixelY));

	surf2Dwrite(rgba, texOut, pixelX * sizeof(uchar4), pixelY);
}

struct addAlpha
{
	CUDA_CALLABLE addAlpha() = default;

	__host__ __device__
	uchar4 operator()(uchar3 bgr)
	{
		uchar4 rgba;
		rgba.x = bgr.z;
		rgba.y = bgr.y;
		rgba.z = bgr.x;
		rgba.w = 255;
		return rgba;
	}
};

struct BGR2RGB
{
	CUDA_CALLABLE BGR2RGB() = default;

	__host__ __device__
	uchar4 operator()(uchar4 bgra) {
		uchar4 rgba;
		rgba.x = bgra.z;
		rgba.y = bgra.y;
		rgba.z = bgra.x;
		rgba.w = bgra.w;
		return rgba;
	}
};

Texture::Texture(const std::string& file) : Picture(file)
{
	thrust::device_vector<uchar4> deviceData(m_width * m_height);
	if ( m_channels == 3 )
	{
		thrust::device_vector<uchar3> deviceOldData(m_width * m_height);
		thrust::copy((uchar3*)m_data.data(), ((uchar3*)m_data.data()) + m_width * m_height, deviceOldData.begin());
		thrust::transform(deviceOldData.begin(), deviceOldData.end(), deviceData.begin(), addAlpha());
	}
	else if ( m_channels == 4 )
	{
		thrust::device_vector<uchar4> deviceOldData(m_width* m_height);
		thrust::copy((uchar4*)m_data.data(), ((uchar4*)m_data.data()) + m_width * m_height, deviceOldData.begin());
		thrust::transform(deviceOldData.begin(), deviceOldData.end(), deviceData.begin(), BGR2RGB());
	}
	else
	{
		throw std::runtime_error("Failed to create texture! Choose picture with 3 or 4 channels.");
	}

	auto baseLevelData = thrust::raw_pointer_cast(deviceData.data());
	m_numLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(m_width, m_height))));

	cudaExtent extent = make_cudaExtent(m_width, m_height, 0);
	m_cuChannelDesc = cudaCreateChannelDesc<uchar4>();

	CudaCheck(cudaMallocMipmappedArray(&m_cuMipmapArray, &m_cuChannelDesc, extent, m_numLevels));

	cudaArray_t baseLevel;
	CudaCheck(cudaGetMipmappedArrayLevel(&baseLevel, m_cuMipmapArray, 0));

	cudaMemcpy3DParms copyParams{};
	copyParams.srcPtr = make_cudaPitchedPtr(baseLevelData, m_width * 4, m_width, m_height);
	copyParams.dstArray = baseLevel;
	copyParams.extent.width = m_width;
	copyParams.extent.height = m_height;
	copyParams.extent.depth = 1;
	copyParams.kind = cudaMemcpyDeviceToDevice;
	CudaCheck(cudaMemcpy3D(&copyParams));

	// generate multilevel mipmap
	for ( uint32_t level = 1; level < m_numLevels; ++level )
	{
		cudaArray_t levelFrom;
		cudaArray_t levelTo;

		CudaCheck(cudaGetMipmappedArrayLevel(&levelTo, m_cuMipmapArray, level));
		CudaCheck(cudaGetMipmappedArrayLevel(&levelFrom, m_cuMipmapArray, level - 1));

		cudaExtent extentLevelTo{};
		CudaCheck(cudaArrayGetInfo(nullptr, &extentLevelTo, nullptr, levelTo));

		cudaExtent extentLevelFrom{};
		CudaCheck(cudaArrayGetInfo(nullptr, &extentLevelFrom, nullptr, levelFrom));

		uint32_t width = m_width >> level;
		uint32_t height = m_height >> level;

		cudaTextureObject_t texIn;

		cudaResourceDesc texResDesc{};
		memset(&texResDesc, 0, sizeof(cudaResourceDesc));
		texResDesc.resType = cudaResourceTypeArray;
		texResDesc.res.array.array = levelFrom;

		cudaTextureDesc texDesc{};
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.normalizedCoords = 1;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeClamp;
		texDesc.readMode = cudaReadModeElementType;

		CudaCheck(cudaCreateTextureObject(&texIn, &texResDesc, &texDesc, nullptr));

		cudaSurfaceObject_t surfaceOut;
		cudaResourceDesc surfDesc{};
		memset(&surfDesc, 0, sizeof(cudaResourceDesc));
		surfDesc.resType = cudaResourceTypeArray;
		surfDesc.res.array.array = levelTo;
		CudaCheck(cudaCreateSurfaceObject(&surfaceOut, &surfDesc));

		dim3 blkDim(16, 16);
		dim3 gridDim((width + blkDim.x - 1) / blkDim.x, (height + blkDim.y - 1) / blkDim.y);
		textureDownsampling KERNEL_DIM(gridDim, blkDim) (surfaceOut, texIn, width, height);

		CudaCheck(cudaDeviceSynchronize());
		CudaCheck(cudaDestroyTextureObject(texIn));
		CudaCheck(cudaDestroySurfaceObject(surfaceOut));
	}

	// create mipmap texture
	cudaResourceDesc mipmapTexResDes{};
	mipmapTexResDes.resType = cudaResourceTypeMipmappedArray;
	mipmapTexResDes.res.mipmap.mipmap = m_cuMipmapArray;

	memset(&m_cuTextureDesc, 0, sizeof(cudaTextureDesc));
	m_cuTextureDesc.normalizedCoords = 1;
	m_cuTextureDesc.filterMode = cudaFilterModePoint;
	m_cuTextureDesc.addressMode[0] = cudaAddressModeClamp;
	m_cuTextureDesc.addressMode[1] = cudaAddressModeClamp;
	m_cuTextureDesc.addressMode[2] = cudaAddressModeClamp;
	m_cuTextureDesc.maxMipmapLevelClamp = m_numLevels - 1;
	m_cuTextureDesc.readMode = cudaReadModeElementType;

	CudaCheck(cudaCreateTextureObject(&m_cuTextureObj, &mipmapTexResDes, &m_cuTextureDesc, nullptr));
}

Texture::Texture(Texture&& other)
{
	m_numLevels = other.m_numLevels;
	m_cuTextureObj = other.m_cuTextureObj;
	m_cuChannelDesc = other.m_cuChannelDesc;
	m_cuMipmapArray = other.m_cuMipmapArray;
	m_cuTextureDesc = other.m_cuTextureDesc;

	other.m_cuMipmapArray = nullptr;
	other.m_cuTextureObj = 0;
}

void Texture::operator=(Texture&& other)
{
	m_numLevels = other.m_numLevels;
	m_cuTextureObj = other.m_cuTextureObj;
	m_cuChannelDesc = other.m_cuChannelDesc;
	m_cuMipmapArray = other.m_cuMipmapArray;
	m_cuTextureDesc = other.m_cuTextureDesc;

	other.m_cuMipmapArray = nullptr;
	other.m_cuTextureObj = 0;
}

Texture::~Texture()
{
	if ( m_cuMipmapArray )
	{
		CudaCheck(cudaFreeMipmappedArray(m_cuMipmapArray));
	}
	if ( m_cuTextureObj )
	{
		CudaCheck(cudaDestroyTextureObject(m_cuTextureObj));
	}
}
