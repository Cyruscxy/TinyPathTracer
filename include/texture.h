#pragma once

#ifndef TEXTURE_H
#define TEXTURE_H

#include "picture.h"
#include <cuda_runtime.h>

class Texture : private Picture
{
public:
	Texture() = default;
	Texture(const std::string& file);
	~Texture();
	
	cudaTextureObject_t		m_cuTextureObj;

private:
	uint32_t m_numLevels;

	cudaChannelFormatDesc	m_cuChannelDesc;
	cudaTextureDesc			m_cuTextureDesc;
	cudaMipmappedArray_t	m_cuMipmapArray;
};

#endif
