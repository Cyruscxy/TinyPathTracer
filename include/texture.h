#pragma once

#ifndef TEXTURE_H
#define TEXTURE_H

#include "picture.h"
#include <cuda_runtime.h>

class Texture : public Picture
{
public:
	Texture() = default;
	Texture(const std::string& file);
	Texture(Texture&& other);
	void operator=(Texture&& other);
	~Texture();

	cudaTextureObject_t getTexture() { return m_cuTextureObj; }
	
	uint32_t m_numLevels;

private:
	cudaTextureObject_t		m_cuTextureObj;
	cudaChannelFormatDesc	m_cuChannelDesc;
	cudaTextureDesc			m_cuTextureDesc;
	cudaMipmappedArray_t	m_cuMipmapArray;
};

#endif
