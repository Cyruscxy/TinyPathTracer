#pragma once

#ifndef PICTURE_H
#define PICTURE_H

#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

#define FREEIMAGE_COLORORDER FREEIMAGE_COLORORDER_RGB
#include "FreeImage/FreeImage.h"

struct Picture
{
	Picture() = default;
	~Picture() = default;

	Picture(const std::string& file)
	{
		auto start = file.find_last_of('.') + 1;
		auto format = file.substr(start, file.length() - start);
		std::for_each(format.begin(), format.end(), toupper);
		auto fif = FreeImage_GetFIFFromFormat(format.c_str());

		FIBITMAP* image = FreeImage_Load(fif, file.c_str());
		if ( image == nullptr )
		{
			throw std::runtime_error("Failed to open file " + file);
		}

		m_width =		FreeImage_GetWidth(image);
		m_height =		FreeImage_GetHeight(image);
		m_pitch =		FreeImage_GetPitch(image);
		m_bpp =			FreeImage_GetBPP(image);
		m_channels =	m_bpp / sizeof(BYTE);
		size_t size = (size_t)m_pitch * m_height * m_channels;
		m_data.resize(size);

		BYTE* ptr = FreeImage_GetBits(image);
		memcpy(m_data.data(), ptr, size);

		FreeImage_Unload(image);
	}

	std::vector<uint8_t> m_data;
	uint32_t m_width;
	uint32_t m_height;
	uint32_t m_pitch;
	uint32_t m_bpp;
	uint32_t m_channels;
};

#endif
