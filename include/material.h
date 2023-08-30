#pragma once

#ifndef MATERIAL_H
#define MATERIAL_H

#include "math/vec.h"
#include "assert.h"
#include <vector>

struct Spectrum
{
	CUDA_CALLABLE Spectrum(): r(0.0f), g(0.0f), b(0.0f) {}
	CUDA_CALLABLE Spectrum(Real _r, Real _g, Real _b): r(_r), g(_g), b(_b) {}
	CUDA_CALLABLE Spectrum(Real f): r(f), g(f), b(f) {}
	CUDA_CALLABLE Spectrum(Vec3 c): r(c.x), g(c.y), b(c.z) {}
	Spectrum(std::vector<float> c)
	{
		assert(c.size() == 3);
		r = c[0]; g = c[1]; b = c[2];
	}
	Spectrum(std::vector<double> c)
	{
		assert(c.size() == 3);
		r = static_cast<Real>(c[0]);
		g = static_cast<Real>(c[1]);
		b = static_cast<Real>(c[2]);
	}

	CUDA_CALLABLE inline Spectrum operator+=(Spectrum c)
	{
		r += c.r; g += c.g; b += c.b;
		return *this;
	}
	CUDA_CALLABLE inline Spectrum operator-=(Spectrum c)
	{
		r -= c.r; g -= c.g; b -= c.b;
		return *this;
	}
	CUDA_CALLABLE inline Spectrum operator*=(Spectrum c)
	{
		r *= c.r; g *= c.g; b *= c.b;
		return *this;
	}
	CUDA_CALLABLE inline Spectrum operator/=(Real s)
	{
		Real invS = 1.0f / s; r /= invS; g /= invS; b /= invS;
		return *this;
	}
	CUDA_CALLABLE inline Spectrum operator*=(Real s)
	{
		r *= s; g *= s; b *= s;
		return *this;
	}
	CUDA_CALLABLE inline Spectrum operator+(Spectrum c)
	{
		return Spectrum(r + c.r, g + c.g, b + c.b);
	}
	CUDA_CALLABLE inline Spectrum operator-(Spectrum c)
	{
		return Spectrum(r - c.r, g - c.g, b - c.b);
	}
	CUDA_CALLABLE inline Spectrum operator*(Spectrum c)
	{
		return Spectrum(r * c.r, g * c.g, b * c.b);
	}
	CUDA_CALLABLE inline Spectrum operator*(Real s)
	{
		return Spectrum(r * s, g * s, b * s);
	}
	CUDA_CALLABLE inline Spectrum operator/(Real s)
	{
		return *this * (1.0f / s);
	}
	CUDA_CALLABLE inline uchar3 toUChar()
	{
		uchar3 rgb;
		rgb.x = (unsigned char)(clamp(r * 255.0f, 255.0f, 0.0f));
		rgb.y = (unsigned char)(clamp(g * 255.0f, 255.0f, 0.0f));
		rgb.z = (unsigned char)(clamp(b * 255.0f, 255.0f, 0.0f));
		return rgb;
	}

	Real r, g, b;
};

struct Material
{
	CUDA_CALLABLE inline Material() :
	baseColor(0.82f, 0.67f, 0.16f),
	//absorption(0.0f),
	emissionFactor(0.0f),
	eta(0.0f),
	metallic(0.0f),
	subsurface(0.0f),
	specular(0.5f),
	roughness(0.5f),
	specularTint(0.0f),
	anisotropic(0.0f),
	sheen(0.0f),
	sheenTint(0.0f),
	clearcoat(0.0f),
	clearcoatGloss(1.0f)
	{ }

	Spectrum baseColor;
	//Vec3 absorption;

	Real emissionFactor;
	Real eta;
	Real metallic;
	Real subsurface;
	Real specular;
	Real roughness;
	Real specularTint;
	Real anisotropic;
	Real sheen;
	Real sheenTint;
	Real clearcoat;
	Real clearcoatGloss;
};


#endif