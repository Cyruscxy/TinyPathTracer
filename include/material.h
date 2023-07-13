#pragma once

#ifndef MATERIAL_H
#define MATERIAL_H

#include "math/vec.h"

struct Material
{
	Material() :
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

	Vec3 baseColor;
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