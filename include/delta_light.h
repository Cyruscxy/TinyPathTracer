#ifndef DELTA_LIGHT
#define DELTA_LIGHT

#include "material.h"

constexpr Real LUMENS_PER_WATTS = 683.0f;
constexpr Real WATTS_PER_LUMEN = 1.0f / 683.0f;

enum DeltaLightType
{
	POINT_LIGHT,
	DIRECTIONAL_LIGHT,
	SPOT_LIGHT
};

struct Incoming
{
	CUDA_CALLABLE Incoming(): distance(0.0f) {}
	Spectrum radiance;
	Vec3 direction;
	Real distance;
};

// calculate the attenuated radiance for distance
CUDA_CALLABLE inline void CalcDistAttenuation(Incoming& light)
{
	// default radius: 1.0f
	constexpr Real invRadiusSqr = 0.01f;
	Real distanceSqr = light.distance * light.distance;
	Real distanceAttenuation = 1.0f / (distanceSqr + 1.0f);
	distanceAttenuation *= square(saturate(1.0f - square(distanceSqr * invRadiusSqr)));
	light.radiance *= distanceAttenuation;
}

struct PointLight
{
	CUDA_CALLABLE PointLight(): color(1.0f), intensity(1.0f) {}
	CUDA_CALLABLE inline Incoming sample(const Vec3& p)
	{
		Incoming ret;
		ret.direction = this->pos - p;
		ret.distance = ret.direction.norm();
		ret.direction = ret.direction / ret.distance;
		ret.radiance = this->color * this->intensity;
		return ret;
	}

	Spectrum color;
	Real intensity;
	Vec3 pos;
};

struct DirectionalLight
{
	CUDA_CALLABLE DirectionalLight(): color(1.0f), intensity(1.0f) {}
	CUDA_CALLABLE inline Incoming sample(const Vec3& p)
	{
		Incoming ret;
		ret.direction = -this->direction;
		ret.radiance = this->color * this->intensity;
		ret.distance = 0.0f;
		return ret;
	}

	Spectrum color;
	Real intensity;
	Vec3 direction;
};

struct SpotLight
{
	CUDA_CALLABLE SpotLight(): color(1.0f), intensity(1.0f), cosOuterAngle(0.0f), invCosConeDifference(0.0f) {}
	CUDA_CALLABLE inline Incoming sample(const Vec3& p)
	{
		Incoming ret;
		ret.direction = this->pos - p;
		ret.distance = ret.direction.norm();
		ret.direction = ret.direction / ret.distance;

		// spot attenuation
		Real cosTheta = dot(-ret.direction, this->direction);
		Real coneAngleFalloff = square(saturate(cosTheta - this->cosOuterAngle) * this->invCosConeDifference);

		ret.radiance = this->color * this->intensity * coneAngleFalloff;
		return ret;
	}

	Spectrum color;
	Real intensity;
	Vec3 pos;
	Vec3 direction;
	Real cosOuterAngle;
	Real invCosConeDifference;
};

union Light
{
	CUDA_CALLABLE Light(): pl() {}
	
	PointLight pl;
	DirectionalLight dl;
	SpotLight sl;
};

struct DeltaLight
{
	CUDA_CALLABLE DeltaLight(): type(POINT_LIGHT), light() {}
	CUDA_CALLABLE inline Incoming sample(const Vec3& p)
	{
		Incoming ret;
		if (this->type == DeltaLightType::POINT_LIGHT)
		{
			ret = this->light.pl.sample(p);
		}
		else if (this->type == DeltaLightType::DIRECTIONAL_LIGHT)
		{
			ret = this->light.dl.sample(p);
		}
		else if (this->type == DeltaLightType::SPOT_LIGHT)
		{
			ret = this->light.sl.sample(p);
		}

		CalcDistAttenuation(ret);
		return ret;
	}

	DeltaLightType type;
	Light light;
};

#endif