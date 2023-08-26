#ifndef DELTA_LIGHT
#define DELTA_LIGHT

#include "material.h"

enum DeltaLightType
{
	POINT,
	DIRECTIONAL,
	SPOT
};

struct Incoming
{
	Incoming(): distance(0.0f) {}
	Spectrum radiance;
	Vec3 direction;
	Real distance;
};

struct PointLight
{
	PointLight(): color(1.0f), intensity(1.0f) {}
	CUDA_CALLABLE Incoming sample(const Vec3& p);

	Spectrum color;
	Real intensity;
	Vec3 pos;
};

struct DirectionalLight
{
	DirectionalLight(): color(1.0f), intensity(1.0f) {}
	CUDA_CALLABLE Incoming sample();

	Spectrum color;
	Real intensity;
};

struct SpotLight
{
	SpotLight(): color(1.0f), intensity(1.0f), innerAngle(30.0f), outerAngle(45.0f) {}
	CUDA_CALLABLE Incoming sample();

	Spectrum color;
	Real intensity;
	Vec3 pos;
	Real innerAngle; // degree
	Real outerAngle; // degree
};

union Light
{
	Light(): pl() {}
	
	PointLight pl;
	DirectionalLight dl;
	SpotLight sl;
};

struct DeltaLight
{
	DeltaLight(): type(POINT), light() {}
	CUDA_CALLABLE Incoming sample();

	DeltaLightType type;
	Light light;
};

#endif