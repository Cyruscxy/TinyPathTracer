#include "delta_light.h"

Incoming PointLight::sample(const Vec3& p)
{
	Incoming ret;
	ret.direction = pos - p;
	ret.distance = ret.direction.norm();
	ret.radiance = color * intensity;
	ret.direction = ret.direction / ret.distance;
	return ret;
}

