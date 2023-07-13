#pragma once

#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "math/vec.h"



struct Sphere
{
	Vec3 m_origin;
	Real m_radius;

	BBox box() { return BBox(m_origin - Vec3(m_radius), m_origin + Vec3(m_radius)); }
};


#endif
