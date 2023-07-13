#pragma once

#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <thrust/device_vector.h>

#include "camera.h"
#include "ray.h"
#include "vkEngine.h"

class PathTracer
{
public:
	PathTracer();

	void sampleRay();
	void doTrace();

private:
	Camera	m_camera;
	VkEngine m_displayer;
	int m_width;
	int m_height;
	thrust::device_vector<Ray> m_rays;

};

#endif