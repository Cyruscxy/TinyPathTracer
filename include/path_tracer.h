#pragma once

#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <thrust/device_vector.h>
#include "sampler.h"
#include "camera.h"
#include "ray.h"
#include "mesh.cuh"
#include "vkEngine.h"


class PathTracer
{
public:
	PathTracer();
	
	void render(const std::string& meshFile);
private:
	VkEngine m_displayer;
	int m_width;
	int m_height;
	thrust::device_vector<Ray> m_rays;
	thrust::device_vector<curandState> m_randStates;

	void doTrace(DeviceScene& scene, Camera& camera, int nSamplesPerPixel);
};

#endif