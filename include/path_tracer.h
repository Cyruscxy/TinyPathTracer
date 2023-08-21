#pragma once

#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <thrust/device_vector.h>
#include "sampler.h"
#include "camera.h"
#include "ray.h"
#include "mesh.cuh"
#include "vkEngine.h"
#include "bvh.cuh"


class PathTracer
{
public:
	PathTracer();
	
	void render(const std::string& meshFile);
private:
	VkEngine m_displayer;
	int m_width;
	int m_height;
	thrust::device_vector<curandState> m_randStates;
	thrust::device_vector<Spectrum> m_radiance;
	thrust::device_vector<Vec3> m_wVertices;
	thrust::device_vector<Vec3> m_wNormals;
	thrust::device_vector<Ray> m_rays;
	std::vector<BVHNode> m_lastNodes;

	void doTrace(DeviceScene& scene, Camera& camera, unsigned char* framebuffer, int nSamplesPerPixel);
};

#endif