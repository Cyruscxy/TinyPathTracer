#pragma once

#ifndef MESH_H
#define MESH_H

#include <string>
#include <vector>
#include <map>
#include <thrust/device_vector.h>
#include <array>
#include "math/vec.h"
#include "material.h"
#include "camera.h"

struct Mesh
{
	Mesh() = default;

	inline void readVerticesFromRawPtr(float* ptr, size_t count)
	{
		m_vertexCoords.resize(count);
		for ( size_t i = 0; i < count; ++i )
		{
			m_vertexCoords[i] = Vec3(
				static_cast<Real>(ptr[3 * i + 0]),
				static_cast<Real>(ptr[3 * i + 1]),
				static_cast<Real>(ptr[3 * i + 2]));
		}
	}

	inline void readNormalsFromRawPtr(float* ptr, size_t count)
	{
		m_vertexNormals.resize(count);
		for (size_t i = 0; i < count; ++i)
		{
			m_vertexNormals[i] = Vec3(
				static_cast<Real>(ptr[3 * i + 0]),
				static_cast<Real>(ptr[3 * i + 1]),
				static_cast<Real>(ptr[3 * i + 2]));
		}
	}

	inline void readTexcoordsFromRawPtr(float* ptr, size_t count)
	{
		m_vertexTexCoords.resize(count);
		for (size_t i = 0; i < count; ++i)
		{
			m_vertexTexCoords[i] = Vec2(
				static_cast<Real>(ptr[2 * i + 0]),
				static_cast<Real>(ptr[2 * i + 1]));
		}
	}

	template<typename IndexType>
	inline void readIndicesFromRawPtr(IndexType* ptr, size_t count)
	{
		m_polygons.resize(count);
		for (size_t i = 0; i < count; ++i)
		{
			m_polygons[i] = static_cast<uint32_t>(ptr[i]);
		}
	}
	
	std::vector<uint32_t> m_polygons;
	std::vector<Vec3> m_vertexCoords;
	std::vector<Vec3> m_vertexNormals;
	std::vector<Vec2> m_vertexTexCoords;
	std::string m_material;
	std::shared_ptr<Transform> m_transform;
};

// closed interval
// use binary search to look up
struct MtlInterval
{
	int begin;
	int mtlIdx;
};

struct DeviceScene
{
	thrust::device_vector<uint32_t> indices;
	thrust::device_vector<Vec3> vertices;
	thrust::device_vector<Vec3> normals;
	thrust::device_vector<Vec2> texCoords;

	thrust::device_vector<Material> materials;
	thrust::device_vector<MtlInterval> materialsLUT;
};

class Scene
{
public:
	Scene() = default;
	Scene(const std::string& filename, const std::string& type);

	DeviceScene copySceneToDevice();

private:
	void readFromGLTF(const std::string& filename);

private:
	Camera m_camera;
	std::vector<Mesh> m_meshes;
	std::map<std::string, Material> m_materials;

	// TODO: Add model hierarchy support. 
};

#endif