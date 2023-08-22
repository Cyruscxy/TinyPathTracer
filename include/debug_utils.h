#pragma once

#ifndef DEBUG_UTILS
#define DEBUG_UTILS

#include <thrust/device_vector.h>
#include <vector>
#include <utility>
#include "FreeImage/FreeImage.h"
#include "material.h"

template<typename  T>
void checkDeviceVector(thrust::device_vector<T>& src)
{
	std::vector<T> dst(src.size());
	thrust::copy(src.begin(), src.end(), dst.begin());
}

inline void specToImage(thrust::device_vector<Spectrum>& radiance, int width, int height) 
{
    std::vector<Spectrum> h_radiance(radiance.size());
    thrust::copy(radiance.begin(), radiance.end(), h_radiance.begin());

    FIBITMAP* tmp = FreeImage_Allocate(width, height, 24);
    if (!tmp) std::cerr << "Failed to allocate memory for result!" << std::endl;
    uint32_t pitch = FreeImage_GetPitch(tmp);

    BYTE* ptr = FreeImage_GetBits(tmp);
    for (uint32_t j = 0; j < height; ++j) {
        BYTE* pixel = (BYTE*)ptr;
        for (uint32_t i = 0; i < width; ++i) {
            auto color = (h_radiance[i + j * width] / 32).toUChar();
            pixel[0] = color.z;
            pixel[1] = color.y;
            pixel[2] = color.x;
            pixel += 3;
        }
        ptr += pitch;
    }

    std::string path;
#ifdef PATH_TO_MESH_DIR
    path += PATH_TO_MESH_DIR;
#endif

    if (!FreeImage_Save(FIF_PNG, tmp, (path + "out.png").c_str())) 
        std::cout << "Failed to save image!" << std::endl;
    FreeImage_Unload(tmp);
}

inline void checkBVHNodes(thrust::device_vector<BVHNode>& nodes)
{
    size_t nFaces = (nodes.size() + 1) / 2;
    std::vector<BVHNode> interNodes(nodes.size());
    thrust::copy(nodes.begin(), nodes.end(), interNodes.begin());
    std::vector<int> children(nodes.size(), 0);
    std::vector<int> parent(nFaces - 1, 0);
    for ( int i = 0; i < nFaces - 1; ++i )
    {
        children[interNodes[i].info.intern.leftChild] += 1;
        children[interNodes[i].info.intern.rightChild] += 1;
    }
    for ( auto& node : interNodes )
    {
        parent[node.parent] += 1;
    }

    for ( int i = 0; i < children.size(); ++i )
    {
	    if ( children[i] != 1 )
	    {
            std::cout << i << ": " << children[i] << " children" << std::endl;
	    }
    }
    for (int i = 0; i < parent.size(); ++i)
    {
	    if ( parent[i] != 2 )
	    {
            std::cout << i << ": " << parent[i] << " parent " << std::endl;
	    }
    }
    
}

inline void checkVertices(thrust::device_vector<uint32_t>& indices, thrust::device_vector<Vec3>& vertices)
{
    std::vector<uint32_t> target{ 467, 466, 469, 407, 342, 344, 346, 347, 345 };
    std::vector<uint32_t> hi(indices.size());
    std::vector<Vec3> hv(vertices.size());
    thrust::copy(indices.begin(), indices.end(), hi.begin());
    thrust::copy(vertices.begin(), vertices.end(), hv.begin());

    for ( auto fid: target )
    {
        std::cout << fid << ": ";
        for ( int i = 0; i < 3; ++i )
        {
            uint32_t id = hi[3 * fid + i];
            Vec3 v = hv[id];
            // std::cout << "(" << v.x << ", " << v.y << ", " << v.z << ") ";
            std::cout << id << " ";
        }
        std::cout << std::endl;
    }
}

inline void checkBVHVariance(thrust::device_vector<BVHNode>& currentNodes, std::vector<BVHNode>& lastNodes)
{
    std::vector<BVHNode> nodes(currentNodes.size());
    thrust::copy(currentNodes.begin(), currentNodes.end(), nodes.begin());

    if ( !lastNodes.empty() )
    {
	    for ( int i = 0; i < lastNodes.size(); ++i )
	    {
            bool equal = true;
            equal &= nodes[i].parent == lastNodes[i].parent;
            equal &= nodes[i].info.intern.leftChild == lastNodes[i].info.intern.leftChild;
            equal &= nodes[i].info.intern.rightChild == lastNodes[i].info.intern.rightChild;
            if ( !equal )
            {
                std::cout << "bvh node diff: " << i << std::endl;
            }
	    } 
    }

    lastNodes = std::move(nodes);
}

inline void checkHitStatus(thrust::device_vector<int>& hitRecord, int width, int height)
{
    std::vector<int> hits(width * height);
    thrust::copy(hitRecord.begin(), hitRecord.end(), hits.begin());

    FIBITMAP* tmp = FreeImage_Allocate(width, height, 24);
    if (!tmp) std::cerr << "Failed to allocate memory for result!" << std::endl;
    uint32_t pitch = FreeImage_GetPitch(tmp);

    BYTE* ptr = FreeImage_GetBits(tmp);
    for (uint32_t j = 0; j < height; ++j) {
        BYTE* pixel = (BYTE*)ptr;
        for (uint32_t i = 0; i < width; ++i) {
            unsigned char color;
            if ( hits[i + j * width] > -1 )
            {
                color = 125;
            }
            else
            {
                color = 0;
            }
             
            pixel[0] = color;
            pixel[1] = color;
            pixel[2] = color;
            pixel += 3;
        }
        ptr += pitch;
    }

    std::string path;
#ifdef PATH_TO_MESH_DIR
    path += PATH_TO_MESH_DIR;
#endif

    if (!FreeImage_Save(FIF_PNG, tmp, (path + "hit.png").c_str()))
        std::cout << "Failed to save image!" << std::endl;
    FreeImage_Unload(tmp);
}

#endif
