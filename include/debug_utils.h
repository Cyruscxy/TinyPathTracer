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
            auto color = h_radiance[i + j * width].toUChar();
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
    std::vector<BVHNode> interNodes(nFaces - 1);
    thrust::copy(nodes.begin(), nodes.begin() + nFaces - 1, interNodes.begin());
    std::vector<int> children(nodes.size(), 0);
    int i = 0;
    for ( auto& node : interNodes)
    {
        children[node.info.intern.leftChild] += 1;
        children[node.info.intern.rightChild] += 1;
        if (node.info.intern.leftChild == 2) std::cout << i << " left" << std::endl;
        if (node.info.intern.rightChild == 2) std::cout << i << " right" << std::endl;
        ++i;
    }

    /*for ( int i = 0; i < children.size(); ++i )
    {
	    if ( children[i] > 1 )
	    {
            std::cout << i << ": " << children[i] << std::endl;
	    }
    }*/
}

#endif
