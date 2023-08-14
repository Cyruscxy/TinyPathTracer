#include "path_tracer.h"
#include "intellisense_cuda.h"
#include "ray.h"
#include "bvh.cuh"
#include "geometry_queries.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <time.h>

/*
 * Kernel Functions Start
 */
constexpr int BLK_DIM = 16;
constexpr int BLK_SIZE = BLK_DIM * BLK_DIM;
constexpr int DEPTH_TRACE = 8;

__global__ void setupRandSeed(size_t seed, curandState* state, size_t n)
{
	size_t tid = threadIdx.x + threadIdx.y * blockDim.x;
	if ( tid < n ) curand_init(seed, tid, 0, &state[tid]);
}

__global__ void sampleRay(Ray* rays, int width, int height, curandState* globalState, Real vFov, Real aspectRatio)
{
	int tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	int bid = blockIdx.x + gridDim.x * blockIdx.y;
	int tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;
	int pixelX = threadIdx.x + blockDim.x * blockIdx.x;
	int pixelY = threadIdx.y + blockDim.y * blockIdx.y;

	if (pixelX >= width || pixelY >= height) return;

	curandState* localState = globalState + tidLocal;
	Ray* localRay = &rays[pixelX + pixelY * width];

	Vec2 pixelLocation = DeviceSampler::RectUniform(localState);
	Real pdf = DeviceSampler::RectPDF(pixelLocation);
	pixelLocation += Vec2(pixelX, pixelY); // screen space

	Vec2 invScale(1.0f / width, 1.0f / height);
	pixelLocation *= invScale; // normalized device space

	Real tan_vFov = tanf(MathConst::Rad * vFov * 0.5f);
	Real sensorHeight = 2.0f * tan_vFov;
	Real sensorWidth = aspectRatio * sensorHeight;
	Vec3 sensorPlaneConner(-0.5f * sensorWidth, -0.5f * sensorHeight, -1.0f);
	Vec3 sensorPlaneRightUp(sensorWidth, sensorHeight, 0.0f);

	localRay->m_direction = sensorPlaneConner + sensorPlaneRightUp * Vec3(pixelLocation, 0.0f);
	localRay->m_direction = normalize(localRay->m_direction);
	localRay->m_origin = Vec3(0.0f);
	localRay->m_distBound = Vec2(0.0f, REAL_MAX);
	localRay->m_distance = 0.0f;
}

CUDA_CALLABLE inline
int traverseBVH(Ray& ray, BVHNode* nodes, Vec3* vertices, uint32_t* indices, int size, Real& dist)
{
	int callStacks[64];
	int stackPtr = 1;
	callStacks[0] = 0;

	int hitIdx = -1;

	while (stackPtr > 0)
	{
		int currentIdx = callStacks[stackPtr - 1];
		stackPtr--;

		if (currentIdx >= size - 1) // is leaf node
		{
			int fid = nodes[currentIdx].info.leaf.fid;
			auto vid0 = indices[3 * fid + 0];
			auto vid1 = indices[3 * fid + 1];
			auto vid2 = indices[3 * fid + 2];
			Real hitDist;
			// Shall we do the ray-triangle test here, or later after all collected ?
			if (rayHitTriangle(ray, vertices[vid0], vertices[vid1], vertices[vid2], hitDist))
			{
				if (hitDist < dist) { dist = hitDist; hitIdx = currentIdx; }
			}
		}
		else
		{
			int leftChild = nodes[currentIdx].info.intern.leftChild;
			int rightChild = nodes[currentIdx].info.intern.rightChild;
			if (rayHitBBox(ray, nodes[leftChild].box))
			{
				callStacks[stackPtr] = leftChild;
				stackPtr++;
			}
			if (rayHitBBox(ray, nodes[rightChild].box))
			{
				callStacks[stackPtr] = leftChild;
				stackPtr++;
			}
		}
	}

	return hitIdx;
}

// worry about warp divergence
CUDA_CALLABLE inline
int mtlBinarySearch(int index, MtlInterval* mtlLUT, int objCnt)
{
	// binary search
	int left = 0;
	int right = objCnt - 1;
	while (right != left)
	{
		int mid = (left + right) >> 1;
		if (index < mtlLUT[mid].begin) right = mid;
		else left = mid;
	}
	return mtlLUT[right].mtlIdx;
}

CUDA_CALLABLE inline
int mtlLinearSearch(int index, MtlInterval* mtlLUT, int objCnt)
{
	int i = 0;
	for (; i < objCnt; ++i)
	{
		if (index >= mtlLUT[i].begin) break;
	}
	return mtlLUT[i].mtlIdx;
}

CUDA_CALLABLE inline
Vec3 getBaryCoords(Vec3 v0, Vec3 v1, Vec3 v2, Vec3 pos)
{
	Real gamma = cross(pos - v0, v1 - v0).norm();
	Real alpha = cross(pos - v1, v2 - v1).norm();
	Real beta = cross(pos - v2, v0 - v2).norm();
	Real invSum = 1.0f / (alpha + beta + gamma);
	alpha *= invSum;
	beta *= invSum;
	gamma *= invSum;
	return Vec3(alpha, beta, gamma);
}

CUDA_CALLABLE inline
Vec3 reflect(Vec3 dir, Vec3 normal)
{
	return dir - 2.0f * dot(dir, normal) * normal;
}

CUDA_CALLABLE inline
Vec3 refract(Vec3 dir, Vec3 normal, Real ior)
{
	Real cosThetaI = dot(dir, normal);
	// eta_i / eta_t
	Real eta = cosThetaI > 0.0f ? ior : 1.0f / ior;

	// cosThetaI = abs(cosThetaI);
	Real sin2ThetaI = 1.0f - cosThetaI * cosThetaI;
	Real sin2ThetaT = eta * eta * sin2ThetaI;

	// Check if TIR? 
	if (sin2ThetaT > 1.0f) return reflect(dir, normal);

	Real cosThetaT = sqrtf(1.0f - sin2ThetaT);
	cosThetaT = cosThetaI < 0.0f ? -cosThetaT : cosThetaT;
	Real invEta = 1.0f / eta;
	return invEta * dir - (cosThetaI * invEta + cosThetaT) * normal;
}

CUDA_CALLABLE inline
Real shlickFresnel(Real u)
{
	Real m = min(1.0f, max(0.0f, 1.0f - u));
	Real m2 = m * m;
	return m2 * m2 * m;
}

__device__ __inline__
Real getNewDirection(Ray& ray, Vec3 normal, Material* mtl, curandState* state, Vec3& nextDir)
{
	// get new dir
	Vec3 out_dir;
	Real probability;
	if (mtl->eta > 0.0f) {
		// TODO: Add Fresnel effect
		out_dir = refract(ray.m_direction, normal, mtl->eta);
		probability = 1.0f;
	}
	else if ( mtl->metallic > 0.0f )
	{
		out_dir = reflect(ray.m_direction, normal);
		probability = 1.0f;
	}
	else
	{
		out_dir = DeviceSampler::HemishpereUniform(state, normal);
		probability = DeviceSampler::HemishpereUniformPDF(out_dir, normal);
	}
	nextDir = out_dir;
	return probability;
}

__device__ __inline__
int findIdxOfTrans(int tid, MtlInterval* interval, int objCnt)
{
	int i = 0;
	for (; i < objCnt; ++i)
	{
		if (tid >= interval[i].begin) break;
	}
	return i;
}

__global__
void transform(Vec3* vertices, Vec3* normals, uint32_t* indices, MtlInterval* interval, Mat4* vertTrans, 
	Mat4* normalTrans, Vec3* wVertices, Vec3* wNormals, int size, int nObjs)
{
	int tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	int bid = blockIdx.x + gridDim.x * blockIdx.y;
	int tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;
	if (tidGlobal >= size) return;

	int idxTrans = findIdxOfTrans(tidGlobal, interval, nObjs);
	Mat4 transMat = vertTrans[idxTrans];
#pragma unroll
	for ( int i = 0; i < 3; ++i )
	{
		uint32_t vId = indices[3 * tidGlobal + i];
		wVertices[vId] = (transMat * Vec4(vertices[vId], 1.0f)).xyz();
	}
	transMat = normalTrans[idxTrans];
#pragma unroll
	for (int i = 0; i < 3; ++i)
	{
		uint32_t vId = indices[3 * tidGlobal + i];
		wNormals[vId] = (transMat * Vec4(normals[vId], 0.0f)).xyz();
	}
}

__global__
void trace(BVHNode* nodes, Vec3* vertices, Vec3* normals, uint32_t* indices, MtlInterval* mtlLUT, Material* materials, 
	Spectrum* color, Ray* rays, curandState* globalState, int width, int height, int size, int objCnt)
{
	int tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	int bid = blockIdx.x + gridDim.x * blockIdx.y;
	int tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;
	int pixelX = threadIdx.x + blockDim.x * blockIdx.x;
	int pixelY = threadIdx.y + blockDim.y * blockIdx.y;

	__shared__ Real scratch[BLK_SIZE * DEPTH_TRACE];

	if (pixelX >= width || pixelY >= height) return;

	auto localState = globalState + tidLocal;
	auto pStack = scratch + tidLocal * DEPTH_TRACE;
	int mtlIdxStack[DEPTH_TRACE];
	Spectrum directLightStack[DEPTH_TRACE];

	int depth = 0;
	while (depth < DEPTH_TRACE)
	{
		Real dist = REAL_MAX;
		Ray localRay = rays[pixelX + pixelY * width];
		int hitIdx = traverseBVH(localRay, nodes, vertices, indices, size, dist);

		if (hitIdx < 0) break; 

		int mtlIndex = mtlBinarySearch(hitIdx, mtlLUT, objCnt);

		uint32_t i0 = indices[hitIdx * 3 + 0];
		uint32_t i1 = indices[hitIdx * 3 + 1];
		uint32_t i2 = indices[hitIdx * 3 + 2];
		Vec3 v0 = vertices[i0];
		Vec3 v1 = vertices[i1];
		Vec3 v2 = vertices[i2];
		Vec3 hitPos = localRay.m_origin + localRay.m_direction * localRay.m_distance;
		Vec3 baryCoords = getBaryCoords(v0, v1, v2, hitPos);

		v0 = normals[i0];
		v1 = normals[i1];
		v2 = normals[i2];
		Vec3 actualNormal = baryCoords.x * v0 + baryCoords.y * v1 + baryCoords.z * v2;

		Vec3 nextDir;
		Real probability = getNewDirection(localRay, actualNormal, materials + mtlIndex, localState, nextDir);
		localRay.m_origin = hitPos;

		// sample direct light
		{
			// TODO: Sample delta light
			Vec3 directDir;
			Real directProb = getNewDirection(localRay, actualNormal, materials + mtlIndex, localState, directDir);
			localRay.m_direction = directDir;
			hitIdx = traverseBVH(localRay, nodes, vertices, indices, size, dist);
			if ( hitIdx >= 0 )
			{
				int directMtlIdx = mtlBinarySearch(hitIdx, mtlLUT, objCnt);
				directLightStack[depth] = materials[directMtlIdx].baseColor * materials[directMtlIdx].emissionFactor / directProb;
			}
			else
			{
				directLightStack[depth] = Spectrum(0.0f, 0.0f, 0.0f);
			}
		}

		pStack[depth] = probability;
		mtlIdxStack[depth] = mtlIndex;
		localRay.m_direction = nextDir;
		depth++;
	}

	// sample indirect light
	depth--;
	Spectrum radiance;
	while ( depth >= 0 )
	{
		Material* mtl = &materials[mtlIdxStack[depth]];
		Spectrum emission;
		if ( mtl->emissionFactor > 0.0f ) emission = mtl->baseColor * mtl->emissionFactor; 

		radiance = (emission + (directLightStack[depth] + radiance) * mtl->baseColor) / pStack[depth];
		depth--;
	}

	color[pixelX + pixelY * width] += radiance;
}

__global__
void copyToFB(Spectrum* samples, unsigned char* data, int height, int width, int nSamplesPerPixel)
{
	int pixelX = threadIdx.x + blockDim.x * blockIdx.x;
	int pixelY = threadIdx.y + blockDim.y * blockIdx.y;
	if (pixelX >= width || pixelY >= height) return;

	int offset = pixelY * width + pixelX;
	Spectrum* localSamples = &samples[offset];
	Spectrum color;
	 
	for ( int i = 0; i < nSamplesPerPixel; ++i )
	{
		color += localSamples[i];
	}
	uchar3 rgb = color.toUChar();

	unsigned char* pixel = &data[offset];
	pixel[0] = rgb.x;
	pixel[1] = rgb.y;
	pixel[2] = rgb.z;
}

/*
 * Kernel Functions End
 */

PathTracer::PathTracer() :
m_displayer(),
m_width(m_displayer.m_windowExtent.width),
m_height(m_displayer.m_windowExtent.height),
m_rays(m_width * m_height),
m_randStates(m_width * m_height) {}

void PathTracer::doTrace(DeviceScene& dScene, Camera& camera, int nSamplesPerPixel)
{
	time_t tseed;
	time(&tseed);

	size_t nNormals = dScene.normals.size();
	size_t nVerts = dScene.vertices.size();
	size_t nFaces = dScene.indices.size() / 3;
	int nObjs = dScene.materialsLUT.size();

	Real vFov = camera.getVFov();
	Real aspRatio = camera.getAspRatio();

	unsigned char* framebuffer = m_displayer.m_cudaFramebufferMem;

	curandState* dp_states = thrust::raw_pointer_cast(m_randStates.data());
	int nPixels = m_height * m_width;
	setupRandSeed KERNEL_DIM((nPixels + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE) (tseed, dp_states, nPixels);

	thrust::device_vector<Spectrum> radiance(nPixels);
	thrust::device_vector<Vec3> wVertices(nVerts);
	thrust::device_vector<Vec3> wNormals(nNormals);

	Spectrum* dp_radiance = thrust::raw_pointer_cast(radiance.data());
	Ray* dp_rays = thrust::raw_pointer_cast(m_rays.data());
	uint32_t* dp_indices = thrust::raw_pointer_cast(dScene.indices.data());
	Vec3* dp_vertices = thrust::raw_pointer_cast(dScene.vertices.data());
	Vec3* dp_normals = thrust::raw_pointer_cast(dScene.normals.data());
	Vec2* dp_texCoords = thrust::raw_pointer_cast(dScene.texCoords.data());
	Material* dp_mtls = thrust::raw_pointer_cast(dScene.materials.data());
	MtlInterval* dp_mtlInterval = thrust::raw_pointer_cast(dScene.materialsLUT.data());
	Mat4* dp_vertTrans = thrust::raw_pointer_cast(dScene.vertTrans.data());
	Mat4* dp_normalTrans = thrust::raw_pointer_cast(dScene.normalTrans.data());
	Vec3* dp_wVertices = thrust::raw_pointer_cast(wVertices.data());
	Vec3* dp_wNormals = thrust::raw_pointer_cast(wNormals.data());

	dim3 blk_config(BLK_DIM, BLK_DIM);
	dim3 grid_config((m_width + BLK_DIM - 1) / BLK_DIM, (m_height + BLK_DIM - 1) / BLK_DIM);

	// TODO: Transform meshes into world coordinates
	transform KERNEL_DIM((nFaces + BLK_SIZE - 1 / BLK_SIZE), BLK_SIZE) (dp_vertices, dp_normals, dp_indices, 
		dp_mtlInterval, dp_vertTrans, dp_normalTrans, dp_wVertices, dp_wNormals, nFaces, nObjs);

	// constructs bvh every frame
	BVH bvh(nFaces);
	bvh.construct(dScene.vertices, dScene.indices);
	BVHNode* dp_bvhNodes = thrust::raw_pointer_cast(bvh.m_nodes.data());

	for ( int idx = 0; idx < nSamplesPerPixel; ++idx )
	{
		sampleRay KERNEL_DIM(grid_config, blk_config) (dp_rays, m_width, m_height, dp_states, vFov, aspRatio);

		trace KERNEL_DIM(grid_config, blk_config) (dp_bvhNodes, dp_vertices, dp_normals, dp_indices, dp_mtlInterval,
			dp_mtls, dp_radiance, dp_rays, dp_states, m_width, m_height, nFaces, nObjs);
	}

	copyToFB KERNEL_DIM(grid_config, blk_config) (dp_radiance, framebuffer, m_height, m_width, nSamplesPerPixel);
}

void PathTracer::render(const std::string& meshFile)
{
	Scene scene(meshFile, "gltf");
	int nSamplesPerPixel = 8;

	
}
