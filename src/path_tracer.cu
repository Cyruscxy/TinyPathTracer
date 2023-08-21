#include "path_tracer.h"
#include "intellisense_cuda.h"
#include "geometry_queries.h"
#include <cuda_runtime.h>
#include <thrust/device_malloc.h>
#include <thrust/count.h>
#include <cstdint>
#include <time.h>


namespace 
{
	// For debug
	constexpr bool RENDER_NORMAL = true;

	constexpr int BLK_DIM = 16;
	constexpr int BLK_SIZE = BLK_DIM * BLK_DIM;
	constexpr int DEPTH_TRACE = 8;

	struct HitStatus
	{
		CUDA_CALLABLE HitStatus(): hitIdx(-1), hitDist(REAL_MAX), uv() {}

		int hitIdx;
		Real hitDist;
		Vec2 uv;
	};
}

#include "debug_utils.h"

/*
 * Kernel Functions Start
 */
__global__ void setupRandSeed(size_t seed, curandState* state, size_t n)
{
	size_t tid = threadIdx.x + threadIdx.y * blockDim.x;
	if ( tid < n ) curand_init(seed, tid, 0, &state[tid]);
}

__global__ void sampleRay(Ray* rays, int width, int height, curandState* globalState, Real vFov, 
	Real aspectRatio, Real nearPlane, Mat4* cameraToWorld)
{
	int tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	int bid = blockIdx.x + gridDim.x * blockIdx.y;
	int tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;
	int pixelX = threadIdx.x + blockDim.x * blockIdx.x;
	int pixelY = threadIdx.y + blockDim.y * blockIdx.y;

	if (pixelX >= width || pixelY >= height) return;

	__shared__ Mat4 c2w;
	c2w = *cameraToWorld;

	curandState* localState = globalState + pixelX + pixelY * width;
	Ray* localRay = &rays[pixelX + pixelY * width];

	Vec2 pixelLocation = DeviceSampler::RectUniform(localState);
	pixelLocation += Vec2(pixelX, pixelY); 
	pixelLocation *= Vec2(1.0f / width, 1.0f / height);

	Real tanHalfVFov = tan( vFov * 0.5f);
	Real sensorHeight = 2.0f * tanHalfVFov;
	Real sensorWidth = aspectRatio * sensorHeight;

	pixelLocation *= Vec2(sensorWidth, sensorHeight);
	Vec3 dir = Vec3(pixelLocation, 0.0f) - Vec3(0.5f * sensorWidth, 0.5f * sensorHeight, 1.0f);
	dir = (c2w * Vec4(dir, 0.0f)).xyz();
	localRay->m_origin = (c2w * Vec4(0.0f, 0.0f, 0.0f, 1.0f)).xyz();
	localRay->m_direction = normalize(dir);
}

CUDA_CALLABLE inline
void traverseBVH(Ray ray, BVHNode* nodes, Vec3* vertices, uint32_t* indices, int* stack, int size, HitStatus& status)
{
	int stackPtr = 1;
	stack[0] = 0;
	
	while (stackPtr > 0)
	{
		int currentIdx = stack[stackPtr - 1];
		stackPtr--;
		// printf("%d\n", currentIdx);

		if (currentIdx >= size - 1) // is leaf node
		{
			int fid = nodes[currentIdx].info.leaf.fid;
			auto vid0 = indices[3 * fid + 0];
			auto vid1 = indices[3 * fid + 1];
			auto vid2 = indices[3 * fid + 2];
			Real hitDist;
			Vec2 uv;
			// Shall we do the ray-triangle test here, or later after all collected ?
			if (rayHitTriangle(ray, vertices[vid0], vertices[vid1], vertices[vid2], hitDist, uv))
			{
				if (hitDist < status.hitDist && hitDist > 0.0f)
				{
					status.hitDist = hitDist;
					status.hitIdx = fid;
					status.uv = uv;
				}
			}
		}
		else // is internal node
		{
			int leftChild = nodes[currentIdx].info.intern.leftChild;
			int rightChild = nodes[currentIdx].info.intern.rightChild;
			if (rayHitBBox(ray, nodes[leftChild].box))
			{
				stack[stackPtr] = leftChild;
				stackPtr++;
			}
			if (rayHitBBox(ray, nodes[rightChild].box))
			{
				stack[stackPtr] = rightChild;
				stackPtr++;
			}
		}
		/*if (currentIdx == 969)
		{
			printf("\n");
		}*/
	}
}

// worry about warp divergence
CUDA_CALLABLE inline
int mtlBinarySearch(int fid, MtlInterval* mtlLUT, int objCnt)
{
	// binary search
	int left = 0;
	int right = objCnt - 1;
	while (right != left)
	{
		int mid = (left + right) >> 1;
		if (fid < mtlLUT[mid].begin) right = mid;
		else left = mid;
	}
	return mtlLUT[right].mtlIdx;
}

CUDA_CALLABLE inline
int mtlLinearSearch(int fid, MtlInterval* mtlLUT, int objCnt)
{
	int i = 0;
	for (; i < objCnt; ++i)
	{
		if (fid >= mtlLUT[i].begin) break;
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
int findIdxOfTrans(int fid, MtlInterval* interval, int objCnt)
{
	int i = objCnt - 1;
	for (; i >= 0; --i)
	{
		int begin = interval[i].begin;
		if (fid >= begin) return i;
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
		wNormals[vId] = normalize((transMat * Vec4(normals[vId], 0.0f)).xyz());
	}
}

__global__
void trace(BVHNode* nodes, Vec3* vertices, Vec3* normals, uint32_t* indices, MtlInterval* mtlLUT, Material* materials, 
	Spectrum* color, Ray* rays, curandState* globalState, int* stacks, int* hitRecord, int width, int height, int size, int objCnt)
{
	int tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	int bid = blockIdx.x + gridDim.x * blockIdx.y;
	int tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;
	int pixelX = threadIdx.x + blockDim.x * blockIdx.x;
	int pixelY = threadIdx.y + blockDim.y * blockIdx.y;

	__shared__ Real scratch[BLK_SIZE * DEPTH_TRACE];

	if (pixelY != 680) return;
	if (pixelX != 777) return;
	if (pixelX >= width || pixelY >= height) return;
	int offset = pixelX + pixelY * width;
	auto localState = globalState + offset;
	auto localHitRecord = hitRecord + offset;
	auto localStack = stacks + offset * 64;
	auto pStack = scratch + tidLocal * DEPTH_TRACE;
	int mtlIdxStack[DEPTH_TRACE];
	Spectrum directLightStack[DEPTH_TRACE];

	if constexpr ( RENDER_NORMAL )
	{
		Ray localRay = rays[pixelX + pixelY * width];
		HitStatus status;
		traverseBVH(localRay, nodes, vertices, indices, localStack, size, status);
		if (status.hitIdx == -1) printf("not hit\n");

		*localHitRecord = status.hitIdx;

		if (status.hitIdx < 0) return;

		uint32_t i0 = indices[status.hitIdx * 3 + 0];
		uint32_t i1 = indices[status.hitIdx * 3 + 1];
		uint32_t i2 = indices[status.hitIdx * 3 + 2];

		Real& u = status.uv.x;
		Real& v = status.uv.y;
		Real w = 1.0f - u - v;
		Vec3 normal = w * normals[i0] + u * normals[i1] + v * normals[i2];
		auto c = Spectrum(normal.habs() * 255.0f);
		color[pixelX + pixelY * width] += c;
	}
	else
	{
		int depth = 0;
		for ( ; depth < DEPTH_TRACE; ++depth )
		{
			Real dist = REAL_MAX;
			Ray localRay = rays[pixelX + pixelY * width];
			HitStatus status;
			traverseBVH(localRay, nodes, vertices, indices, localStack, size, status);

			*localHitRecord = status.hitIdx;

			if (status.hitIdx < 0) return;

			uint32_t i0 = indices[status.hitIdx * 3 + 0];
			uint32_t i1 = indices[status.hitIdx * 3 + 1];
			uint32_t i2 = indices[status.hitIdx * 3 + 2];

			Real& u = status.uv.x;
			Real& v = status.uv.y;
			Real w = 1.0f - u - v;
			Vec3 normal = w * normals[i0] + u * normals[i1] + v * normals[i2];
			Vec3 hitPos = w * vertices[i0] + u * vertices[i1] + v * vertices[i2];

			int mtlIndex = mtlBinarySearch(status.hitIdx, mtlLUT, objCnt);
			Vec3 newDir;
			Real prob = getNewDirection(localRay, normal, materials + mtlIndex, localState, newDir);

			localRay.m_origin = hitPos + MathConst::Delta * newDir;

			// sample direct light
			{
				// TODO: Sample delta light
				Vec3 directDir;
				Real directProb = getNewDirection(localRay, normal, materials + mtlIndex, localState, directDir);
				localRay.m_direction = directDir;
				traverseBVH(localRay, nodes, vertices, indices, localStack, size, status);
				if (status.hitIdx >= 0)
				{
					int directMtlIdx = mtlBinarySearch(status.hitIdx, mtlLUT, objCnt);
					directLightStack[depth] = materials[directMtlIdx].baseColor * materials[directMtlIdx].emissionFactor / directProb;
				}
				else
				{
					directLightStack[depth] = Spectrum(0.0f, 0.0f, 0.0f);
				}
			}

			pStack[depth] = prob;
			mtlIdxStack[depth] = mtlIndex;
			localRay.m_direction = newDir;
		}

		// sample indirect light
		depth--;
		Spectrum radiance;
		while (depth >= 0)
		{
			Material* mtl = &materials[mtlIdxStack[depth]];
			Spectrum emission;
			if (mtl->emissionFactor > 0.0f) emission = mtl->baseColor * mtl->emissionFactor;

			radiance = (emission + (directLightStack[depth] + radiance) * mtl->baseColor) / pStack[depth];
			depth--;
		}

		color[pixelX + pixelY * width] += radiance;
	}
}

__global__
void copyToFB(Spectrum* radiance, unsigned char* data, int height, int width, int nSamplesPerPixel)
{
	int pixelX = threadIdx.x + blockDim.x * blockIdx.x;
	int pixelY = threadIdx.y + blockDim.y * blockIdx.y;
	if (pixelX >= width || pixelY >= height) return;

	int offset = pixelY * width + pixelX;
	Spectrum totalRad = radiance[offset];
	totalRad = totalRad / nSamplesPerPixel;
	uchar3 rgb = totalRad.toUChar();

	int fbOffset = (height - pixelY - 1) * width + pixelX;
	unsigned char* pixel = &data[fbOffset * 4];
	pixel[0] = rgb.z;
	pixel[1] = rgb.y;
	pixel[2] = rgb.x;
}

/*
 * Kernel Functions End
 */

PathTracer::PathTracer() :
m_displayer(),
m_width(m_displayer.m_windowExtent.width),
m_height(m_displayer.m_windowExtent.height),
m_randStates(m_width * m_height),
m_radiance(m_width * m_height),
m_wVertices(0),
m_wNormals(0),
m_rays(m_height * m_width) {}

void PathTracer::doTrace(DeviceScene& d_scene, Camera& camera, unsigned char* framebuffer, int nSamplesPerPixel)
{
	time_t tseed;
	time(&tseed);

	size_t nNormals = d_scene.normals.size();
	size_t nVerts = d_scene.vertices.size();
	size_t nFaces = d_scene.indices.size() / 3;
	int nObjs = d_scene.materialsLUT.size();

	Real vFov = camera.getVFov();
	Real aspRatio = camera.getAspRatio();
	Real nearPlane = camera.getNearPlane();
	std::vector<Mat4> c2w(1);
	c2w[0] = camera.m_transform->localToWorld();
	thrust::device_vector<Mat4> d_c2w(1);
	thrust::copy(c2w.begin(), c2w.end(), d_c2w.begin());
	Mat4* dp_c2w = thrust::raw_pointer_cast(d_c2w.data());

	curandState* dp_states = thrust::raw_pointer_cast(m_randStates.data());
	int nPixels = m_height * m_width;
	setupRandSeed KERNEL_DIM((nPixels + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE) (tseed, dp_states, nPixels);

	if (m_wVertices.size() != nVerts) m_wVertices.resize(nVerts);
	if (m_wNormals.size() != nVerts) m_wNormals.resize(nNormals);
	Ray* dp_rays = thrust::raw_pointer_cast(m_rays.data());

	Spectrum* dp_radiance = thrust::raw_pointer_cast(m_radiance.data());
	uint32_t* dp_indices = thrust::raw_pointer_cast(d_scene.indices.data());
	Vec3* dp_vertices = thrust::raw_pointer_cast(d_scene.vertices.data());
	Vec3* dp_normals = thrust::raw_pointer_cast(d_scene.normals.data());
	Vec2* dp_texCoords = thrust::raw_pointer_cast(d_scene.texCoords.data());
	Material* dp_mtls = thrust::raw_pointer_cast(d_scene.materials.data());
	MtlInterval* dp_mtlInterval = thrust::raw_pointer_cast(d_scene.materialsLUT.data());
	Mat4* dp_vertTrans = thrust::raw_pointer_cast(d_scene.vertTrans.data());
	Mat4* dp_normalTrans = thrust::raw_pointer_cast(d_scene.normalTrans.data());
	Vec3* dp_wVertices = thrust::raw_pointer_cast(m_wVertices.data());
	Vec3* dp_wNormals = thrust::raw_pointer_cast(m_wNormals.data());

	dim3 blk_config(BLK_DIM, BLK_DIM);
	dim3 grid_config((m_width + BLK_DIM - 1) / BLK_DIM, (m_height + BLK_DIM - 1) / BLK_DIM);

	/*checkDeviceVector(d_scene.vertices);
	checkDeviceVector(d_scene.vertTrans);
	checkDeviceVector(d_scene.indices);
	checkDeviceVector(d_scene.normalTrans);
	checkDeviceVector(d_scene.materialsLUT);*/

	thrust::fill(m_radiance.begin(), m_radiance.end(), Spectrum());
	thrust::device_vector<int> d_stacks(nPixels * 64);
	int* dp_stacks = thrust::raw_pointer_cast(d_stacks.data());

	transform KERNEL_DIM(((nFaces + BLK_SIZE - 1) / BLK_SIZE), BLK_SIZE) (dp_vertices, dp_normals, dp_indices, 
		dp_mtlInterval, dp_vertTrans, dp_normalTrans, dp_wVertices, dp_wNormals, nFaces, nObjs);
	/*checkDeviceVector(wVertices);
	checkDeviceVector(wNormals);*/
	//checkVertices(d_scene.indices, m_wVertices);
	CUDA_CHECK(cudaDeviceSynchronize());

	// constructs bvh every frame
	BVH bvh(nFaces);
	bvh.construct(m_wVertices, d_scene.indices);
	BVHNode* dp_bvhNodes = thrust::raw_pointer_cast(bvh.m_nodes.data());
	//checkBVHNodes(bvh.m_nodes);

	thrust::device_vector<int> d_hitRecord(m_width * m_height);
	int* dp_hitRecord = thrust::raw_pointer_cast(d_hitRecord.data());

	for ( int idx = 0; idx < nSamplesPerPixel; ++idx )
	{
		sampleRay KERNEL_DIM(grid_config, blk_config) (dp_rays, m_width, m_height, dp_states, vFov, aspRatio, nearPlane, dp_c2w);
		checkDeviceVector(m_rays);
		CUDA_CHECK(cudaDeviceSynchronize());

		trace KERNEL_DIM(grid_config, blk_config) (dp_bvhNodes, dp_wVertices, dp_wNormals, dp_indices, dp_mtlInterval,
			dp_mtls, dp_radiance, dp_rays, dp_states, dp_stacks, dp_hitRecord, m_width, m_height, nFaces, nObjs);
		CUDA_CHECK(cudaDeviceSynchronize());
		checkDeviceVector(d_hitRecord);
	}
	
	copyToFB KERNEL_DIM(grid_config, blk_config) (dp_radiance, framebuffer, m_height, m_width, nSamplesPerPixel);
}

void PathTracer::render(const std::string& meshFile)
{
	Scene scene(meshFile, "gltf");
	int nSamplesPerPixel = 1;

	DeviceScene d_scene = scene.copySceneToDevice();
	Camera& camera = scene.m_camera;
	std::function<void(unsigned char* framebuffer)> pathTrace;
	pathTrace = [this, &d_scene, &camera, nSamplesPerPixel](unsigned char* framebuffer)
	{
		doTrace(d_scene, camera, framebuffer, nSamplesPerPixel);
	};

	try
	{
		m_displayer.init();
		m_displayer.run(pathTrace);
		m_displayer.cleanup();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}
}
