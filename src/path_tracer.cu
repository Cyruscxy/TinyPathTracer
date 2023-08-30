#include "path_tracer.h"
#include "intellisense_cuda.h"
#include "geometry_queries.h"
#include <cuda.h>
#include <thrust/device_malloc.h>
#include <cstdint>
#include <time.h>


namespace 
{
	// For debug
	constexpr bool RENDER_NORMAL = false;

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
	size_t tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	size_t bid = blockIdx.x + gridDim.x * blockIdx.y;
	size_t tidGlobal = tidLocal + bid * BLK_SIZE;
	if ( tidGlobal < n ) curand_init(seed, tidGlobal, 0, &state[tidGlobal]);
}

__device__ __inline__
Ray sampleRays(curandState* randState, Real vFov, Real aspectRatio, Mat4& cameraToWorld,
	int pixelX, int pixelY, int width, int height)
{
	Vec2 pixelLocation = DeviceSampler::RectUniform(randState);
	pixelLocation += Vec2(pixelX, pixelY);
	pixelLocation *= Vec2(1.0f / width, 1.0f / height);

	Real tanHalfVFov = tan(vFov * 0.5f);
	Real sensorHeight = 2.0f * tanHalfVFov;
	Real sensorWidth = aspectRatio * sensorHeight;

	pixelLocation *= Vec2(sensorWidth, sensorHeight);
	Vec3 dir = Vec3(pixelLocation, 0.0f) - Vec3(0.5f * sensorWidth, 0.5f * sensorHeight, 1.0f);
	dir = normalize((cameraToWorld * Vec4(dir, 0.0f)).xyz());
	Vec3 origin = (cameraToWorld * Vec4(0.0f, 0.0f, 0.0f, 1.0f)).xyz();
	return Ray(origin, dir);
}

CUDA_CALLABLE inline
void traverseBVH(Ray ray, BVHNode* nodes, Vec3* vertices, uint32_t* indices, int nFaces, HitStatus& status)
{
	int call_stack[64];
	int stackPtr = 1;
	call_stack[0] = 0;

	while (stackPtr > 0)
	{
		int currentIdx = call_stack[stackPtr - 1];
		stackPtr--;

		if (currentIdx >= nFaces - 1) // is leaf node
		{
			int fid = nodes[currentIdx].info.leaf.fid;
			auto vid0 = indices[3 * fid + 0];
			auto vid1 = indices[3 * fid + 1];
			auto vid2 = indices[3 * fid + 2];
			Real hitDist;
			Vec2 uv;
			if (rayHitTriangle(ray, vertices[vid0], vertices[vid1], vertices[vid2], hitDist, uv))
			{
				if (hitDist < status.hitDist && hitDist > MathConst::Delta)
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
				call_stack[stackPtr] = leftChild;
				stackPtr++;
			}
			if (rayHitBBox(ray, nodes[rightChild].box))
			{
				call_stack[stackPtr] = rightChild;
				stackPtr++;
			}
		}
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
	int i = objCnt - 1;
	for (; i >= 0; --i)
	{
		int begin = mtlLUT[i].begin;
		if (fid >= begin) break;
	}
	return mtlLUT[i].mtlIdx;
}

CUDA_CALLABLE inline
Vec3 reflect(Vec3 dir, Vec3 normal)
{
	return dir - 2.0f * dot(dir, normal) * normal;
}

CUDA_CALLABLE inline
Vec3 refract(Vec3 dir, Vec3 normal, Real ior, Real& cosThetaI, Real& eta, bool& tir)
{
	cosThetaI = dot(dir, normal);
	// eta_i / eta_t
	eta = cosThetaI > 0.0f ? ior : 1.0f / ior;
	normal = cosThetaI > 0.0f ? -normal : normal;
	cosThetaI = abs(cosThetaI);
	
	Real sin2ThetaI = 1.0f - cosThetaI * cosThetaI;
	Real sin2ThetaT = eta * eta * sin2ThetaI;

	// Check if TIR? 
	if (sin2ThetaT >= 1.0f) {
		tir = true;
		return {};
	}

	Real cosThetaT = sqrt(1.0f - sin2ThetaT);
	return eta * dir + (cosThetaI * eta - cosThetaT) * normal;
}

CUDA_CALLABLE inline
Real shlickFresnel(Real cosThetaI, Real eta)
{
	Real f0 = (1.0f - eta) / (1.0f + eta);
	f0 *= f0;
	Real m = clamp(1.0f - cosThetaI, 1.0f, 0.0f);
	Real m2 = m * m;
	return f0 + (1.0f - f0) * m2 * m2 * m;
}

CUDA_CALLABLE inline
Real Fresnel(Real cosThetaI, Real eta)
{
	Real sin2ThetaT = eta * eta * (1.0f - cosThetaI * cosThetaI);
	if (sin2ThetaT >= 1.0f) return 1.0f;
	Real cosThetaT = sqrt(1.0f - sin2ThetaT);

	Real r1 = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
	Real r2 = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
	return 0.5f * (r1 * r1 + r2 * r2);
}

__device__ __inline__
Real getNewDirection(Ray& ray, Vec3 normal, Material* mtl, curandState* state, Vec3& nextDir, Real& attenFactor)
{
	// get new dir
	Real probability;
	if (mtl->eta > 0.0f) {
		bool tir = false;
		Real eta, cosThetaI;
		Vec3 refractDir = refract(ray.m_direction, normal, mtl->eta, cosThetaI, eta, tir);
		Vec3 reflectDir = reflect(ray.m_direction, normal);
		Real fr = tir ? 1.0f : shlickFresnel(cosThetaI, eta);

		if ( DeviceSampler::CoinFlip(state, fr) )
		{
			nextDir = reflectDir;
		}
		else
		{
			nextDir = refractDir;
		}
		probability = 1.0f;
		attenFactor = 1.0f;
	}
	else if ( mtl->metallic > 0.0f )
	{
		attenFactor = 1.0f;
		nextDir = reflect(ray.m_direction, normal);
		probability = 1.0f;
	}
	else
	{
		Real sign = dot(ray.m_direction, normal) > 0.0f ? -1.0f : 1.0f;
		normal *= sign;
		nextDir = DeviceSampler::HemisphereCosine(state, normal);
		attenFactor = abs(dot(nextDir, normal)) / MathConst::PI;
		probability = DeviceSampler::HemishpereCosinePDF(nextDir, normal);
	}
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

__device__ __inline__
Spectrum sampleDeltaLights(DeltaLight* lights, int nLights, const Vec3& pos, Material* mtl, BVHNode* nodes, 
	Vec3* vertices, uint32_t* indices, int nFaces, HitStatus& status)
{
	Spectrum radiance;
	int currentIdx = status.hitIdx;
	for ( int i = 0; i < nLights; ++i )
	{
		status = {};
		Incoming incoming = lights[i].sample(pos);
		//Incoming incoming;

		Ray shadowRay(pos, incoming.direction);
		traverseBVH(shadowRay, nodes, vertices, indices, nFaces, status);
		if ( status.hitIdx == -1 )
		{
			radiance += mtl->baseColor * incoming.radiance;
		}
	}

	return radiance;
}

__global__
void trace(BVHNode* nodes, Vec3* vertices, Vec3* normals, uint32_t* indices, DeltaLight* lights, MtlInterval* mtlLUT, 
	Material* materials, Spectrum* color, curandState* globalState, int width, int height, int nFaces, int objCnt, 
	int nLIghts, Real vFov, Real aspectRatio, Mat4* cameraToWorld, int nSamplesPerPixel)
{
	int tidLocal = threadIdx.x + threadIdx.y * blockDim.x;
	int bid = blockIdx.x + gridDim.x * blockIdx.y;
	int tidGlobal = tidLocal + bid * blockDim.x * blockDim.y;
	int pixelX = threadIdx.x + blockDim.x * blockIdx.x;
	int pixelY = threadIdx.y + blockDim.y * blockIdx.y;


	__shared__ Ray rayBuffer[BLK_SIZE];
	__shared__ Mat4 c2w[1];
	if (tidLocal == 0) c2w[0] = *cameraToWorld;
	__syncthreads();
	
	if (pixelX >= width || pixelY >= height) return;
	int offset = pixelX + pixelY * width;
	Real pStack[DEPTH_TRACE];
	int mtlIdxStack[DEPTH_TRACE];
	Spectrum directLightStack[DEPTH_TRACE];
	Spectrum attenuation[DEPTH_TRACE];

	auto localState = globalState + offset;

	if constexpr ( RENDER_NORMAL )
	{
		rayBuffer[tidLocal] = sampleRays(localState, vFov, aspectRatio, c2w[0], pixelX, pixelY, width, height);
		auto& ray = rayBuffer[tidLocal];

		HitStatus status;
		traverseBVH(ray, nodes, vertices, indices, nFaces, status);

		if (status.hitIdx < 0) return;

		uint32_t i0 = indices[status.hitIdx * 3 + 0];
		uint32_t i1 = indices[status.hitIdx * 3 + 1];
		uint32_t i2 = indices[status.hitIdx * 3 + 2];

		Real& u = status.uv.x;
		Real& v = status.uv.y;
		Real w = 1.0f - u - v;
		Vec3 normal = w * normals[i0] + u * normals[i1] + v * normals[i2];
		auto c = Spectrum(normal.habs());
		color[pixelX + pixelY * width] += c;
	}
	else
	{
		Spectrum totalRad;
		for ( ; nSamplesPerPixel > 0; nSamplesPerPixel -= 1 )
		{
			rayBuffer[tidLocal] = sampleRays(localState, vFov, aspectRatio, c2w[0], pixelX, pixelY, width, height);
			auto& ray = rayBuffer[tidLocal];
			
			int depth = 0;
			for (; depth < DEPTH_TRACE; ++depth)
			{
				HitStatus status;
				traverseBVH(ray, nodes, vertices, indices, nFaces, status);
				
				if (status.hitIdx < 0) break; 

				uint32_t i0 = indices[status.hitIdx * 3 + 0];
				uint32_t i1 = indices[status.hitIdx * 3 + 1];
				uint32_t i2 = indices[status.hitIdx * 3 + 2];

				Real& u = status.uv.x;
				Real& v = status.uv.y;
				Real w = 1.0f - u - v;
				Vec3 normal = normalize(w * normals[i0] + u * normals[i1] + v * normals[i2]);
				ray.m_origin = w * vertices[i0] + u * vertices[i1] + v * vertices[i2];
				//ray.m_origin = ray.m_origin + status.hitDist * ray.m_direction;

				int mtlIndex = mtlLinearSearch(status.hitIdx, mtlLUT, objCnt);

				Vec3 newDir;
				Real attenFactor;
				Real prob = getNewDirection(ray, normal, materials + mtlIndex, localState, newDir, attenFactor);
				attenuation[depth] = materials[mtlIndex].baseColor * attenFactor;
				pStack[depth] = prob;

				// sample direct light
				Vec3 directDir;
				Spectrum directRadiance = sampleDeltaLights(lights, nLIghts, ray.m_origin, materials + mtlIndex, 
					nodes, vertices, indices, nFaces, status);
				status = {};
				if (!(materials[mtlIndex].eta >= 1.0f || materials[mtlIndex].metallic > 0.0f)) {
					Real directProb = getNewDirection(ray, normal, materials + mtlIndex, localState, directDir, attenFactor);
					ray.m_direction = directDir;
					traverseBVH(ray, nodes, vertices, indices, nFaces, status);
					if (status.hitIdx >= 0)
					{
						int directMtlIdx = mtlLinearSearch(status.hitIdx, mtlLUT, objCnt);
						directLightStack[depth] = Spectrum(1.0f) * 
							(Spectrum(materials[directMtlIdx].emissionFactor)) + directRadiance;
					}
					else
					{
						directLightStack[depth] = directRadiance;
					}
				}
				else
				{
					directLightStack[depth] = directRadiance;
				}

				mtlIdxStack[depth] = mtlIndex;
				if (materials[mtlIndex].emissionFactor > 0.0f)
				{
					depth++;
					break;
				}
				ray.m_direction = newDir;
			}

			// sample indirect light
			depth -= 1;
			Spectrum radiance;
			while (depth >= 0)
			{
				auto* mtl = materials + mtlIdxStack[depth];
				if ( mtl->emissionFactor > 0.0f )
				{
					radiance = Spectrum(1.0f) * mtl->emissionFactor;
				}
				else
				{
					radiance = ((directLightStack[depth] + radiance) * attenuation[depth]) / pStack[depth];
				}
				depth -= 1;
			}
			totalRad += radiance;
		}
		color[offset] += totalRad;
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
	if constexpr ( !RENDER_NORMAL )
	{
		totalRad = totalRad / nSamplesPerPixel;
	}
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
m_wNormals(0) {}

void PathTracer::doTrace(DeviceScene& d_scene, Camera& camera, unsigned char* framebuffer, int nSamplesPerPixel)
{
	time_t tseed;
	time(&tseed);

	size_t nNormals = d_scene.normals.size();
	size_t nVerts = d_scene.vertices.size();
	size_t nFaces = d_scene.indices.size() / 3;
	int nObjs = d_scene.materialsLUT.size();
	int nLights = d_scene.lights.size();

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

	Spectrum* dp_radiance = thrust::raw_pointer_cast(m_radiance.data());
	uint32_t* dp_indices = thrust::raw_pointer_cast(d_scene.indices.data());
	Vec3* dp_vertices = thrust::raw_pointer_cast(d_scene.vertices.data());
	Vec3* dp_normals = thrust::raw_pointer_cast(d_scene.normals.data());
	Vec2* dp_texCoords = thrust::raw_pointer_cast(d_scene.texCoords.data());
	Material* dp_mtls = thrust::raw_pointer_cast(d_scene.materials.data());
	MtlInterval* dp_mtlInterval = thrust::raw_pointer_cast(d_scene.materialsLUT.data());
	DeltaLight* dp_lights = thrust::raw_pointer_cast(d_scene.lights.data());
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

	transform KERNEL_DIM(((nFaces + BLK_SIZE - 1) / BLK_SIZE), BLK_SIZE) (dp_vertices, dp_normals, dp_indices, 
		dp_mtlInterval, dp_vertTrans, dp_normalTrans, dp_wVertices, dp_wNormals, nFaces, nObjs);
	/*checkDeviceVector(m_wVertices);
	checkDeviceVector(m_wNormals);*/
	//checkVertices(d_scene.indices, m_wVertices);
	CUDA_CHECK(cudaDeviceSynchronize());

	// constructs bvh every frame
	BVH bvh(nFaces);
	bvh.construct(m_wVertices, d_scene.indices);
	BVHNode* dp_bvhNodes = thrust::raw_pointer_cast(bvh.m_nodes.data());
	// checkBVHNodes(bvh.m_nodes);

	//thrust::device_vector<int> d_hitRecord(m_width * m_height, -1);
	//int* dp_hitRecord = thrust::raw_pointer_cast(d_hitRecord.data());

	
	trace KERNEL_DIM(grid_config, blk_config) (dp_bvhNodes, dp_wVertices, dp_wNormals, dp_indices, dp_lights,
		dp_mtlInterval, dp_mtls, dp_radiance, dp_states, m_width, m_height, nFaces, nObjs, nLights, vFov, aspRatio, 
		dp_c2w, nSamplesPerPixel);
	CUDA_CHECK(cudaDeviceSynchronize());
	/*checkDeviceVector(d_hitRecord);
	checkHitStatus(d_hitRecord, m_width, m_height);*/
	
	copyToFB KERNEL_DIM(grid_config, blk_config) (dp_radiance, framebuffer, m_height, m_width, nSamplesPerPixel);
}

void PathTracer::render(const std::string& meshFile)
{
	Scene scene(meshFile, "gltf");
	int nSamplesPerPixel = 32;

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
