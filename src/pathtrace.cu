#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;


// TODO: static variables for device memory, any extra info you need, etc
// ...
static int * dev_materialIDs_1 = NULL;
static int * dev_materialIDs_2 = NULL;


// first bounce intersections cache
static ShadeableIntersection * dev_first_bounce_intersections = NULL;

// texture buffer
glm::vec3 ** dev_textures = NULL;
glm::vec2 * dev_textureSizes = NULL;

glm::vec3 ** dev_normal_maps = NULL;
glm::vec2 * dev_normal_mapSizes = NULL;

void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_materialIDs_1, pixelcount * sizeof(int));
	cudaMalloc(&dev_materialIDs_2, pixelcount * sizeof(int));

	cudaMalloc(&dev_first_bounce_intersections, pixelcount * sizeof(ShadeableIntersection));

	// copy texutre memory to device
	int textureSize = hst_scene->textures.size();
	cudaMalloc((void**)&dev_textures, textureSize * sizeof(glm::vec3*));
	cudaMalloc((void**)&dev_textureSizes, textureSize * sizeof(glm::vec2));
	std::vector<glm::vec3*> textures;
	std::vector<glm::vec2> textureSizes;
	glm::vec3* tmp;
	for (int i = 0; i < textureSize; ++i)
	{
		int texPixelCount = hst_scene->textures[i]->pixelCount();
		cudaMalloc((void**)&tmp, texPixelCount * sizeof(glm::vec3));
		cudaMemcpy(tmp, hst_scene->textures[i]->pixels, texPixelCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
		textures.push_back(tmp);
		textureSizes.push_back(hst_scene->textures[i]->getSize());
	}
	cudaMemcpy(dev_textures, textures.data(), textureSize * sizeof(glm::vec3*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_textureSizes, textureSizes.data(), textureSize * sizeof(glm::vec2), cudaMemcpyHostToDevice);
	
	// copy normal maps into device memory
	int normalMapSize = hst_scene->normalMaps.size();
	cudaMalloc((void**)&dev_normal_maps, normalMapSize * sizeof(glm::vec3*));
	cudaMalloc((void**)&dev_normal_mapSizes, normalMapSize * sizeof(glm::vec2));
	std::vector<glm::vec3*> normal_maps;
	std::vector<glm::vec2> normal_map_sizes;
	for (int i = 0; i < normalMapSize; ++i)
	{
		int texPixelCount = hst_scene->normalMaps[i]->pixelCount();
		cudaMalloc((void**)&tmp, texPixelCount * sizeof(glm::vec3));
		cudaMemcpy(tmp, hst_scene->normalMaps[i]->pixels, texPixelCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
		normal_maps.push_back(tmp);
		normal_map_sizes.push_back(hst_scene->normalMaps[i]->getSize());
	}
	cudaMemcpy(dev_normal_maps, normal_maps.data(), normalMapSize * sizeof(glm::vec3*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_normal_mapSizes, normal_map_sizes.data(), normalMapSize * sizeof(glm::vec2), cudaMemcpyHostToDevice);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);

	// TODO: clean up any extra device memory you created
	cudaFree(dev_materialIDs_1);
	cudaFree(dev_materialIDs_2);
	cudaFree(dev_first_bounce_intersections);

	//free texture memory on device
	cudaFree(dev_textures);
	cudaFree(dev_textureSizes);

	// free normal map memory
	cudaFree(dev_normal_maps);
	cudaFree(dev_normal_mapSizes);

	// check error
	checkCUDAError("pathtraceFree");
}

/**
* helper function for DOF
* improvement for ConcentrixSampleDisc
* reference : http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html
*/
__device__ glm::vec2 ConcentricSampleDisc(float u1, float u2)
{
	float phi, r;
	float a = 2.0f * u1 - 1.0f;
	float b = 2.0f * u2 - 1.0f;

	if (a*a > b*b)
	{
		r = a;
		phi = (PI / 4.0f) *(b / a);
	}
	else
	{
		r = b;
		phi = (PI / 2.0f) - (PI / 4.0f)*(a / b);
	}

	return glm::vec2(r*glm::cos(phi), r*glm::sin(phi));
}


/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(
	Camera cam, int iter, int traceDepth, PathSegment* pathSegments, 
	bool stochasticAA)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
		//segment.color = glm::vec3(0, 0, 0);

		// TODO: implement antialiasing by jittering the ray
		if (!stochasticAA)
		{
			segment.ray.direction = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
				- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
				);
		}
		else
		{
			thrust::uniform_real_distribution<float> u01(0, 1);
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
			float dx = u01(rng);
			float dy = u01(rng);

			float fx = (float)x + dx;
			float fy = (float)y + dy;

			segment.ray.direction = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)fx - (float)cam.resolution.x * 0.5f)
				- cam.up * cam.pixelLength.y * ((float)fy - (float)cam.resolution.y * 0.5f)
				);
		}

		if (cam.DOF.x > 0.0f) // Depth of field
		{
			thrust::uniform_real_distribution<float> u01(0, 1);
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
			glm::vec2 lenUV = ConcentricSampleDisc(u01(rng), u01(rng));
			lenUV *= cam.DOF.x;
			float ft = glm::abs(cam.DOF.y / cam.view.z);
			glm::vec3 pfocus = segment.ray.direction * ft + segment.ray.origin;

			segment.ray.origin += lenUV.x * cam.right + lenUV.y * cam.up;
			segment.ray.direction = glm::normalize(pfocus - segment.ray.origin);
		}


		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO: 
// pathTraceOneBounce handles ray intersections, generate intersections for shading, 
// and scatter new ray. You might want to call scatterRay from interactions.h
__global__ void pathTraceOneBounce(
	int iter,
	int depth,
	int num_paths,
	PathSegment * pathSegments,
	Geom * geoms,
	int geoms_size,
	Material * materials,
	int material_size,
	ShadeableIntersection * intersections,
	Geom csg_box, Geom csg_sphere
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;
		glm::vec2 uv = glm::vec2(-1, -1);

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv = glm::vec2(-1, -1);

		// naive parse through global geoms
		thrust::uniform_real_distribution<float> u01(0, 1);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_index, depth);
		int csgMaterialID;

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];
			float u = geom.hasMotionBlur ? u01(rng) : -1;

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv, u);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv, u);
			}
			else if (geom.type == CSG)
			{
				t = csgIntersectionTest(geom, pathSegment.ray, csg_box, csg_sphere, tmp_intersect, tmp_normal, tmp_uv, csgMaterialID);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				uv = tmp_uv;
			}
		}


		// TODO: scatter the ray, generate intersections for shading
		// feel free to modify the code below

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			//intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].materialId = (geoms[hit_geom_index].type == CSG ? csgMaterialID : geoms[hit_geom_index].materialid);
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].hit_geom_index = hit_geom_index;
			intersections[path_index].uv = uv;
		}
	}
}

/**  
 * Shading function 
 */
__global__ void shadingAndEvaluatingBSDF(
	int iter,
	int depth,
	int num_paths,
	ShadeableIntersection * shadeableIntersections,
	PathSegment * pathSegments,
	Geom * geoms,
	int geoms_size,
	Material * materials,
	glm::vec3** textures,
	glm::vec2*  textureSizes,
	glm::vec3** normal_maps,
	glm::vec2* normal_mapSizes
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths)
		return;
	
	ShadeableIntersection &isx = shadeableIntersections[idx];
	PathSegment &pathSeg = pathSegments[idx];
	if (isx.t > 0.0f)
	{
		Material &material = materials[isx.materialId];
		
		glm::vec3 color;
		if (material.emittance > 0) // light source
		{
			pathSeg.color *= material.color * material.emittance;
			pathSeg.remainingBounces = 0;
		}
		else // bounce ray
		{
			glm::vec3 intersectPoint = pathSeg.ray.origin + pathSeg.ray.direction * isx.t;
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			scatterRay(pathSeg, intersectPoint, isx.surfaceNormal, material, textures, textureSizes, isx.uv, rng, normal_maps, normal_mapSizes, geoms[isx.hit_geom_index]);
			
			pathSeg.remainingBounces--;
		}		
	}
	else // hit nothing
	{
		pathSeg.color = glm::vec3(0);
		pathSeg.remainingBounces = 0;
	}
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

// update terminated segments to image
__global__ void kernUpdateTerminatedSegmentsToImage(
	int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths && iterationPaths[index].remainingBounces <= 0)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}
/**
 * predictor for thrust::remove_if
 */
struct terminate_path
{
	__host__ __device__
	bool operator()(const PathSegment & pathSeg)
	{
		return (pathSeg.remainingBounces <= 0);
	}
};

__global__ void kernGetMaterialIDs(
	int nPaths, int * dev_materialIDs1, int * dev_materialIDs2, ShadeableIntersection * intersections)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= nPaths)
		return;

	dev_materialIDs2[index] = dev_materialIDs1[index] = intersections[index].materialId;
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;
	const bool reshuffleByMaterialIDs = hst_scene->state.reshuffleByMaterialIDs;
	const bool useFirstBounceIntersectionCache = hst_scene->state.useFirstBounceIntersectionCache;
	const bool stochasticAntialising = hst_scene->state.stochasticAntiliasing;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths, stochasticAntialising);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	int remainingNumPaths = num_paths;
	dim3 numblocksPathSegmentTracing;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		numblocksPathSegmentTracing = (remainingNumPaths + blockSize1d - 1) / blockSize1d;

		// first bounce caching related, so ugly
		if (!useFirstBounceIntersectionCache ||
			(useFirstBounceIntersectionCache && ((depth == 0 && iter == 1) || (depth > 0)))
			)
		{
			// tracing
			pathTraceOneBounce << <numblocksPathSegmentTracing, blockSize1d >> > (
				iter,
				depth,
				remainingNumPaths,
				dev_paths,
				dev_geoms,
				hst_scene->geoms.size(),
				dev_materials,
				hst_scene->materials.size(),
				dev_intersections,
				csg_box,csg_sphere
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}

		if (useFirstBounceIntersectionCache && (depth == 0 && iter == 1))
		{
			cudaMemcpy(dev_first_bounce_intersections, dev_intersections, remainingNumPaths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		
		if (useFirstBounceIntersectionCache && (depth == 0 && iter > 1))
		{
			cudaMemcpy(dev_intersections, dev_first_bounce_intersections, remainingNumPaths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}

		depth++;


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
		
		if (reshuffleByMaterialIDs)
		{
			// after tracing one bounce , get materialIDs for reshuffling intersections and pathSegmetns
			kernGetMaterialIDs << <numblocksPathSegmentTracing, blockSize1d >> >(
				remainingNumPaths,
				dev_materialIDs_1,
				dev_materialIDs_2,
				dev_intersections
				);

			thrust::sort_by_key(thrust::device, dev_materialIDs_1, dev_materialIDs_1 + remainingNumPaths, dev_paths);
			thrust::sort_by_key(thrust::device, dev_materialIDs_2, dev_materialIDs_2 + remainingNumPaths, dev_intersections);
		}

		// shading and generate new directions using BSDF evaluation
		shadingAndEvaluatingBSDF << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			depth,
			remainingNumPaths,
			dev_intersections,
			dev_paths,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_materials,
			dev_textures,
			dev_textureSizes,
			dev_normal_maps,
			dev_normal_mapSizes
			);
		checkCUDAError("shading");
		cudaDeviceSynchronize();

		// update terminated segments to final image
		kernUpdateTerminatedSegmentsToImage << <numblocksPathSegmentTracing, blockSize1d >> >(
			remainingNumPaths,
			dev_image,
			dev_paths
			);
		checkCUDAError("udpate terminated segments");
		cudaDeviceSynchronize();

		// stream compaction, delete that paths that remainingBounces <= 0
		//std::cout << "before compaction = " << remainingNumPaths;
		PathSegment *newPathEnd = thrust::remove_if(thrust::device, dev_paths, dev_paths + remainingNumPaths, terminate_path());
		if (newPathEnd != NULL)
		{
			remainingNumPaths = newPathEnd - dev_paths;
		}
		else
		{
			remainingNumPaths = 0;
		}
		checkCUDAError("thrust::remove_if");
		//std::cout << ", after compaction = " << remainingNumPaths << std::endl;

		iterationComplete = (depth >= traceDepth || remainingNumPaths <= 0); // TODO: should be based off stream compaction results.
	}

	// Assemble this iteration and apply it to the image
	//dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	//finalGather << <numBlocksPixels, blockSize1d >> >(num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> >(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
