#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
	CSG
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    enum GeomType type;
    int materialid;

    glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	glm::mat4 transform;
	glm::mat4 inverseTransform;
	glm::mat4 invTranspose;

	// for motion blur
	bool hasMotionBlur;
	glm::vec3 translation_dst;
	glm::vec3 rotation_dst;
	glm::vec3 scale_dst;
	//glm::mat4 transform_dst;
	//glm::mat4 inverseTransform_dst;
	//glm::mat4 invTranspose_dst;

};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

	int texId;
	int normapId;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
	glm::vec2 DOF; // dof.x = lenRadiux, dof.y = focalLength
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;

	bool reshuffleByMaterialIDs;
	bool useFirstBounceIntersectionCache;

	bool stochasticAntialiasing;

	bool useStreamCompaction;
};

struct PathSegment {
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
	float t;
	glm::vec3 surfaceNormal;
	int materialId;
	int hit_geom_index;
	glm::vec2 uv;
};

// constructive solid geometry
enum CSGOPS
{
	DIFF,
	DIFF_INV,
	UNION,
	INTERSECT
};

struct CSGNode
{
	CSGNode* leftG;
	CSGNode* rightG;

	Geom* geo;
	CSGOPS op;

};