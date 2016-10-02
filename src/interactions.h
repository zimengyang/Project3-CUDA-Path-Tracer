#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
glm::vec3 normal, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = sqrt(u01(rng)); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = u01(rng) * TWO_PI;

	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
PathSegment & pathSegment,
glm::vec3 intersect,
glm::vec3 normal,
const Material &m,
thrust::default_random_engine &rng) {
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.

	// consider the material as - diffuse/reflection/refraction,
	// three possibilities add up to 1
	// diffuse possibility = 1 - material.hasReflection - material.hasRefraction

	thrust::uniform_real_distribution<float> u01(0, 1);
	float prob = u01(rng);

	// diffuse, reflection, refraction
	float bsdfProb[3] = { 1 - m.hasReflective - m.hasRefractive, m.hasReflective, m.hasRefractive };
	int index = 0;
	float bsdfCDF = 0;
	for (int i = 0; i < 3; ++i)
	{
		if (prob > bsdfCDF && prob <= bsdfCDF + bsdfProb[i])
		{
			index = i;
			break;
		}
		bsdfCDF += bsdfProb[i];
	}

	glm::vec3 color = glm::vec3(1, 1, 1);
	glm::vec3 rayD = pathSegment.ray.direction;
	// calculate and assign new ray direction
	glm::vec3 dir;
	switch (index)
	{
	case 1: // specular-reflective
		dir = glm::reflect(pathSegment.ray.direction, normal);
		color = m.color * m.specular.color;
		break;
	case 2: // specular-refractive
	{
		// naive refract computation
		//if (glm::dot(normal, rayD) < 0) // entering object
		//{
		//	dir = glm::refract(rayD, normal, 1.0f / m.indexOfRefraction);
		//}
		//else // leaving object
		//{
		//	dir = glm::refract(rayD, normal, m.indexOfRefraction);
		//}

		//if (glm::length(dir) < 1e-4f)
		//	dir = glm::reflect(rayD, normal);

		// use Schlick's approximation (Fresnel factor)
		float indexRatio = m.indexOfRefraction;
		float costheta = glm::dot(normal, rayD);
		if (costheta < 0)
			indexRatio = 1.0f / indexRatio;

		float R0 = (1 - indexRatio) / (1 + indexRatio);
		R0 = R0 * R0;
		float c = 1 - glm::abs(costheta);
		R0 = R0 + (1 - R0) * c*c*c*c*c;

		if (R0 > u01(rng))  // reflection
		{
			dir = glm::reflect(rayD, normal);
		}
		else
		{
			dir = glm::refract(rayD, normal, indexRatio);
		}

		//if (glm::length(dir) < 1e-4f)
		//	dir = glm::reflect(rayD, normal);

		color = m.color * m.specular.color;
		break;
	}
	default: // diffuse
		dir = calculateRandomDirectionInHemisphere(normal, rng);
		color = m.color;
		break;
	}

	pathSegment.ray.direction = glm::normalize(dir);
	pathSegment.ray.origin = intersect + 1e-3f * pathSegment.ray.direction;
	pathSegment.color *= color;
}
