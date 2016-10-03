#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
	return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
	return glm::vec3(m * v);
}

__host__ __device__ glm::vec3 lerpVec3(glm::vec3 &a, glm::vec3 &b, float &u) {
	return a*(1 - u) + u*(b);
}
__host__ __device__ void CalculateMotionBlurMatrix(Geom &geo, float& u)
{
	if (u > 0.9f)
		u = 1.0f;
	glm::vec3 translation = lerpVec3(geo.translation, geo.translation_dst, u);
	glm::vec3 rotation = lerpVec3(geo.rotation, geo.rotation_dst, u);
	glm::vec3 scale = lerpVec3(geo.scale, geo.scale_dst, u);

	glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
	glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
	glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);

	geo.transform = translationMat * rotationMat * scaleMat;
	geo.inverseTransform = glm::inverse(geo.transform);
	geo.invTranspose = glm::inverseTranspose(geo.transform);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, float u) {

	//motion blur
	if (u >= 0.0f)
		CalculateMotionBlurMatrix(box, u);

	Ray q;
	q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

	float tmin = -1e38f;
	float tmax = 1e38f;
	glm::vec3 tmin_n;
	glm::vec3 tmax_n;
	for (int xyz = 0; xyz < 3; ++xyz) {
		float qdxyz = q.direction[xyz];
		/*if (glm::abs(qdxyz) > 0.00001f)*/ {
			float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
			float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
			float ta = glm::min(t1, t2);
			float tb = glm::max(t1, t2);
			glm::vec3 n;
			n[xyz] = t2 < t1 ? +1 : -1;
			if (ta > 0 && ta > tmin) {
				tmin = ta;
				tmin_n = n;
			}
			if (tb < tmax) {
				tmax = tb;
				tmax_n = n;
			}
		}
	}

	if (tmax >= tmin && tmax > 0) {
		outside = true;
		if (tmin <= 0) {
			tmin = tmax;
			tmin_n = tmax_n;
			outside = false;
		}
		intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
		return glm::length(r.origin - intersectionPoint);
	}
	return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, float u) {

	//motion blur
	if (u >= 0.0f)
		CalculateMotionBlurMatrix(sphere, u);

	float radius = .5;

	glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

	float vDotDirection = glm::dot(rt.origin, rt.direction);
	float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
	if (radicand < 0) {
		return -1;
	}

	float squareRoot = sqrt(radicand);
	float firstTerm = -vDotDirection;
	float t1 = firstTerm + squareRoot;
	float t2 = firstTerm - squareRoot;

	float t = 0;
	if (t1 < 0 && t2 < 0) {
		return -1;
	}
	else if (t1 > 0 && t2 > 0) {
		t = min(t1, t2);
		outside = true;
	}
	else {
		t = max(t1, t2);
		outside = false;
	}

	glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

	intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
	normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
	if (!outside) {
		normal = -normal;
	}

	return glm::length(r.origin - intersectionPoint);
}



/**
*	test intersection function for Constructive Solid Geometry
*/
Geom csg_box, csg_sphere;
__host__ __device__ void csgSphereIntersectionTest(Geom sphere, Ray r,
	float &tmin, glm::vec3 &point_min, glm::vec3 & normal_min,
	float &tmax, glm::vec3 &point_max, glm::vec3 & normal_max)
{

	float radius = .5;

	glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

	float vDotDirection = glm::dot(rt.origin, rt.direction);
	float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
	if (radicand < 0) {
		tmin = -1;
		tmax = -1;
		return;
	}

	float squareRoot = sqrt(radicand);
	float firstTerm = -vDotDirection;
	float t1 = firstTerm + squareRoot;
	float t2 = firstTerm - squareRoot;

	if (t1 > 0 && t2 > 0)
	{
		tmin = glm::min(t1, t2); // local
		tmax = glm::max(t1, t2); // local

		glm::vec3 objspaceIntersection = getPointOnRay(rt, tmin);

		point_min = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
		normal_min = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
		tmin = glm::length(point_min - r.origin);

		objspaceIntersection = getPointOnRay(rt, tmax);
		point_max = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
		normal_max = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
		tmax = glm::length(point_max - r.origin);

	}
	else
	{
		tmin = tmax = -1;
	}

}

__host__ __device__ void csgBoxIntersectionTest(Geom box, Ray r,
	float &_tmin, glm::vec3 &point_min, glm::vec3 & normal_min,
	float &_tmax, glm::vec3 &point_max, glm::vec3 & normal_max)
{

	Ray q;
	q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

	float tmin = -1e38f;
	float tmax = 1e38f;
	glm::vec3 tmin_n;
	glm::vec3 tmax_n;
	for (int xyz = 0; xyz < 3; ++xyz) {
		float qdxyz = q.direction[xyz];
		/*if (glm::abs(qdxyz) > 0.00001f)*/ {
			float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
			float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
			float ta = glm::min(t1, t2);
			float tb = glm::max(t1, t2);
			glm::vec3 n;
			n[xyz] = t2 < t1 ? +1 : -1;
			if (ta > 0 && ta > tmin) {
				tmin = ta;
				tmin_n = n;
			}
			if (tb < tmax) {
				tmax = tb;
				tmax_n = n;
			}
		}
	}

	if (tmax >= tmin && tmax > 0 && tmin > 0) {
		_tmin = glm::min(tmin, tmax);
		_tmax = glm::max(tmin, tmax);
		point_min = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, _tmin), 1.0f));
		normal_min = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
		_tmin = glm::length(point_min - r.origin);

		point_max = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, _tmax), 1.0f));
		normal_max = glm::normalize(multiplyMV(box.transform, glm::vec4(tmax_n, 0.0f)));
		_tmax = glm::length(point_max - r.origin);
	}
	else
	{
		_tmin = _tmax = -1;
	}
}


/**
* reference: slides from CIS560 computer graphcis!
* I should REALLY REALLY REALLY read them carefully.
*/
__host__ __device__ void csgDifference(Geom A, Geom B, Ray r,
	float& t, glm::vec3 & point, glm::vec3 & normal, int & materialID)
{
	float Atmin, Atmax, Btmin, Btmax;
	glm::vec3 A_point_min, A_point_max, B_point_min, B_point_max;
	glm::vec3 A_normal_min, A_normal_max, B_normal_min, B_normal_max;

	if (A.type == CUBE)
		csgBoxIntersectionTest(A, r, Atmin, A_point_min, A_normal_min, Atmax, A_point_max, A_normal_max);
	else
		csgSphereIntersectionTest(A, r, Atmin, A_point_min, A_normal_min, Atmax, A_point_max, A_normal_max);

	if (B.type == CUBE)
		csgBoxIntersectionTest(B, r, Btmin, B_point_min, B_normal_min, Btmax, B_point_max, B_normal_max);
	else
		csgSphereIntersectionTest(B, r, Btmin, B_point_min, B_normal_min, Btmax, B_point_max, B_normal_max);


	if (Btmin < 0) // nothing hit B, return isx A
	{
		t = Atmin;
		point = A_point_min;
		normal = A_normal_min;
		materialID = A.materialid;
		return;
	}
	else // Btmin > 0
	{
		if (Atmin < Btmin)
		{
			t = Atmin;
			point = A_point_min;
			normal = A_normal_min;
			materialID = A.materialid;
			return;
		}
		else if (Btmax < Atmax)
		{
			t = Btmax;
			point = B_point_max;
			normal = -B_normal_max;
			materialID = B.materialid;
			return;
		}
		else
		{
			t = -1;
			return;
		}
	}
}

__host__ __device__ void csgUnion(Geom A, Geom B, Ray r,
	float& t, glm::vec3 & point, glm::vec3 & normal, int & materialID)
{
	float Atmin, Atmax, Btmin, Btmax;
	glm::vec3 A_point_min, A_point_max, B_point_min, B_point_max;
	glm::vec3 A_normal_min, A_normal_max, B_normal_min, B_normal_max;

	if (A.type == CUBE)
		csgBoxIntersectionTest(A, r, Atmin, A_point_min, A_normal_min, Atmax, A_point_max, A_normal_max);
	else
		csgSphereIntersectionTest(A, r, Atmin, A_point_min, A_normal_min, Atmax, A_point_max, A_normal_max);

	if (B.type == CUBE)
		csgBoxIntersectionTest(B, r, Btmin, B_point_min, B_normal_min, Btmax, B_point_max, B_normal_max);
	else
		csgSphereIntersectionTest(B, r, Btmin, B_point_min, B_normal_min, Btmax, B_point_max, B_normal_max);

	if (Atmin > 0 && Btmin > 0)
	{
		if (Atmin < Btmin)
		{
			t = Atmin;
			point = A_point_min;
			normal = A_normal_min;
			materialID = A.materialid;
			return;
		}
		else 
		{
			t = Btmin;
			point = B_point_min;
			normal = B_normal_min;
			materialID = B.materialid;
			return;
		}
	}
	else
	{
		if (Atmin < 0)
		{
			t = Btmin;
			point = B_point_min;
			normal = B_normal_min;
			materialID = B.materialid;
			return;
		}
		else
		{
			t = Atmin;
			point = A_point_min;
			normal = A_normal_min;
			materialID = A.materialid;
			return;
		}
	}
}

__host__ __device__ void csgIntersect(Geom A, Geom B, Ray r,
	float& t, glm::vec3 & point, glm::vec3 & normal, int & materialID)
{
	float Atmin, Atmax, Btmin, Btmax;
	glm::vec3 A_point_min, A_point_max, B_point_min, B_point_max;
	glm::vec3 A_normal_min, A_normal_max, B_normal_min, B_normal_max;

	if (A.type == CUBE)
		csgBoxIntersectionTest(A, r, Atmin, A_point_min, A_normal_min, Atmax, A_point_max, A_normal_max);
	else
		csgSphereIntersectionTest(A, r, Atmin, A_point_min, A_normal_min, Atmax, A_point_max, A_normal_max);

	if (B.type == CUBE)
		csgBoxIntersectionTest(B, r, Btmin, B_point_min, B_normal_min, Btmax, B_point_max, B_normal_max);
	else
		csgSphereIntersectionTest(B, r, Btmin, B_point_min, B_normal_min, Btmax, B_point_max, B_normal_max);

	if (Atmin < Btmin && Atmax > Btmin) // B
	{
		t = Btmin;
		point = B_point_min;
		normal = B_normal_min;
		materialID = B.materialid;
		return;
	}
	else if (Atmin > Btmin && Atmin < Btmax) //A
	{
		t = Atmin;
		point = A_point_min;
		normal = A_normal_min;
		materialID = A.materialid;
		return;
	}
	else
	{
		t = -1;
	}

}


__host__ __device__ float csgIntersectionTest(Geom csg, Ray r,
	Geom csg_primitive1, Geom csg_primitive2,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, int & csgMaterialID) 
{

	Ray r_local;
	r_local.origin = multiplyMV(csg.inverseTransform, glm::vec4(r.origin, 1.0f));
	r_local.direction = glm::normalize(multiplyMV(csg.inverseTransform, glm::vec4(r.direction, 0.0f)));

	float t;

	// Difference test
	csgDifference(csg_primitive1, csg_primitive2, r_local,
		t, intersectionPoint, normal, csgMaterialID);
	
	//// Union test
	//csgIntersect(csg_primitive1, csg_primitive2, r_local,
	//	t, intersectionPoint, normal, csgMaterialID);

	//// Intersect test
	//csgIntersect(csg_primitive1, csg_primitive2, r_local,
	//	t, intersectionPoint, normal, csgMaterialID);

	if (t < 0)
	{
		return -1;
	}
	else
	{
		intersectionPoint = multiplyMV(csg.transform, glm::vec4(intersectionPoint, 1.f));
		normal = glm::normalize(multiplyMV(csg.invTranspose, glm::vec4(normal, 0.f)));
		t = glm::length(intersectionPoint - r.origin);
		return t;
	}
	
}