﻿// smallptCUDA by Sam Lapere, 2015
// based on smallpt, a path tracer by Kevin Beason, 2008  

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "helper_math.h"// from http://www.icmc.usp.br/~castelo/CUDA/common/inc/cutil_math.h
#include <cuda.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"
#undef __CUDA_INTERNAL_COMPILATION__
#include <vector_types.h>
#include <vector_functions.h>
#include "device_launch_parameters.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_render.h"
#include "linear_algebra.h"
#include "loadModel.h"

#define M_PI 3.14159265359f
#define width 512	// screenwidth
#define height 512	// screenheight
#define samps  1	// samples per pixel per pass

int total_number_of_triangles = 0;
int frames = 0;

// scene bounding box
float3 scene_aabbox_min;
float3 scene_aabbox_max;

// the scene triangles are stored in a 1D CUDA texture of float4 for memory alignment
// store two edges instead of vertices
// each triangle is stored as three float4s: (float4 first_vertex, float4 edge1, float4 edge2)
texture<float4, 1, cudaReadModeElementType> triangle_texture;
texture<float4, 1, cudaReadModeElementType> bvhNodesPosTexture;
texture<int, 1, cudaReadModeElementType> triIndicesTexture;

// hardcoded camera position
__device__ float3 firstcamorig = { 50, 52, 295.6 };

// OpenGL vertex buffer object for real-time viewport
GLuint vbo;
void* d_vbo_buffer = NULL;

__device__ __inline__ int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }

__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }

// standard ray box intersection routines (for debugging purposes only)
// based on Intersect::RayBox() in original Aila/Laine code
__device__ __inline__ float spanBeginKepler2(float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z, float d) {

	Vector3f t0 = Vector3f(lo_x, lo_y, lo_z);
	Vector3f t1 = Vector3f(hi_x, hi_y, hi_z);

	Vector3f realmin = min3f(t0, t1);

	float raybox_tmin = realmin.max(); // maxmin

	//return Vec2f(tmin, tmax);
	return raybox_tmin;
}

__device__ __inline__ float spanEndKepler2(float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z, float d) {

	Vector3f t0 = Vector3f(lo_x, lo_y, lo_z);
	Vector3f t1 = Vector3f(hi_x, hi_y, hi_z);

	Vector3f realmax = max3f(t0, t1);

	float raybox_tmax = realmax.min(); /// minmax

	//return Vec2f(tmin, tmax);
	return raybox_tmax;
}


__device__ Vector3f intersectRayTriangle(const Vector4f& rayOri, const Vector4f& rayDir, const Vector3f& v0, const Vector3f& v1, const Vector3f& v2)
{
	const float EPSILON = 0.00001f;
	const Vector3f miss(WORLD_MAX, WORLD_MAX, WORLD_MAX);

	Vector3f edge_u = v1 - v0;
	Vector3f edge_v = v2 - v0;

	Vector3f vec_t = Vector3f(rayOri.x, rayOri.y, rayOri.z) - v0;
	Vector3f vec_p = cross(Vector3f(rayDir.x, rayDir.y, rayDir.z), edge_v);
	float det = dot(edge_u, vec_p);
	float invdet = 1.0f / det;
	float u = dot(vec_t, vec_p) * invdet;
	Vector3f vec_q = cross(vec_t, edge_u);
	float v = dot(Vector3f(rayDir.x, rayDir.y, rayDir.z), edge_u);

	if (det > EPSILON)
	{
		if (u < 0.0f || u > 1.0f) return miss; // 1.0 want = det * 1/det  
		if (v < 0.0f || (u + v) > 1.0f) return miss;
		// if u and v are within these bounds, continue and go to float t = dot(...	           
	}

	else if (det < -EPSILON)
	{
		if (u > 0.0f || u < 1.0f) return miss;
		if (v > 0.0f || (u + v) < 1.0f) return miss;
		// else continue
	}

	else // if det is not larger (more positive) than EPSILON or not smaller (more negative) than -EPSILON, there is a "miss"
		return miss;

	float t = dot(edge_v, vec_q) * invdet;

	if (t > rayOri.w && t < rayOri.w)
		return Vector3f(u, v, t);

	// otherwise (t < raytmin or t > raytmax) miss
	return miss;
}

struct Ray {
	float3 orig;	// ray origin
	float3 dir;		// ray direction	
	__device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {}
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance(), only DIFF used here

// SPHERES

struct Sphere {

	float rad;				// radius 
	float3 pos, emi, col;	// position, emission, color 
	Refl_t refl;			// reflection type (DIFFuse, SPECular, REFRactive)

	__device__ float intersect(const Ray& r) const { // returns distance, 0 if nohit 

		// Ray/sphere intersection
		// Quadratic formula required to solve ax^2 + bx + c = 0 
		// Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 

		float3 op = pos - r.orig;  // 
		float t, epsilon = 0.01f;
		float b = dot(op, r.dir);
		float disc = b * b - dot(op, op) + rad * rad; // discriminant
		if (disc < 0) return 0; else disc = sqrtf(disc);
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
	}
};

// TRIANGLES

// the classic ray triangle intersection: http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
// for an explanation see http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection

__device__ float RayTriangleIntersection(const Ray& r,
	const float3& v0,
	const float3& edge1,
	const float3& edge2)
{

	float3 tvec = r.orig - v0;
	float3 pvec = cross(r.dir, edge2);
	float  det = dot(edge1, pvec);

	det = __fdividef(1.0f, det);  // CUDA intrinsic function 

	float u = dot(tvec, pvec) * det;

	if (u < 0.0f || u > 1.0f)
		return -1.0f;

	float3 qvec = cross(tvec, edge1);

	float v = dot(r.dir, qvec) * det;

	if (v < 0.0f || (u + v) > 1.0f)
		return -1.0f;

	return dot(edge2, qvec) * det;
}

__device__ float3 getTriangleNormal(const int triangleIndex) {

	float4 edge1 = tex1Dfetch(triangle_texture, triangleIndex * 3 + 1);
	float4 edge2 = tex1Dfetch(triangle_texture, triangleIndex * 3 + 2);

	// cross product of two triangle edges yields a vector orthogonal to triangle plane
	float3 trinormal = cross(make_float3(edge1.x, edge1.y, edge1.z), make_float3(edge2.x, edge2.y, edge2.z));
	trinormal = normalize(trinormal);

	return trinormal;
}

__device__ void intersectAllTriangles(const Ray& r, float& t_scene, int& triangle_id, const int number_of_triangles, int& geomtype) {

	for (int i = 0; i < number_of_triangles; i++)
	{
		// the triangles are packed into the 1D texture using three consecutive float4 structs for each triangle, 
		// first float4 contains the first vertex, second float4 contains the first precomputed edge, third float4 contains second precomputed edge like this: 
		// (float4(vertex.x,vertex.y,vertex.z, 0), float4 (egde1.x,egde1.y,egde1.z,0),float4 (egde2.x,egde2.y,egde2.z,0)) 

		// i is triangle index, each triangle represented by 3 float4s in triangle_texture
		float4 v0 = tex1Dfetch(triangle_texture, i * 3);
		float4 edge1 = tex1Dfetch(triangle_texture, i * 3 + 1);
		float4 edge2 = tex1Dfetch(triangle_texture, i * 3 + 2);

		// intersect ray with reconstructed triangle	
		float t = RayTriangleIntersection(r,
			make_float3(v0.x, v0.y, v0.z),
			make_float3(edge1.x, edge1.y, edge1.z),
			make_float3(edge2.x, edge2.y, edge2.z));

		// keep track of closest distance and closest triangle
		// if ray/tri intersection finds an intersection point that is closer than closest intersection found so far
		if (t < t_scene && t > 0.001)
		{
			t_scene = t;
			triangle_id = i;
			geomtype = 3;
		}
	}
}
__device__ void intersectBVH(const float4 rayOri, const float4 rayDir)
{
	int thread_index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
}

// AXIS ALIGNED BOXES

// helper functions
inline __device__ float3 minf3(float3 a, float3 b) { return make_float3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z); }
inline __device__ float3 maxf3(float3 a, float3 b) { return make_float3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z); }
inline __device__ float minf1(float a, float b) { return a < b ? a : b; }
inline __device__ float maxf1(float a, float b) { return a > b ? a : b; }

struct Box {

	float3 min; // minimum bounds
	float3 max; // maximum bounds
	float3 emi; // emission
	float3 col; // colour
	Refl_t refl; // material type

	// ray/box intersection
	// for theoretical background of the algorithm see 
	// http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
	// optimised code from http://www.gamedev.net/topic/495636-raybox-collision-intersection-point/
	__device__ float intersect(const Ray& r) const {

		float epsilon = 0.001f; // required to prevent self intersection

		float3 tmin = (min - r.orig) / r.dir;
		float3 tmax = (max - r.orig) / r.dir;

		float3 real_min = minf3(tmin, tmax);
		float3 real_max = maxf3(tmin, tmax);

		float minmax = minf1(minf1(real_max.x, real_max.y), real_max.z); // 射线离开boundingbox后，与另一个轴的交点这段路程
		float maxmin = maxf1(maxf1(real_min.x, real_min.y), real_min.z); // 射线与一个轴相交后，到与boundingbox相交这段前的路程

		if (minmax >= maxmin) { return maxmin > epsilon ? maxmin : 0; }
		else return 0;
	}

	// calculate normal for point on axis aligned box
	__device__ float3 Box::normalAt(float3& point) {

		float3 normal = make_float3(0.f, 0.f, 0.f);
		float min_distance = 1e8;
		float distance;
		float epsilon = 0.001f;

		if (fabs(min.x - point.x) < epsilon) normal = make_float3(-1, 0, 0);
		else if (fabs(max.x - point.x) < epsilon) normal = make_float3(1, 0, 0);
		else if (fabs(min.y - point.y) < epsilon) normal = make_float3(0, -1, 0);
		else if (fabs(max.y - point.y) < epsilon) normal = make_float3(0, 1, 0);
		else if (fabs(min.z - point.z) < epsilon) normal = make_float3(0, 0, -1);
		else normal = make_float3(0, 0, 1);

		return normal;
	}
};

// scene: 9 spheres forming a Cornell box
// small enough to fit in constant GPU memory
__constant__ Sphere spheres[] = {
	// FORMAT: { float radius, float3 position, float3 emission, float3 colour, Refl_t material }
	// cornell box
	//{ 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left 1e5f
	//{ 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Right 
	//{ 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back 
	//{ 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 0.00f, 0.00f, 0.00f }, DIFF }, //Front 
	//{ 1e5f, { 50.0f, -1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Bottom 
	//{ 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
	//{ 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 0.99f, 0.99f, 0.99f }, SPEC }, // small sphere 1
	//{ 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.f, .0f }, { 0.09f, 0.49f, 0.3f }, REFR }, // small sphere 2
	//{ 600.0f, { 50.0f, 681.6f - .5f, 81.6f }, { 3.0f, 2.5f, 2.0f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light 12, 10 ,8

	//outdoor scene: radius, position, emission, color, material

	//{ 1600, { 3000.0f, 10, 6000 }, { 37, 34, 30 }, { 0.f, 0.f, 0.f }, DIFF },  // 37, 34, 30 // sun
	//{ 1560, { 3500.0f, 0, 7000 }, { 50, 25, 2.5 }, { 0.f, 0.f, 0.f }, DIFF },  //  150, 75, 7.5 // sun 2
	
	{ 10000, { 50.0f, 40.8f, -1060 }, { 0.0003, 0.01, 0.15 }, { 0.175f, 0.175f, 0.25f }, DIFF }, // sky
	{ 100000, { 50.0f, -100000, 0 }, { 0.0, 0.0, 0 }, { 0.8f, 0.2f, 0.f }, DIFF }, // ground
	{ 82.5, { 30.0f, 180.5, 42 }, { 16, 12, 6 }, { .6f, .6f, 0.6f }, DIFF },  // small sphere 1
	/*
	{ 110000, { 50.0f, -110048.5, 0 }, { 3.6, 2.0, 0.2 }, { 0.f, 0.f, 0.f }, DIFF },  // horizon brightener
	
	{ 4e4, { 50.0f, -4e4 - 30, -3000 }, { 0, 0, 0 }, { 0.2f, 0.2f, 0.2f }, DIFF }, // mountains
	
	
	{ 12, { 115.0f, 10, 105 }, { 0.0, 0.0, 0.0 }, { 0.9f, 0.9f, 0.9f }, REFR },  // small sphere 2
	{ 22, { 65.0f, 22, 24 }, { 0, 0, 0 }, { 0.9f, 0.9f, 0.9f }, SPEC }, // small sphere 3
	
	*/
};

__constant__ Box boxes[] = {
	// FORMAT: { float3 minbounds,    float3 maxbounds,         float3 emission,    float3 colour,       Refl_t }
	
	/*{ { 5.0f, 0.0f, 70.0f }, { 45.0f, 11.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f }, DIFF },
	{ { 85.0f, 0.0f, 95.0f }, { 95.0f, 20.0f, 105.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f }, DIFF },*/
	{ { 75.0f, 20.0f, 85.0f }, { 105.0f, 22.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f }, DIFF },
	
};


__device__ inline bool intersect_scene(const Ray& r, float& t, int& sphere_id, int& box_id, int& triangle_id, const int number_of_triangles, int& geomtype, const float3& bbmin, const float3& bbmax) {

	float tmin = 1e20;
	float tmax = -1e20;
	float d = 1e21;
	float k = 1e21;
	float q = 1e21;
	float inf = t = 1e20;
	
	// SPHERES
	// intersect all spheres in the scene
	float numspheres = sizeof(spheres) / sizeof(Sphere);
	for (int i = int(numspheres); i--;)  // for all spheres in scene
		// keep track of distance from origin to closest intersection point
		if ((d = spheres[i].intersect(r)) && d < t) { t = d; sphere_id = i; geomtype = 1; }

	// BOXES
	// intersect all boxes in the scene
	float numboxes = sizeof(boxes) / sizeof(Box);
	for (int i = int(numboxes); i--;) // for all boxes in scene
		if ((k = boxes[i].intersect(r)) && k < t) { t = k; box_id = i; geomtype = 2; }
	
	// TRIANGLES
	Box scene_bbox; // bounding box around triangle meshes
	scene_bbox.min = bbmin;
	scene_bbox.max = bbmax;

	// if ray hits bounding box of triangle meshes, intersect ray with all triangles
    //scene_bbox.intersect(r)
	intersectAllTriangles(r, t, triangle_id, number_of_triangles, geomtype);

	// t is distance to closest intersection of ray with all primitives in the scene (spheres, boxes and triangles)
	return t < inf;
}


// hash function to calculate new seed for each frame
// see http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
uint WangHash(uint a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

__device__ void intersecctBVHTriangle()
{
	int thread_index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	int nodeAddress = 0; // 从 0 号点开始
	
	while (true)
	{
		while (true)
		{
			float4 child_pos_x = tex1Dfetch(bvhNodesPosTexture, nodeAddress);
			float4 child_pos_y = tex1Dfetch(bvhNodesPosTexture, nodeAddress);
			float4 child_pos_z = tex1Dfetch(bvhNodesPosTexture, nodeAddress);
		}
	}
}

// radiance function
// compute path bounces in scene and accumulate returned color from each path sgment
__device__ float3 radiance(Ray& r, curandState* randstate, const int totaltris, const float3& scene_aabb_min, const float3& scene_aabb_max) { // returns ray color

	// colour mask
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);
	// accumulated colour
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f);

	for (int bounces = 0; bounces < 5; bounces++) {  // iteration up to 4 bounces (instead of recursion in CPU code)

		// reset scene intersection function parameters
		float t = 100000; // distance to intersection 
		int sphere_id = -1;
		int box_id = -1;   // index of intersected sphere 
		int triangle_id = -1;
		int geomtype = -1;
		float3 f;  // primitive colour
		float3 emit; // primitive emission colour
		float3 x; // intersection point
		float3 n; // normal
		float3 nl; // oriented normal
		float3 d; // ray direction of next path segment
		Refl_t refltype;

		int hitSphereIdx = -1;
		int hitTriIdx = -1;
		int bestTriIdx = -1;
		float hitSphereDist = 1e20;
		float hitDistance = 1e20;

		// intersect ray with scene
		// intersect_scene keeps track of closest intersected primitive and distance to closest intersection point
		if (!intersect_scene(r, t, sphere_id, box_id, triangle_id, totaltris, geomtype, scene_aabb_min, scene_aabb_max))
			return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black

		// else: we've got a hit with a scene primitive
		// determine geometry type of primitive: sphere/box/triangle

		// if sphere:
		if (geomtype == 1) {
			Sphere& sphere = spheres[sphere_id]; // hit object with closest intersection
			x = r.orig + r.dir * t;  // intersection point on object
			n = normalize(x - sphere.pos);		// normal
			nl = dot(n, r.dir) < 0 ? n : n * -1; // correctly oriented normal
			f = sphere.col;   // object colour
			refltype = sphere.refl;
			emit = sphere.emi;  // object emission
			accucolor += (mask * emit);
		}

		// if box:
		if (geomtype == 2) {
			Box& box = boxes[box_id];
			x = r.orig + r.dir * t;  // intersection point on object
			n = normalize(box.normalAt(x)); // normal
			nl = dot(n, r.dir) < 0 ? n : n * -1;  // correctly oriented normal
			f = box.col;  // box colour
			refltype = box.refl;
			emit = box.emi; // box emission
			accucolor += (mask * emit);
		}

		// if triangle:
		if (geomtype == 3) {
			int tri_index = triangle_id;
			x = r.orig + r.dir * t;  // intersection point
			n = normalize(getTriangleNormal(tri_index));  // normal 
			nl = dot(n, r.dir) < 0 ? n : n * -1;  // correctly oriented normal

			// colour, refltype and emit value are hardcoded and apply to all triangles
			// no per triangle material support yet
			f = make_float3(0.9f, 0.4f, 0.1f);  // triangle colour
			refltype = DIFF;
			emit = make_float3(0.0f, 0.0f, 0.0f);
			accucolor += (mask * emit);
		}

		// SHADING: diffuse, specular or refractive

		// ideal diffuse reflection (see "Realistic Ray Tracing", P. Shirley)
		if (refltype == DIFF) {

			// create 2 random numbers
			float r1 = 2 * M_PI * curand_uniform(randstate);
			float r2 = curand_uniform(randstate);
			float r2s = sqrtf(r2);

			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			float3 w = nl;
			float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
			float3 v = cross(w, u);

			// compute cosine weighted random ray direction on hemisphere 
			d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));

			// offset origin next path segment to prevent self intersection
			x += nl * 0.03;

			// multiply mask with colour of object
			mask *= f;
		}

		// ideal specular reflection (mirror) 
		if (refltype == SPEC) {

			// compute relfected ray direction according to Snell's law
			d = r.dir - 2.0f * n * dot(n, r.dir);

			// offset origin next path segment to prevent self intersection
			x += nl * 0.01f;

			// multiply mask with colour of object
			mask *= f;
		}

		// ideal refraction (based on smallpt code by Kevin Beason)
		if (refltype == REFR) {

			bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?
			float nc = 1.0f;  // Index of Refraction air
			float nt = 1.5f;  // Index of Refraction glass/water
			float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
			float ddn = dot(r.dir, nl);
			float cos2t = 1.0f - nnt * nnt * (1.f - ddn * ddn);

			if (cos2t < 0.0f) // total internal reflection 
			{
				d = reflect(r.dir, n); //d = r.dir - 2.0f * n * dot(n, r.dir);
				x += nl * 0.01f;
			}
			else // cos2t > 0
			{
				// compute direction of transmission ray
				float3 tdir = normalize(r.dir * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrtf(cos2t))));

				float R0 = (nt - nc) * (nt - nc) / (nt + nc) * (nt + nc);
				float c = 1.f - (into ? -ddn : dot(tdir, n));
				float Re = R0 + (1.f - R0) * c * c * c * c * c;
				float Tr = 1 - Re; // Transmission
				float P = .25f + .5f * Re;
				float RP = Re / P;
				float TP = Tr / (1.f - P);

				// randomly choose reflection or transmission ray
				if (curand_uniform(randstate) < 0.25) // reflection ray
				{
					mask *= RP;
					d = reflect(r.dir, n);
					x += nl * 0.02f;
				}
				else // transmission ray
				{
					mask *= TP;
					d = tdir; //r = Ray(x, tdir); 
					x += nl * 0.0005f; // epsilon must be small to avoid artefacts
				}
			}
		}

		// set up origin and direction of next path segment
		r.orig = x;
		r.dir = d;
	}

	// add radiance up to a certain ray depth
	// return accumulated ray colour after all bounces are computed
	return accucolor;
}

// required to convert colour to a format that OpenGL can display  
union Colour  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

__global__ void render_kernel(float3* output, float3* accumbuffer, const int numtriangles, int framenumber, uint hashedframenumber, float3 scene_bbmin, float3 scene_bbmax) {   // float3 *gputexdata1, int *texoffsets

	// assign a CUDA thread to every pixel by using the threadIndex
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// create random number generator, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedframenumber + threadId, 0, 0, &randState);

	Ray cam(firstcamorig, normalize(make_float3(0, -0.042612, -1)));
	float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f);  // ray direction offset along X-axis 
	float3 cy = normalize(cross(cx, cam.dir)) * .5135; // ray dir offset along Y-axis, .5135 is FOV angle
	float3 pixelcol; // final pixel color       

	int i = (height - y - 1) * width + x; // pixel index

	pixelcol = make_float3(0.0f, 0.0f, 0.0f); // reset to zero for every pixel	

	for (int s = 0; s < samps; s++) {

		// compute primary ray direction
		float3 d = cx * ((.25 + x) / width - .5) + cy * ((.25 + y) / height - .5) + cam.dir;
		// normalize primary ray direction
		d = normalize(d);
		// add accumulated colour from path bounces
		pixelcol += radiance(Ray(cam.orig + d * 40, d), &randState, numtriangles, scene_bbmin, scene_bbmax) * (1. / samps);
	}       // Camera rays are pushed ^^^^^ forward to start in interior 

	// add pixel colour to accumulation buffer (accumulates all samples) 
	accumbuffer[i] += pixelcol;
	// averaged colour: divide colour by the number of calculated frames so far
	float3 tempcol = accumbuffer[i] / framenumber;

	Colour fcolour;
	float3 colour = make_float3(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f));
	// convert from 96-bit to 24-bit colour + perform gamma correction
	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255), (unsigned char)(powf(colour.y, 1 / 2.2f) * 255), (unsigned char)(powf(colour.z, 1 / 2.2f) * 255), 1);
	// store pixel coordinates and pixelcolour in OpenGL readable outputbuffer
	output[i] = make_float3(x, y, fcolour.c);
}

void Timer(int obsolete) {

	glutPostRedisplay();
	glutTimerFunc(30, Timer, 0);
}

__device__ float timer = 0.0f;

inline float clamp(float x) { return x < 0 ? 0 : x>1 ? 1 : x; }

//inline int toInt(float x){ return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }  // RGB float in range [0,1] to int in range [0, 255]

// buffer for accumulating samples over several frames
float3* accumulatebuffer;
// output buffer
float3* dptr;

void disp(void)
{
	frames++;
	printf_s("frame = %d\n", frames);
	if (frames == 20)
	{
		//{ cudaMemset(accumulatebuffer, 1, width * height * sizeof(float3)); frames = 0; }
	}
	cudaThreadSynchronize();

	// map vertex buffer object for acces by CUDA 
	cudaGLMapBufferObject((void**)&dptr, vbo);

	//clear all pixels:
	glClear(GL_COLOR_BUFFER_BIT);

	// RAY TRACING:
	// dim3 grid(WINDOW / block.x, WINDOW / block.y, 1);
	// dim3 CUDA specific syntax, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 block(16, 16, 1);
	dim3 grid(width / block.x, height / block.y, 1);

	// launch CUDA path tracing kernel, pass in a hashed seed based on number of frames
	render_kernel << < grid, block >> > (dptr, accumulatebuffer, total_number_of_triangles, frames, WangHash(frames), scene_aabbox_max, scene_aabbox_min);  // launches CUDA render kernel from the host

	cudaThreadSynchronize();

	// unmap buffer
	cudaGLUnmapBufferObject(vbo);
	//glFlush();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, width * height);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
	glutPostRedisplay();
}

// load triangle data in a CUDA texture
extern "C"
{
	void bindTriangles(float* dev_triangle_p, unsigned int number_of_triangles)
	{
		triangle_texture.normalized = false;                      // access with normalized texture coordinates
		triangle_texture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		triangle_texture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * number_of_triangles * 3;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, triangle_texture, dev_triangle_p, channelDesc, size);
	}
}


TriangleMesh mesh1;
float* dev_triangle_p; // the cuda device pointer that points to the uploaded triangles


void cudaRender(const int* triInds)
{
	bool firstTime = true;
	if (firstTime)
	{
		firstTime = false;
		cudaChannelFormatDesc channel0desc = cudaCreateChannelDesc<int>();
		cudaBindTexture(NULL, &triIndicesTexture, triInds, &channel0desc, 1 * sizeof(int));
	}
}

// 1. load triangle mesh data from obj files
// 2. copy data to CPU memory (into vector<float4> triangles)
// 3. copy to CUDA global memory (allocated with dev_triangle_p pointer)
// 4. copy to CUDA texture memory with bindtriangles()
void initCUDAmemoryTriMesh()
{
	
	loadObj("cube.obj");
	//loadObj("cube.obj", mesh1);
	// scalefactor and offset to position/scale triangle meshes
	float scalefactor1 = 10;
	float scalefactor2 = 10;  // 300
	Vector3f offset1 = Vector3f(90, 22, 100);// (30, -2, 80);
	float3 offset2 = make_float3(30, 12, 80);

	std::vector<float4> cuda_triangles;
	
	for (unsigned int i = 0; i < 12; i++)
	{
		int idx1 = triangles[i]._idx1;
		int idx2 = triangles[i]._idx2;
		int idx3 = triangles[i]._idx3;
		// make a local copy of the triangle vertices
		Vector3f v0 = vertices_pos[idx1 - 1];
		Vector3f v1 = vertices_pos[idx2 - 1];
		Vector3f v2 = vertices_pos[idx3 - 1];

		// scale
		v0 *= scalefactor1;
		v1 *= scalefactor1;
		v2 *= scalefactor1;

		// translate
		v0 += offset1;
		v1 += offset1;
		v2 += offset1;

		// store triangle data as float4
		// store two edges per triangle instead of vertices, to save some calculations in the
		// ray triangle intersection test
		cuda_triangles.push_back(make_float4(v0.x, v0.y, v0.z, 0));
		cuda_triangles.push_back(make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, 0));
		cuda_triangles.push_back(make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, 0));
	}
	/*
	float temp = 0;
	// compute bounding box of this mesh
	//mesh1.bounding_box[0] *= scalefactor1; mesh1.bounding_box[0] += offset1;
	//mesh1.bounding_box[1] *= scalefactor1; mesh1.bounding_box[1] += offset1;
	
	for (unsigned int i = 0; i < mesh1.faces.size(); i++)
	{
		float3 v0 = mesh1.verts[mesh1.faces[i].v[0] - 1];
		float3 v1 = mesh1.verts[mesh1.faces[i].v[1] - 1];
		float3 v2 = mesh1.verts[mesh1.faces[i].v[2] - 1];

		v0 *= scalefactor2;
		v1 *= scalefactor2;
		v2 *= scalefactor2;

		v0 += offset2;
		v1 += offset2;
		v2 += offset2;

		cuda_triangles.push_back(make_float4(v0.x, v0.y, v0.z, 0));
		cuda_triangles.push_back(make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, 1));
		cuda_triangles.push_back(make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, 0));
	}

	mesh1.bounding_box[0] *= scalefactor2; mesh1.bounding_box[0] += offset2;
	mesh1.bounding_box[1] *= scalefactor2; mesh1.bounding_box[1] += offset2;
	*/
	//std::cout << "total number of triangles check:" << mesh1.faces.size() + mesh2.faces.size() << " == " << cuda_triangles.size() / 3 << std::endl;
	
	// calculate total number of triangles in the scene
	size_t triangle_size = cuda_triangles.size() * sizeof(float4);
	int total_num_triangles = cuda_triangles.size() / 3;
	total_number_of_triangles = total_num_triangles;

	if (triangle_size > 0)
	{
		// allocate memory for the triangle meshes on the GPU
		cudaMalloc((void**)&dev_triangle_p, triangle_size);

		// copy triangle data to GPU
		cudaMemcpy(dev_triangle_p, &cuda_triangles[0], triangle_size, cudaMemcpyHostToDevice);

		// load triangle data into a CUDA texture
		bindTriangles(dev_triangle_p, total_num_triangles);
	}
	// compute scene bounding box by merging bounding boxes of individual meshes 
	
	scene_aabbox_min = make_float3(-WORLD_MAX,- WORLD_MAX, -WORLD_MAX);
	scene_aabbox_max = make_float3(WORLD_MAX, WORLD_MAX, WORLD_MAX);
	//scene_aabbox_min = mesh1.bounding_box[0] - make_float3(1,1,1);
	//scene_aabbox_max = mesh1.bounding_box[1] + make_float3(1, 1, 1);
	//scene_aabbox_min = fminf(scene_aabbox_min, mesh1.bounding_box[0]);
	//scene_aabbox_max = fmaxf(scene_aabbox_max, mesh1.bounding_box[1]);
	
}



void createVBO(GLuint* vbo)
{
	//create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//initialize VBO
	unsigned int size = width * height * sizeof(float3);  // 3 floats
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//register VBO with CUDA
	cudaGLRegisterBufferObject(*vbo);
}

int main(int argc, char** argv) {

	// allocate memmory for the accumulation buffer on the GPU
	cudaMalloc(&accumulatebuffer, width * height * sizeof(float3));
	// load triangle meshes in CUDA memory
	initCUDAmemoryTriMesh();
	// init glut for OpenGL viewport
	glutInit(&argc, argv);
	// specify the display mode to be RGB and single buffering
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	// specify the initial window position
	glutInitWindowPosition(100, 100);
	// specify the initial window size
	glutInitWindowSize(width, height);
	// create the window and set title
	glutCreateWindow("Basic triangle mesh path tracer in CUDA");
	// init OpenGL
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, width, 0.0, height);
	fprintf(stderr, "OpenGL initialized \n");
	// register callback function to display graphics:
	glutDisplayFunc(disp);
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		exit(0);
	}
	fprintf(stderr, "glew initialized  \n");
	// call Timer():
	Timer(0);
	//create VBO (vertex buffer object)
	createVBO(&vbo);
	fprintf(stderr, "VBO created  \n");
	// enter the main loop and process events
	fprintf(stderr, "Entering glutMainLoop...  \n");
	glutMainLoop();

	// free CUDA memory on exit
	cudaFree(accumulatebuffer);
	cudaFree(dev_triangle_p);
	cudaFree(dptr);
}
