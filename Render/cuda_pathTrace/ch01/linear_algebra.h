/*
*  CUDA based triangle mesh path tracer using BVH acceleration by Sam lapere, 2016
*  BVH implementation based on real-time CUDA ray tracer by Thanassis Tsiodras,
*  http://users.softlab.ntua.gr/~ttsiod/cudarenderer-BVH.html
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#ifndef __LINEAR_ALGEBRA_H_
#define __LINEAR_ALGEBRA_H_

#include <cuda_runtime.h> // for __host__  __device__
#include <math.h>
#define WORLD_MIN          (1.175494351e-38f)
#define WORLD_MAX          (3.402823466e+38f)
inline __host__ __device__ float max1f(const float& a, const float& b) { return (a < b) ? b : a; }
inline __host__ __device__ float min1f(const float& a, const float& b) { return (a > b) ? b : a; }
struct Vector2f
{
	union 
	{
		struct { float x, y; };
		float _v[2];
	};

	__host__ __device__ Vector2f(float _x = 0, float _y = 0) :x(_x), y(_y) {};
	__host__ __device__ Vector2f(const Vector2f& v):x(v.x),y(v.y){}

	inline __host__ __device__ bool operator==(const Vector2f& v) { return x == v.x && y == v.y; }
};

struct Vector3f
{
	union {
		struct { float x, y, z; };
		float _v[3];
	};

	__host__ __device__ Vector3f(float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z) {}
	__host__ __device__ Vector3f(const Vector3f& v) : x(v.x), y(v.y), z(v.z) {}
	__host__ __device__ Vector3f(const float4& v) : x(v.x), y(v.y), z(v.z) {}
	inline __host__ __device__ float length() { return sqrtf(x * x + y * y + z * z); }
	// sometimes we dont need the sqrt, we are just comparing one length with another
	inline __host__ __device__ float lengthsq() { return x * x + y * y + z * z; }
	inline __host__ __device__ float max() { return max1f(max1f(x, y), z); }
	inline __host__ __device__ float min() { return min1f(min1f(x, y), z); }
	inline __host__ __device__ void normalize() { float norm = sqrtf(x * x + y * y + z * z); x /= norm; y /= norm; z /= norm; }
	inline __host__ __device__ Vector3f& operator+=(const Vector3f& v) { x += v.x; y += v.y; z += v.z; return *this; }
	inline __host__ __device__ Vector3f& operator-=(const Vector3f& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
	inline __host__ __device__ Vector3f& operator*=(const float& a) { x *= a; y *= a; z *= a; return *this; }
	inline __host__ __device__ Vector3f& operator*=(const Vector3f& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
	inline __host__ __device__ Vector3f operator*(float a) const { return Vector3f(x * a, y * a, z * a); }
	inline __host__ __device__ Vector3f operator/(float a) const { return Vector3f(x / a, y / a, z / a); }
	inline __host__ __device__ Vector3f operator*(const Vector3f& v) const { return Vector3f(x * v.x, y * v.y, z * v.z); }
	inline __host__ __device__ Vector3f operator+(const Vector3f& v) const { return Vector3f(x + v.x, y + v.y, z + v.z); }
	inline __host__ __device__ Vector3f operator-(const Vector3f& v) const { return Vector3f(x - v.x, y - v.y, z - v.z); }
	inline __host__ __device__ Vector3f& operator/=(const float& a) { x /= a; y /= a; z /= a; return *this; }
	inline __host__ __device__ bool operator!=(const Vector3f& v) { return x != v.x || y != v.y || z != v.z; }
};



inline __host__ __device__ Vector3f min3f(const Vector3f& v1, const Vector3f& v2) { return Vector3f(v1.x < v2.x ? v1.x : v2.x, v1.y < v2.y ? v1.y : v2.y, v1.z < v2.z ? v1.z : v2.z); }
inline __host__ __device__ Vector3f max3f(const Vector3f& v1, const Vector3f& v2) { return Vector3f(v1.x > v2.x ? v1.x : v2.x, v1.y > v2.y ? v1.y : v2.y, v1.z > v2.z ? v1.z : v2.z); }
inline __host__ __device__ Vector3f cross(const Vector3f& v1, const Vector3f& v2) { return Vector3f(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); }
inline __host__ __device__ float dot(const Vector3f& v1, const Vector3f& v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
inline __host__ __device__ float dot(const Vector3f& v1, const float4& v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
inline __host__ __device__ float dot(const float4& v1, const Vector3f& v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
inline __host__ __device__ float distancesq(const Vector3f& v1, const Vector3f& v2) { return (v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z); }
inline __host__ __device__ float distance(const Vector3f& v1, const Vector3f& v2) { return sqrtf((v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z)); }

struct Vector4f
{
	union {
		struct { float x, y, z, w; };
		float _v[4];
	};
	///float x, y, z, w;

	__host__ __device__ Vector4f(float _x = 0, float _y = 0, float _z = 0, float _w = 0) : x(_x), y(_y), z(_z), w(_w) {}
	__host__ __device__ Vector4f(const Vector4f& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
	__host__ __device__ Vector4f(const Vector3f& v, const float a) : x(v.x), y(v.y), z(v.z), w(a) {}

	inline __host__ __device__ Vector4f& operator+=(const Vector4f& v) { x += v.x; y += v.y; z += v.z; w += v.w;  return *this; }
	inline __host__ __device__ Vector4f& operator*=(const Vector4f& v) { x *= v.x; y *= v.y; z *= v.z; w *= v.w;  return *this; }
};


float mod(float x, float y)
{
	return x - y * floorf(x / y);
}

float clamp2(float low, float high, float n)
{
	return fmaxf(fminf(n, high), low);
}

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

#endif