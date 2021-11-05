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



/*

*/
struct Vertex :public Vector3f
{
	Vector3f _pos;
	Vector3f _normal;
	Vertex(float px, float py, float pz) :
		_pos(Vector3f(px, py, pz))
	{
		_normal = Vector3f(0, 0, 0);
	}
	Vertex(float px, float py, float pz, float nx, float ny, float nz) :
		_pos(Vector3f(px, py, pz)), _normal(Vector3f(nx, ny, nz))
	{

	}
};

struct Triangle
{
	unsigned int _idx1;
	unsigned int _idx2;
	unsigned int _idx3;

	Vector3f _normal;

	Vector3f bb_bottom;
	Vector3f bb_top;
};


#endif