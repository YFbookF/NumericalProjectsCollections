/*************************************************************************
> File Name: Matrix.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Matrix type compatibles with CUDA.
> Created Time: 2018/01/01
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_MATRIX_H
#define SNOW_SIMULATION_MATRIX_H

#include <Common/Math.h>
#include <CUDA/Vector.h>
#include <CUDA/Quaternion.h>

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/mat3x3.hpp>
#include <glm/gtc/type_ptr.hpp>

struct Matrix3
{
	float data[9];

	__host__ __device__ __forceinline__
	Matrix3(float f = 1.f)
	{
		data[0] = f; data[3] = 0; data[6] = 0;
		data[1] = 0; data[4] = f; data[7] = 0;
		data[2] = 0; data[5] = 0; data[8] = f;
	}

	__host__ __device__ __forceinline__
	Matrix3(float a, float b, float c, float d, float e, float f, float g, float h, float i)
	{
		data[0] = a; data[3] = d; data[6] = g;
		data[1] = b; data[4] = e; data[7] = h;
		data[2] = c; data[5] = f; data[8] = i;
	}

	__host__ __device__ __forceinline__
	Matrix3(const glm::mat3& matrix)
	{
		const float* m = glm::value_ptr(matrix);
		data[0] = m[0]; data[3] = m[3]; data[6] = m[6];
		data[1] = m[1]; data[4] = m[4]; data[7] = m[7];
		data[2] = m[2]; data[5] = m[5]; data[8] = m[8];
	}

	__host__ __device__ __forceinline__
	Matrix3(const Vector3& c0, const Vector3& c1, const Vector3& c2)
	{
		data[0] = c0.x; data[3] = c1.x; data[6] = c2.x;
		data[1] = c0.y; data[4] = c1.y; data[7] = c2.y;
		data[2] = c0.z; data[5] = c1.z; data[8] = c2.z;
	}

	__host__ __device__  __forceinline__
	Matrix3& operator=(const Matrix3& rhs)
	{
		data[0] = rhs[0]; data[3] = rhs[3]; data[6] = rhs[6];
		data[1] = rhs[1]; data[4] = rhs[4]; data[7] = rhs[7];
		data[2] = rhs[2]; data[5] = rhs[5]; data[8] = rhs[8];
		
		return *this;
	}

	__host__ __device__ __forceinline__
	Matrix3& operator=(const glm::mat3& rhs)
	{
		const float* r = glm::value_ptr(rhs);
		data[0] = r[0]; data[3] = r[3]; data[6] = r[6];
		data[1] = r[1]; data[4] = r[4]; data[7] = r[7];
		data[2] = r[2]; data[5] = r[5]; data[8] = r[8];
		
		return *this;
	}

	__host__ __device__
	static bool IsEqual(const Matrix3& A, const Matrix3& B)
	{
		for (size_t i = 0; i < 9; ++i)
		{
			if (IsNotEqual(A[i], B[i]))
			{
				return false;
			}
		}

		return true;
	}

	__host__ __device__ __forceinline__
	glm::mat3 ToGLM() const
	{
		return glm::mat3(
			data[0], data[1], data[2],
			data[3], data[4], data[5],
			data[6], data[7], data[8]);
	}

	__host__ __device__ __forceinline__
		static Matrix3 OuterProduct(const Vector3& v, const Vector3& w)
	{
		return Matrix3(
			v.x * w.x, v.y * w.x, v.z * w.x,
			v.x * w.y, v.y * w.y, v.z * w.y,
			v.x * w.z, v.y * w.z, v.z * w.z);
	}

	__host__ __device__ __forceinline__
	float& operator[](int i)
	{
		return data[i];
	}

	__host__ __device__ __forceinline__
	float operator[](int i) const
	{
		return data[i];
	}

	__host__ __device__ __forceinline__
	Vector3 GetRow(int i) const
	{
		return Vector3(data[i], data[i + 3], data[i + 6]);
	}

	__host__ __device__ __forceinline__
	Vector3 GetColumn(int i) const
	{
		return Vector3(data[3 * i], data[3 * i + 1], data[3 * i + 2]);
	}

	__host__ __device__ __forceinline__
	Matrix3& operator*=(const Matrix3& rhs)
	{
		Matrix3 tmp;
		tmp[0] = data[0] * rhs[0] + data[3] * rhs[1] + data[6] * rhs[2];
		tmp[1] = data[1] * rhs[0] + data[4] * rhs[1] + data[7] * rhs[2];
		tmp[2] = data[2] * rhs[0] + data[5] * rhs[1] + data[8] * rhs[2];
		tmp[3] = data[0] * rhs[3] + data[3] * rhs[4] + data[6] * rhs[5];
		tmp[4] = data[1] * rhs[3] + data[4] * rhs[4] + data[7] * rhs[5];
		tmp[5] = data[2] * rhs[3] + data[5] * rhs[4] + data[8] * rhs[5];
		tmp[6] = data[0] * rhs[6] + data[3] * rhs[7] + data[6] * rhs[8];
		tmp[7] = data[1] * rhs[6] + data[4] * rhs[7] + data[7] * rhs[8];
		tmp[8] = data[2] * rhs[6] + data[5] * rhs[7] + data[8] * rhs[8];
		
		return (*this = tmp);
	}

	__host__ __device__ __forceinline__
	Matrix3 operator*(const Matrix3& rhs) const
	{
		Matrix3 result;
		result[0] = data[0] * rhs[0] + data[3] * rhs[1] + data[6] * rhs[2];
		result[1] = data[1] * rhs[0] + data[4] * rhs[1] + data[7] * rhs[2];
		result[2] = data[2] * rhs[0] + data[5] * rhs[1] + data[8] * rhs[2];
		result[3] = data[0] * rhs[3] + data[3] * rhs[4] + data[6] * rhs[5];
		result[4] = data[1] * rhs[3] + data[4] * rhs[4] + data[7] * rhs[5];
		result[5] = data[2] * rhs[3] + data[5] * rhs[4] + data[8] * rhs[5];
		result[6] = data[0] * rhs[6] + data[3] * rhs[7] + data[6] * rhs[8];
		result[7] = data[1] * rhs[6] + data[4] * rhs[7] + data[7] * rhs[8];
		result[8] = data[2] * rhs[6] + data[5] * rhs[7] + data[8] * rhs[8];
		
		return result;
	}

	__host__ __device__ __forceinline__
	Vector3 operator*(const Vector3& rhs) const
	{
		Vector3 result;
		result.x = data[0] * rhs.x + data[3] * rhs.y + data[6] * rhs.z;
		result.y = data[1] * rhs.x + data[4] * rhs.y + data[7] * rhs.z;
		result.z = data[2] * rhs.x + data[5] * rhs.y + data[8] * rhs.z;
		
		return result;
	}

	__host__ __device__ __forceinline__
	Matrix3& operator+=(const Matrix3& rhs)
	{
		data[0] += rhs[0]; data[3] += rhs[3]; data[6] += rhs[6];
		data[1] += rhs[1]; data[4] += rhs[4]; data[7] += rhs[7];
		data[2] += rhs[2]; data[5] += rhs[5]; data[8] += rhs[8];

		return *this;
	}

	__host__ __device__ __forceinline__
	Matrix3 operator+(const Matrix3& rhs) const
	{
		Matrix3 tmp = *this;
		tmp[0] += rhs[0]; tmp[3] += rhs[3]; tmp[6] += rhs[6];
		tmp[1] += rhs[1]; tmp[4] += rhs[4]; tmp[7] += rhs[7];
		tmp[2] += rhs[2]; tmp[5] += rhs[5]; tmp[8] += rhs[8];
		return tmp;
	}

	__host__ __device__ __forceinline__
	Matrix3& operator-=(const Matrix3& rhs)
	{
		data[0] -= rhs[0]; data[3] -= rhs[3]; data[6] -= rhs[6];
		data[1] -= rhs[1]; data[4] -= rhs[4]; data[7] -= rhs[7];
		data[2] -= rhs[2]; data[5] -= rhs[5]; data[8] -= rhs[8];
		
		return *this;
	}

	__host__ __device__ __forceinline__
	Matrix3 operator-(const Matrix3& rhs) const
	{
		Matrix3 tmp = *this;
		tmp[0] -= rhs[0]; tmp[3] -= rhs[3]; tmp[6] -= rhs[6];
		tmp[1] -= rhs[1]; tmp[4] -= rhs[4]; tmp[7] -= rhs[7];
		tmp[2] -= rhs[2]; tmp[5] -= rhs[5]; tmp[8] -= rhs[8];

		return tmp;
	}

	__host__ __device__ __forceinline__
	Matrix3& operator*=(float f)
	{
		data[0] *= f; data[3] *= f; data[6] *= f;
		data[1] *= f; data[4] *= f; data[7] *= f;
		data[2] *= f; data[5] *= f; data[8] *= f;
		
		return *this;
	}

	__host__ __device__ __forceinline__
	Matrix3 operator*(float f) const
	{
		Matrix3 tmp = *this;
		tmp[0] *= f; tmp[3] *= f; tmp[6] *= f;
		tmp[1] *= f; tmp[4] *= f; tmp[7] *= f;
		tmp[2] *= f; tmp[5] *= f; tmp[8] *= f;

		return tmp;
	}

	__host__ __device__ __forceinline__
	Matrix3& operator/=(float f)
	{
		float fi = 1.f / f;
		data[0] *= fi; data[3] *= fi; data[6] *= fi;
		data[1] *= fi; data[4] *= fi; data[7] *= fi;
		data[2] *= fi; data[5] *= fi; data[8] *= fi;

		return *this;
	}

	__host__ __device__ __forceinline__
	Matrix3 operator/(float f) const
	{
		Matrix3 tmp = *this;
		tmp[0] *= 1.f / f; tmp[3] *= 1.f / f; tmp[6] *= 1.f / f;
		tmp[1] *= 1.f / f; tmp[4] *= 1.f / f; tmp[7] *= 1.f / f;
		tmp[2] *= 1.f / f; tmp[5] *= 1.f / f; tmp[8] *= 1.f / f;

		return tmp;
	}

	__host__ __device__ __forceinline__
	static Matrix3 Transpose(const Matrix3& m)
	{
		return Matrix3(
			m[0], m[3], m[6],
			m[1], m[4], m[7],
			m[2], m[5], m[8]);
	}

	// Optimize I + A
	__host__ __device__ __forceinline__
	static Matrix3 AddIdentity(const Matrix3& A)
	{
		return Matrix3(
			A[0] + 1.f, A[1], A[2],
			A[3], A[4] + 1.f, A[5],
			A[6], A[7], A[8] + 1.f);
	}

	__host__ __device__ __forceinline__
	static Matrix3 Emult(const Matrix3& A, const Matrix3& B)
	{
		Matrix3 tmp;
		tmp[0] = A[0] * B[0];
		tmp[1] = A[1] * B[1];
		tmp[2] = A[2] * B[2];
		tmp[3] = A[3] * B[3];
		tmp[4] = A[4] * B[4];
		tmp[5] = A[5] * B[5];
		tmp[6] = A[6] * B[6];
		tmp[7] = A[7] * B[7];
		tmp[8] = A[8] * B[8];
		
		return tmp;
	}

	__host__ __device__ __forceinline__
	static float InnerProduct(const Matrix3& A, const Matrix3& B)
	{
		return
			A[0] * B[0] + A[1] * B[1] + A[2] * B[2] +
			A[3] * B[3] + A[4] * B[4] + A[5] * B[5] +
			A[6] * B[6] + A[7] * B[7] + A[8] * B[8];
	}

	// Optimize transpose(A) * B;
	__host__ __device__ __forceinline__
	static Matrix3 MultiplyATB(const Matrix3& A, const Matrix3& B)
	{
		Matrix3 tmp;
		tmp[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
		tmp[1] = A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
		tmp[2] = A[6] * B[0] + A[7] * B[1] + A[8] * B[2];
		tmp[3] = A[0] * B[3] + A[1] * B[4] + A[2] * B[5];
		tmp[4] = A[3] * B[3] + A[4] * B[4] + A[5] * B[5];
		tmp[5] = A[6] * B[3] + A[7] * B[4] + A[8] * B[5];
		tmp[6] = A[0] * B[6] + A[1] * B[7] + A[2] * B[8];
		tmp[7] = A[3] * B[6] + A[4] * B[7] + A[5] * B[8];
		tmp[8] = A[6] * B[6] + A[7] * B[7] + A[8] * B[8];
		
		return tmp;
	}

	// Optimize A * transpose(B);
	__host__ __device__ __forceinline__
	static Matrix3 MultiplyABT(const Matrix3& A, const Matrix3& B)
	{
		Matrix3 tmp;
		tmp[0] = A[0] * B[0] + A[3] * B[3] + A[6] * B[6];
		tmp[1] = A[1] * B[0] + A[4] * B[3] + A[7] * B[6];
		tmp[2] = A[2] * B[0] + A[5] * B[3] + A[8] * B[6];
		tmp[3] = A[0] * B[1] + A[3] * B[4] + A[6] * B[7];
		tmp[4] = A[1] * B[1] + A[4] * B[4] + A[7] * B[7];
		tmp[5] = A[2] * B[1] + A[5] * B[4] + A[8] * B[7];
		tmp[6] = A[0] * B[2] + A[3] * B[5] + A[6] * B[8];
		tmp[7] = A[1] * B[2] + A[4] * B[5] + A[7] * B[8];
		tmp[8] = A[2] * B[2] + A[5] * B[5] + A[8] * B[8];

		return tmp;
	}

	// Optimize A * D * transpose(B), where D is diagonal
	__host__ __device__ __forceinline__
	static Matrix3 MultiplyADBT(const Matrix3& A, const Matrix3& D, const Matrix3& B)
	{
		Matrix3 tmp;
		tmp[0] = A[0] * B[0] * D[0] + A[3] * B[3] * D[4] + A[6] * B[6] * D[8];
		tmp[1] = A[1] * B[0] * D[0] + A[4] * B[3] * D[4] + A[7] * B[6] * D[8];
		tmp[2] = A[2] * B[0] * D[0] + A[5] * B[3] * D[4] + A[8] * B[6] * D[8];
		tmp[3] = A[0] * B[1] * D[0] + A[3] * B[4] * D[4] + A[6] * B[7] * D[8];
		tmp[4] = A[1] * B[1] * D[0] + A[4] * B[4] * D[4] + A[7] * B[7] * D[8];
		tmp[5] = A[2] * B[1] * D[0] + A[5] * B[4] * D[4] + A[8] * B[7] * D[8];
		tmp[6] = A[0] * B[2] * D[0] + A[3] * B[5] * D[4] + A[6] * B[8] * D[8];
		tmp[7] = A[1] * B[2] * D[0] + A[4] * B[5] * D[4] + A[7] * B[8] * D[8];
		tmp[8] = A[2] * B[2] * D[0] + A[5] * B[5] * D[4] + A[8] * B[8] * D[8];

		return tmp;
	}

	__host__ __device__ __forceinline__
	static float Determinant(const Matrix3& M)
	{
		return
			M[0] * (M[4] * M[8] - M[7] * M[5]) -
			M[3] * (M[1] * M[8] - M[7] * M[2]) +
			M[6] * (M[1] * M[5] - M[4] * M[2]);
	}

	__host__ __device__ __forceinline__
	static Matrix3 FromQuaternion(const Quaternion& q)
	{
		float qxx = q.x * q.x;
		float qyy = q.y * q.y;
		float qzz = q.z * q.z;
		float qxz = q.x * q.z;
		float qxy = q.x * q.y;
		float qyz = q.y * q.z;
		float qwx = q.w * q.x;
		float qwy = q.w * q.y;
		float qwz = q.w * q.z;
		Matrix3 M;

		M[0] = 1.f - 2.f * (qyy + qzz);
		M[1] = 2.f * (qxy + qwz);
		M[2] = 2.f * (qxz - qwy);
		M[3] = 2.f * (qxy - qwz);
		M[4] = 1.f - 2.f * (qxx + qzz);
		M[5] = 2.f * (qyz + qwx);
		M[6] = 2.f * (qxz + qwy);
		M[7] = 2.f * (qyz - qwx);
		M[8] = 1.f - 2.f * (qxx + qyy);

		return M;
	}

	__host__ __device__ __forceinline__
	static Matrix3 Inverse(const Matrix3& M)
	{
		float invDet = 1.f /
			(M[0] * (M[4] * M[8] - M[7] * M[5]) -
			M[3] * (M[1] * M[8] - M[7] * M[2]) +
			M[6] * (M[1] * M[5] - M[4] * M[2]));
		
		Matrix3 A;
		A[0] = invDet * (M[4] * M[8] - M[5] * M[7]);
		A[1] = invDet * (M[2] * M[7] - M[1] * M[8]);
		A[2] = invDet * (M[1] * M[5] - M[2] * M[4]);
		A[3] = invDet * (M[5] * M[6] - M[3] * M[8]);
		A[4] = invDet * (M[0] * M[8] - M[2] * M[6]);
		A[5] = invDet * (M[2] * M[3] - M[0] * M[5]);
		A[6] = invDet * (M[3] * M[7] - M[4] * M[6]);
		A[7] = invDet * (M[1] * M[6] - M[0] * M[7]);
		A[8] = invDet * (M[0] * M[4] - M[1] * M[3]);
		
		return A;
	}

	// Note: adjugate = transpose(cofactor)
	__host__ __device__ __forceinline__
		static Matrix3 Adjugate(const Matrix3& M)
	{
		Matrix3 A;
		A[0] = (M[4] * M[8] - M[5] * M[7]); A[3] = (M[5] * M[6] - M[3] * M[8]); A[6] = (M[3] * M[7] - M[4] * M[6]);
		A[1] = (M[2] * M[7] - M[1] * M[8]); A[4] = (M[0] * M[8] - M[2] * M[6]); A[7] = (M[1] * M[6] - M[0] * M[7]);
		A[2] = (M[1] * M[5] - M[2] * M[4]); A[5] = (M[2] * M[3] - M[0] * M[5]); A[8] = (M[0] * M[4] - M[1] * M[3]);

		return A;
	}

	__host__ __device__ __forceinline__
	static Matrix3 Cofactor(const Matrix3& M)
	{
		Matrix3 A;
		A[0] = (M[4] * M[8] - M[5] * M[7]); A[3] = (M[2] * M[7] - M[1] * M[8]); A[6] = (M[1] * M[5] - M[2] * M[4]);
		A[1] = (M[5] * M[6] - M[3] * M[8]); A[4] = (M[0] * M[8] - M[2] * M[6]); A[7] = (M[2] * M[3] - M[0] * M[5]);
		A[2] = (M[3] * M[7] - M[4] * M[6]); A[5] = (M[1] * M[6] - M[0] * M[7]); A[8] = (M[0] * M[4] - M[1] * M[3]);

		return A;
	}

	// Should be written with a more robust solver, but this will do for now
	__host__ __device__ __forceinline__
	static Vector3 Solve(const Matrix3& A, const Vector3& b)
	{
		return Matrix3::Inverse(A) * b;
	}

	__host__ __device__ __forceinline__
	static void Print(const Matrix3& m)
	{
		printf("\n%10f %10f %10f\n%10f %10f %10f\n%10f %10f %10f\n",
			m[0], m[3], m[6],
			m[1], m[4], m[7],
			m[2], m[5], m[8]);
	}
};

__host__ __device__ __forceinline__
Matrix3 operator*(float f, const Matrix3& m)
{
	return m * f;
}

#endif