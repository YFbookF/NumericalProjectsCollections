#include "Camera.h"

Camera::Camera()
{
}



Camera::~Camera()
{
}

void Camera::SetPerspectiveMatrix(float width, float height, float nearPlane, float farPlane, float fieldOfView)
{
	
	float tanHalfFOV = tanf(fieldOfView / 2.0f * M_PI / 180.0f);
	nearPlane = 1.0f / tanHalfFOV;
	float areaRatio = width / height;
	PerspectiveMatrix(0, 0) = 1.0f / (tanHalfFOV * areaRatio);
	PerspectiveMatrix(0, 1) = 0.0f;
	PerspectiveMatrix(0, 2) = 0.0f;
	PerspectiveMatrix(0, 3) = 0.0f;

	PerspectiveMatrix(1, 0) = 0.0f;
	PerspectiveMatrix(1, 1) = 1.0f  / tanHalfFOV;
	PerspectiveMatrix(1, 2) = 0.0f;
	PerspectiveMatrix(1, 3) = 0.0f;
	/*
	https://computergraphics.stackexchange.com/questions/6254/how-to-derive-a-perspective-projection-matrix-from-its-components
	f(z) = Az + B
	f(z) = A + B/z

	A + B/near = -1
	A + B/far = 1
	A = (-far - near)/(far - near)
	B = (2 * far * near) / (far - near)

	*/
	PerspectiveMatrix(2, 0) = 0.0f;
	PerspectiveMatrix(2, 1) = 0.0f;
	PerspectiveMatrix(2, 2) = farPlane / (farPlane - nearPlane);
	PerspectiveMatrix(2, 3) = -  farPlane * nearPlane / (farPlane - nearPlane);

	PerspectiveMatrix(3, 0) = 0.0f;
	PerspectiveMatrix(3, 1) = 0.0f;
	PerspectiveMatrix(3, 2) = 1.0f;
	PerspectiveMatrix(3, 3) = 0.0f;

}
