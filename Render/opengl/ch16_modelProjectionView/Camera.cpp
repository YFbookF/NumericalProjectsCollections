#include "Camera.h"

Camera::Camera()
{
	Position = Vector3f(0.0f, 5.0f,-5.0f);
	Front = Vector3f(0.0f, -0.1f, 1.0f);
	Up = Vector3f(0.0f, 1.0f, 0.0f);
	Zoom = 45.0f;
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

	PerspectiveMatrix(2, 2) = (-farPlane - nearPlane) / (farPlane - nearPlane);
	PerspectiveMatrix(2, 3) = 2 * farPlane * nearPlane / (farPlane - nearPlane);

	PerspectiveMatrix(3, 0) = 0.0f;
	PerspectiveMatrix(3, 1) = 0.0f;
	PerspectiveMatrix(3, 2) = 1.0f;
	PerspectiveMatrix(3, 3) = 0.0f;

}

Matrix4f Camera::PerspectiveRH_NO(float fovy, float aspect, float zNear, float zFar)
{
	float tanHalfFovy = tan(fovy / 2);

	Matrix4f result = Matrix4f::Zero();
	result(0, 0) = 1 / (aspect * tanHalfFovy);
	result(1, 1) = 1 / tanHalfFovy;
	result(2, 2) = zFar / (zFar - zNear);
	result(2, 3) = -zFar * zNear / (zFar - zNear);

	//result(2, 2) = (-zFar - zNear) / (zFar - zNear);
	//result(2, 3) =2 *zFar * zNear / (zFar - zNear);

	result(3, 2) = 1;
	

	return result;
}

//https://learnopengl.com/Getting-started/Camera
Matrix4f Camera::lookAt()
{
	
	Vector3f center = Position + Front;
	Vector3f f((center - Position).normalized()); // 向前向量
	Vector3f s = Cross(f, Up).normalized(); // 右向量
	Vector3f u = Cross(s, f); // 上向量
	Matrix4f result = Matrix4f::Identity();
	result(0, 0) = s(0);
	result(0, 1) = s(1);
	result(0, 2) = s(2);
	result(1, 0) = u(0);
	result(1, 1) = u(1);
	result(1, 2) = u(2);
	result(2, 0) = f(0);
	result(2, 1) = f(1);
	result(2, 2) = f(2);
	result(0, 3) = -s.dot(Position);
	result(1, 3) = -u.dot(Position);
	result(2, 3) = -f.dot(Position);
	return result;
}

Vector3f Camera::Cross(Vector3f A, Vector3f B)
{
	return Vector3f(A(1)*B(2) - A(2)*B(1),
		A(2)*B(0) - A(0)*B(2),
		A(0)*B(1) - A(1)*B(0));
}

//https://learnopengl.com/Getting-started/Transformations
Matrix4f Camera::GetRotateMatrix(float angle, float offset_x, float offset_y, float offset_z)
{
	angle = angle / 180.0f * M_PI;
	float c = cos(angle);
	float s = sin(angle);
	float c_1 = 1 - c;
	Vector3f axis(Vector3f(offset_x,offset_y,offset_z).normalized());
	Vector3f temp((1 - c) * axis);
	Matrix4f rotate = Matrix4f::Zero();
	rotate(0, 0) = c +  axis(0) * axis(0) * c_1;
	rotate(0, 1) = axis(0) * axis(1) * c_1 - axis(2) * s; 
	rotate(0, 2) = axis(0) * axis(2) * c_1 + axis(1) * s;

	rotate(1, 0) = axis(1) * axis(0) * c_1 + axis(2) * s;
	rotate(1, 1) = c + axis(1) * axis(1) * c_1;
	rotate(1, 2) = axis(1) * axis(2) * c_1 - axis(0) * s;

	rotate(2, 0) = axis(2) * axis(0) * c_1 - axis(1) * s;
	rotate(2, 1) = axis(2) * axis(1) * c_1 + axis(0) * s;
	rotate(2, 2) = c + axis(2) * axis(2) * c_1;

	rotate(3, 3) = 1;

	return rotate;
}

Matrix4f Camera::GetTranslateMatirx(float offset_x,float offset_y,float offset_z)
{
	Matrix4f translate = Matrix4f::Identity();
	translate(0, 3) = offset_x;
	translate(1, 3) = offset_y;
	translate(2, 3) = offset_z;
	return translate;
}

Matrix4f Camera::GetScaleMatrix(float offset_x, float offset_y, float offset_z)
{
	Matrix4f translate = Matrix4f::Identity();
	translate(0, 0) = offset_x;
	translate(1, 1) = offset_y;
	translate(2, 2) = offset_z;
	return translate;
}