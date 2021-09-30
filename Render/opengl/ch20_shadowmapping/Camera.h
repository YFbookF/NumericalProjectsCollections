#ifndef  CAMERA_H
#define CAMERA_H
#include <Eigen/Core>
#include <Eigen/Dense>
using namespace Eigen;
#define M_PI 3.14159

class Camera
{
public:
	Camera();
	Camera(float px,float py,float pz,float lx,float ly,float lz);
	~Camera();
	
	Matrix4f PerspectiveMatrix;
	float Zoom;
	Vector3f Position;
	Vector3f Front;
	Vector3f Up;
	void SetPerspectiveMatrix(float width,float height,float nearPlane,float farPlane,float fieldOfView);
	void SetPerspectiveMatrix2(float left, float right, float top, float bottom, float near, float far);
	
	Matrix4f PerspectiveRH_NO(float fovy, float aspect, float zNear, float zFar);
	Matrix4f lookAt();
	Vector3f Cross(Vector3f A, Vector3f B);
	Matrix4f GetRotateMatrix(float angle, float offset_x, float offset_y, float offset_z);
	Matrix4f GetTranslateMatirx(float offset_x, float offset_y, float offset_z);
	Matrix4f GetScaleMatrix(float offset_x, float offset_y, float offset_z);

private:

};



#endif // ! CAMERA_H
