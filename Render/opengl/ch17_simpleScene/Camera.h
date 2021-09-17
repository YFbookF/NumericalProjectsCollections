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
	~Camera();
	
	Matrix4f PerspectiveMatrix;
	float Zoom;
	Vector3f Position;
	Vector3f Front;
	Vector3f Up;
	void SetPerspectiveMatrix(float width,float height,float nearPlane,float farPlane,float fieldOfView);
	
	Matrix4f PerspectiveRH_NO(float fovy, float aspect, float zNear, float zFar);
	Matrix4f lookAt();
	Vector3f Cross(Vector3f A, Vector3f B);
	Matrix4f GetRotateMatrix(float angle, float offset_x, float offset_y, float offset_z);
	Matrix4f GetTranslateMatirx(float offset_x, float offset_y, float offset_z);
	Matrix4f GetScaleMatrix(float offset_x, float offset_y, float offset_z);

private:

};



#endif // ! CAMERA_H
