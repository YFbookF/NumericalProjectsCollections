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

	void SetPerspectiveMatrix(float width,float height,float nearPlane,float farPlane,float fieldOfView);
	

private:

};



#endif // ! CAMERA_H
