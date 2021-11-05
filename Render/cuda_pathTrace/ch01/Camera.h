#pragma once

#include "linear_algebra.h"

#define M_PI 3.14159265
#define PI_DIV_TWO = 1.570796

struct Camera
{
	Vector2f resoultion;
	Vector3f position;
	Vector3f view;
	Vector3f up;
	Vector2f fov;

};

class OriCamera
{
private:
	float yaw;
	float pitch;
	float roll;

public:
	void ChangeYaw(float m)
	{
		yaw = mod(yaw + m, 2 * M_PI);
	}
	void ChangePitch(float m)
	{
		pitch = mod(pitch + m, 2 * M_PI);
	}
	void ChangeRoll(float m)
	{
		roll = mod(roll + m, 2 * M_PI);
	}
};