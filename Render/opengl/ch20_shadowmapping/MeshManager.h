#pragma once
#ifndef MESHMANAGER_H
#define MESHMANAGER_H

#include <GL\glew.h>
#include <GL\freeglut.h>

class MeshMananger
{
public:
	unsigned int vertexIdx;
	unsigned int indexIdx;
	unsigned int  cubeVAO, cubeVBO;
	unsigned int planeVAO, planeVBO;

	MeshMananger();
	~MeshMananger();
	void RenderCube();
	void RenderCube(float start_x, float start_y, float start_z, float length_x, float length_y, float length_z);
};

#endif // !MESHMANAGER_H

