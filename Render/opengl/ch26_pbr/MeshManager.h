#pragma once
#ifndef MESHMANAGER_H
#define MESHMANAGER_H

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <Eigen\Core>
#include <vector>

class MeshMananger
{
public:
	unsigned int vertexIdx;
	unsigned int indexIdx, indexCount;
	unsigned int  cubeVAO, cubeVBO;
	unsigned int planeVAO, planeVBO;
	unsigned int sphereVAO, sphereVBO;

	MeshMananger();
	~MeshMananger();
	void RenderCube();
	void RenderSphere();
};

#endif // !MESHMANAGER_H

