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

	MeshMananger();
	~MeshMananger();
	void RenderCube();
};

#endif // !MESHMANAGER_H

