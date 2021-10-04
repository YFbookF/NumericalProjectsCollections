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
	unsigned int quadVAO, quadVBO;

	MeshMananger();
	~MeshMananger();
	void RenderCube();
	void RenderQuad();
};

#endif // !MESHMANAGER_H

