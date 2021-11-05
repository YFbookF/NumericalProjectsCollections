#pragma once
#ifndef __LOADMODEL_H_
#define __LOADMODEL_H_



struct TriangleFace
{
	int v[3]; // vertex indices
};

struct TriangleMesh
{
	std::vector<float3> verts;
	std::vector<TriangleFace> faces;
	float3 bounding_box[2];
};

void loadObj(const std::string filename);
void loadObj(const std::string filename, TriangleMesh& mesh);

#endif // !__LOADMODEL_H_


