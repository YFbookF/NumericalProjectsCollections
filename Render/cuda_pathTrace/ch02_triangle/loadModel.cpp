#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <cfloat>

#include "linear_algebra.h"
#include "cuda_render.h"
#include "loadModel.h"

unsigned int vertices_num;
unsigned int triangle_num;
Vertex* vertices_pos;
Triangle* triangles;
// read triangle data from obj file
void loadObj(const std::string filename)
{
	vertices_num = 8;
	triangle_num = 12;

	vertices_pos = (Vertex*)malloc(vertices_num * sizeof(Vertex));
	triangles = (Triangle*)malloc(triangle_num * sizeof(Triangle));
	std::ifstream in(filename.c_str());

	if (!in.good())
	{
		std::cout << "ERROR: loading obj:(" << filename << ") file not found or not good" << "\n";
		system("PAUSE");
		exit(0);
	}

	char buffer[256], str[255];
	float f1, f2, f3;

	int lineno = 0;
	int vertex_read_idx = 0;
	int triangle_read_idx = 0;

	while (!in.getline(buffer, 255).eof())
	{

		buffer[255] = '\0';
		sscanf_s(buffer, "%s", str, 255);

		// reading a vertex
		if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32)) {
			if (sscanf(buffer, "v %f %f %f", &f1, &f2, &f3) == 3) {
				vertices_pos[vertex_read_idx].x = f1;
				vertices_pos[vertex_read_idx].y = f2;
				vertices_pos[vertex_read_idx].z = f3;
				
				vertex_read_idx += 1;
			}
			else {
				std::cout << "ERROR: vertex not in wanted format in OBJLoader" << "\n";
				exit(-1);
			}
		}

		// reading faceMtls 
		else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))
		{

			int junk0, junk1, junk2, junk3, junk4, junk5;
			int f4, f5, f6;
			int nt = sscanf(buffer, "f %d/%d/%d %d/%d/%d %d/%d/%d", &f4, &junk0, &junk1,
				&f5, &junk2, &junk3, &f6, &junk4, &junk5);
			if (nt != 9) {
				std::cout << "ERROR: I don't know the format of that FaceMtl" << "\n";
				exit(-1);
			}
			triangles[triangle_read_idx]._idx1 = f4;
			triangles[triangle_read_idx]._idx2 = f5;
			triangles[triangle_read_idx]._idx3 = f6;
			printf("%d %d %d \n", triangles[triangle_read_idx]._idx1, triangles[triangle_read_idx]._idx2, triangles[triangle_read_idx]._idx3);
			triangle_read_idx += 1;
		}
		lineno += 1;
	}
	/*
	// calculate the bounding box of the mesh
	mesh.bounding_box[0] = make_float3(1000000, 1000000, 1000000);
	mesh.bounding_box[1] = make_float3(-1000000, -1000000, -1000000);
	for (unsigned int i = 0; i < mesh.verts.size(); i++)
	{
		//update min and max value
		mesh.bounding_box[0] = fminf(mesh.verts[i], mesh.bounding_box[0]);
		mesh.bounding_box[1] = fmaxf(mesh.verts[i], mesh.bounding_box[1]);
	}

	std::cout << "obj file loaded: number of faces:" << mesh.faces.size() << " number of vertices:" << mesh.verts.size() << std::endl;
	std::cout << "obj bounding box: min:(" << mesh.bounding_box[0].x << "," << mesh.bounding_box[0].y << "," << mesh.bounding_box[0].z << ") max:"
		<< mesh.bounding_box[1].x << "," << mesh.bounding_box[1].y << "," << mesh.bounding_box[1].z << ")" << std::endl;
		*/
}
// helpers to load triangle data

inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
// read triangle data from obj file
void loadObj(const std::string filename, TriangleMesh& mesh)
{
	std::ifstream in(filename.c_str());

	if (!in.good())
	{
		std::cout << "ERROR: loading obj:(" << filename << ") file not found or not good" << "\n";
		system("PAUSE");
		exit(0);
	}

	char buffer[256], str[255];
	float f1, f2, f3;

	while (!in.getline(buffer, 255).eof())
	{
		buffer[255] = '\0';
		sscanf_s(buffer, "%s", str, 255);

		// reading a vertex
		if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32)) {
			if (sscanf(buffer, "v %f %f %f", &f1, &f2, &f3) == 3) {
				mesh.verts.push_back(make_float3(f1, f2, f3));
			}
			else {
				std::cout << "ERROR: vertex not in wanted format in OBJLoader" << "\n";
				exit(-1);
			}
		}

		// reading faceMtls 
		else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))
		{
			TriangleFace f;
			int junk0, junk1, junk2, junk3, junk4, junk5;
			int nt = sscanf(buffer, "f %d/%d/%d %d/%d/%d %d/%d/%d", &f.v[0], &junk0, &junk1,
				&f.v[1], &junk2, &junk3, &f.v[2], &junk4, &junk5);
			if (nt != 9) {
				std::cout << "ERROR: I don't know the format of that FaceMtl" << "\n";
				exit(-1);
			}

			mesh.faces.push_back(f);
		}
	}

	// calculate the bounding box of the mesh
	mesh.bounding_box[0] = make_float3(1000000, 1000000, 1000000);
	mesh.bounding_box[1] = make_float3(-1000000, -1000000, -1000000);
	for (unsigned int i = 0; i < mesh.verts.size(); i++)
	{
		//update min and max value
		mesh.bounding_box[0] = fminf(mesh.verts[i], mesh.bounding_box[0]);
		mesh.bounding_box[1] = fmaxf(mesh.verts[i], mesh.bounding_box[1]);
	}

	std::cout << "obj file loaded: number of faces:" << mesh.faces.size() << " number of vertices:" << mesh.verts.size() << std::endl;
	std::cout << "obj bounding box: min:(" << mesh.bounding_box[0].x << "," << mesh.bounding_box[0].y << "," << mesh.bounding_box[0].z << ") max:"
		<< mesh.bounding_box[1].x << "," << mesh.bounding_box[1].y << "," << mesh.bounding_box[1].z << ")" << std::endl;
}