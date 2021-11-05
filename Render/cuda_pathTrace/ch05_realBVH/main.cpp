#include <GL/glew.h>
#include <GL/freeglut.h>
//非常简单的BVH，就是根据索引划分，可以算得上是负优化
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#include "cuda_render.h"
#include "loadModel.h"
#include "linear_algebra.h"
#include "bvh.h"

#include <stdlib.h>
#include <stdio.h>

extern unsigned int vertices_num;
extern Vertex* scene_vertices_pos;
extern unsigned int triangle_num;
extern Triangle* scene_triangles;

Vertex* cudaVertices2 = NULL;
Triangle* cudaTriangles2 = NULL;
float* cudaSlabLimit2 = NULL;// 有6 * node_num个大小，因为每个node_num都有一个aabb
int* cudaTreeInfo2 = NULL;

void Timer(int obsolete) {

	glutPostRedisplay();
	glutTimerFunc(30, Timer, 0);
}

// hash function to calculate new seed for each frame
// see http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
unsigned int WangHash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

int frames = 0;
std::vector<float4> cuda_scene_triangles;
int BVHNode_num;
// 1. load triangle mesh data from obj files
// 2. copy data to CPU memory (into vector<float4> scene_triangles)
// 3. copy to CUDA global memory (allocated with dev_triangle_p pointer)
// 4. copy to CUDA texture memory with bindscene_triangles()
void initCUDAmemoryTriMesh()
{

	loadObj("cube.obj");

	CreateBVH();
	//loadObj("cube.obj", mesh1);
	// scalefactor and offset to position/scale triangle meshes
	float scalefactor1 = 10;
	float scalefactor2 = 10;  // 300
	Vector3f offset1 = Vector3f(90, 22, 100);// (30, -2, 80);
	float3 offset2 = make_float3(30, 12, 80);

	

	for (unsigned int i = 0; i < 12; i++)
	{
		int idx1 = scene_triangles[i]._idx1;
		int idx2 = scene_triangles[i]._idx2;
		int idx3 = scene_triangles[i]._idx3;
		// make a local copy of the triangle vertices
		Vector3f v0 = scene_vertices_pos[idx1 - 1];
		Vector3f v1 = scene_vertices_pos[idx2 - 1];
		Vector3f v2 = scene_vertices_pos[idx3 - 1];

		// scale
		v0 *= scalefactor1;
		v1 *= scalefactor1;
		v2 *= scalefactor1;

		// translate
		v0 += offset1;
		v1 += offset1;
		v2 += offset1;

		// store triangle data as float4
		// store two edges per triangle instead of vertices, to save some calculations in the
		// ray triangle intersection test
		cuda_scene_triangles.push_back(make_float4(v0.x, v0.y, v0.z, 0));
		cuda_scene_triangles.push_back(make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, 0));
		cuda_scene_triangles.push_back(make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, 0));
	}

	BVHNode_num = BVHCahce->box_count;
	// slab limits 中主要存每个bvh节点的包围盒的大小
	float* slab_limits = (float*)malloc(BVHNode_num * 6 * sizeof(float));
	for (unsigned int i = 0; i < BVHNode_num; i++)
	{
		slab_limits[6 * i + 0] = BVHCahce[i].bb_min.x;
		slab_limits[6 * i + 1] = BVHCahce[i].bb_max.x;
		slab_limits[6 * i + 2] = BVHCahce[i].bb_min.y;
		slab_limits[6 * i + 3] = BVHCahce[i].bb_max.y;
		slab_limits[6 * i + 4] = BVHCahce[i].bb_min.z;
		slab_limits[6 * i + 5] = BVHCahce[i].bb_max.z;
	}
	cudaMalloc((void**)&cudaSlabLimit2, BVHNode_num * 6 * sizeof(float));
	cudaMemcpy(cudaSlabLimit2, slab_limits, BVHNode_num * 6 * sizeof(float), cudaMemcpyHostToDevice);

	// tree info 中主要存 bvh树的信息
	int* tree_info = (int*)malloc(BVHNode_num * 4 * sizeof(int));
	for (unsigned int i = 0; i < BVHNode_num; i++)
	{
		tree_info[4 * i + 0] = BVHCahce[i].isLeaf ? 1 : 0;
		tree_info[4 * i + 1] = BVHCahce[i]._left;
		tree_info[4 * i + 2] = BVHCahce[i]._right;
		tree_info[4 * i + 3] = BVHCahce[i].tri_idx;
		printf_s("oo %d %d %d %d\n", tree_info[4 * i + 0], tree_info[4 * i + 1], tree_info[4 * i + 2], tree_info[4 * i + 3]);
	}
	
	cudaMalloc((void**)&cudaTreeInfo2, BVHNode_num * 4 * sizeof(float));
	cudaMemcpy(cudaTreeInfo2, tree_info, BVHNode_num * 4 * sizeof(float), cudaMemcpyHostToDevice);
}

void disp(void)
{
	frames++;
	printf_s("frame = %d\n", frames);
	if (frames == 20)
	{
		//{ cudaMemset(accumulatebuffer, 1, width * height * sizeof(float3)); frames = 0; }
	}
	cudaThreadSynchronize();

	// map vertex buffer object for acces by CUDA 
	cudaGLMapBufferObject((void**)&dptr, vbo);

	//clear all pixels:
	glClear(GL_COLOR_BUFFER_BIT);

	// RAY TRACING:
	// dim3 grid(WINDOW / block.x, WINDOW / block.y, 1);
	// dim3 CUDA specific syntax, block and grid are required to schedule CUDA threads over streaming multiprocessors

	pre_render_kernel(dptr, accumulatebuffer, total_number_of_scene_triangles, frames, WangHash(frames), scene_aabbox_max, scene_aabbox_min, cuda_scene_triangles,cudaSlabLimit2,cudaTreeInfo2, BVHNode_num);
	

	cudaThreadSynchronize();

	// unmap buffer
	cudaGLUnmapBufferObject(vbo);
	//glFlush();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, width * height);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
	glutPostRedisplay();
}

void createVBO(GLuint* vbo)
{
	//create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//initialize VBO
	unsigned int size = width * height * sizeof(float3);  // 3 floats
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//register VBO with CUDA
	cudaGLRegisterBufferObject(*vbo);
}



int main(int argc, char** argv) {

	// allocate memmory for the accumulation buffer on the GPU
	cudaMalloc(&accumulatebuffer, width * height * sizeof(float3));
	// load triangle meshes in CUDA memory
	initCUDAmemoryTriMesh();
	// init glut for OpenGL viewport
	glutInit(&argc, argv);
	// specify the display mode to be RGB and single buffering
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	// specify the initial window position
	glutInitWindowPosition(100, 100);
	// specify the initial window size
	glutInitWindowSize(width, height);
	// create the window and set title
	glutCreateWindow("Basic triangle mesh path tracer in CUDA");
	// init OpenGL
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, width, 0.0, height);
	fprintf(stderr, "OpenGL initialized \n");
	// register callback function to display graphics:
	glutDisplayFunc(disp);
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		//fflush(stderr);
		exit(0);
	}
	fprintf(stderr, "glew initialized  \n");
	// call Timer():
	Timer(0);
	//create VBO (vertex buffer object)
	createVBO(&vbo);
	fprintf(stderr, "VBO created  \n");
	// enter the main loop and process events
	fprintf(stderr, "Entering glutMainLoop...  \n");
	glutMainLoop();

	// free CUDA memory on exit
	cudaFree(accumulatebuffer);
	cudaFree(dev_triangle_p);
	cudaFree(dptr);
}
/*
纹理绑定步骤

第一至三步: main.cpp中

第一步，新建GPU数组

float* dev_triangle_p;

第二步：为GPU数组分配大小

triangle_size = cuda_scene_triangles.size() * sizeof(float4);

cudaMalloc((void**)&dev_triangle_p, triangle_size);

第三步：将CPU数组复制到GPU，这里数组中每个元素都是float4
cudaMemcpy(dev_triangle_p, &cuda_scene_triangles[0], triangle_size, cudaMemcpyHostToDevice);

第四至六步：kernel.cu中

第四步：新建纹理

texture<float4, 1, cudaReadModeElementType> Texture_triangle;

第五步：绑定纹理

size_t size = sizeof(float4) * number_of_scene_triangles * 3;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
cudaBindTexture(0, Texture_triangle, dev_triangle_p, channelDesc, size);

第六步：愉快的使用纹理
float4 edge1 = tex1Dfetch(Texture_triangle, triangleIndex * 3 + 1);



*/