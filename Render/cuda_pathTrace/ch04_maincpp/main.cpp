#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_render.h"

#include <stdlib.h>
#include <stdio.h>

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

	pre_render_kernel(dptr, accumulatebuffer, total_number_of_scene_triangles, frames, WangHash(frames), scene_aabbox_max, scene_aabbox_min);
	

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
