// ch05 使用freeglut 来操作opengl -- 动态三角形
#include <iostream>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include "Camera.h"
#include "ShaderReader.h"
#define STB_IMAGE_IMPLEMENTATION // 不加上会出现未解析的符号 stb_image_load
#include "stb_image.h"

unsigned int VBO, VAO, EBO;
ShaderReader ourShader;
unsigned int gScaleLocation, gWorldToCameraMatrix;
Camera camera;
float scale = 1.0f;
float nearPlane = 2.0f;
unsigned int texture, velocityTexture;
int texture_width, texture_height, nrChannels;

void RenderMainProgram() {
	// 清空颜色缓存
	glClear(GL_COLOR_BUFFER_BIT);

	glUniform1f(gScaleLocation, scale);


	GLfloat world2camera[] = { camera.PerspectiveMatrix(0,0),camera.PerspectiveMatrix(0,1) ,camera.PerspectiveMatrix(0,2),camera.PerspectiveMatrix(0,3) ,
		camera.PerspectiveMatrix(1,0),camera.PerspectiveMatrix(1,1) ,camera.PerspectiveMatrix(1,2),camera.PerspectiveMatrix(1,3) ,
		camera.PerspectiveMatrix(2,0),camera.PerspectiveMatrix(2,1) ,camera.PerspectiveMatrix(2,2),camera.PerspectiveMatrix(2,3) ,
		camera.PerspectiveMatrix(3,0),camera.PerspectiveMatrix(3,1) ,camera.PerspectiveMatrix(3,2),camera.PerspectiveMatrix(3,3) ,
	};
	glUniformMatrix4fv(gWorldToCameraMatrix, 1, GL_TRUE, (const GLfloat*)world2camera);

	glUniform1i(glGetUniformLocation(ourShader.ID, "texture1"), 0);
	glUniform1i(glGetUniformLocation(ourShader.ID, "texture2"), 1);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, velocityTexture);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, texture);
	ourShader.use();
	glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
		   //glDrawArrays(GL_TRIANGLES, 0, 6);
	glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0,2);
	// 交换前后缓存
	glutSwapBuffers();
}

void InitVelocityTexture()
{
	char* rgbImage = (char*)malloc(texture_width * texture_height * 3 * sizeof(char));  // check for NULL

	for (int j = 0; j < texture_height; j++)
	{
		for (int i = 0; i < texture_width; i++)
		{
			int idx = j * texture_width + i;
			rgbImage[idx * 3] = 30;
			rgbImage[idx * 3 + 1] = 0;
			rgbImage[idx * 3 + 2] = 1;
		}
	}
	glGenTextures(1, &velocityTexture);
	glActiveTexture(GL_TEXTURE1);
	// bind texture to render context
	glBindTexture(GL_TEXTURE_2D, velocityTexture);
	unsigned char* data = stbi_load("wall.jpg", &texture_width, &texture_height, &nrChannels, 0);
	// upload texture data
	glTexImage2D(GL_TEXTURE_2D, 0, 3, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgbImage);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

	// don't use mipmapping (since we're not creating any mipmaps); the default
	// minification filter uses mipmapping.  Use linear filtering for minification
	// and magnification.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

}


int main(int argc, char** argv) {

	// 初始化GLUT
	glutInit(&argc, argv);
	// 显示模式：双缓冲、RGBA
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

	// 窗口设置
	glutInitWindowSize(800, 600);      // 窗口尺寸
	glutInitWindowPosition(200, 100);  // 窗口位置
	glutCreateWindow("Tutorial 01");   // 窗口标题

	glutDisplayFunc(RenderMainProgram);
	glutIdleFunc(RenderMainProgram);

	// 缓存清空后的颜色值
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glewInit();
	ourShader.ReadShaderFile("simple.vert", "simple.frag");
	gScaleLocation = glGetUniformLocation(ourShader.ID, "gScale");
	gWorldToCameraMatrix = glGetUniformLocation(ourShader.ID, "worldToCameraMatrix");

	float rect = 1.0f;
	
	/*
	float vertices[] = {
		// positions          // colors       
		 rect,  rect, nearPlane + 10.0f,   1.0f, 0.0f, 0.0f,  1.0f,1.0f,
		 rect, -rect, nearPlane + 10.0f,   0.0f, 1.0f, 0.0f,  1.0f,0.0f,
		-rect, -rect, nearPlane,   0.0f, 0.0f, 1.0f,              0.0f,0.0f,
		-rect,  rect, nearPlane,   1.0f, 1.0f, 0.0f,               0.0f,1.0f
	};
	unsigned int indices[] = {  // note that we start from 0!
		0, 1, 3,  // first Triangle
		1, 2, 3   // second Triangle
	};*/
	const int Nx = 5;
	float vertices[Nx * Nx * 8];
	const int element_num = (Nx - 1) * (Nx - 1) * 2;
	int indices[element_num * 3];
	int cnt = 0;
	float dx = 8.0 / Nx;
	for (int j = 0; j < Nx; j++)
	{
		for (int i = 0; i < Nx; i++)
		{
			vertices[cnt * 8 + 0] = i * dx - 1;
			vertices[cnt * 8 + 1] = j * dx - 1;
			vertices[cnt * 8 + 2] = nearPlane;

			vertices[cnt * 8 + 3] = 1.0;
			vertices[cnt * 8 + 4] = 1.0;
			vertices[cnt * 8 + 5] = 1.0;

			vertices[cnt * 8 + 6] = 1.0f;
			vertices[cnt * 8 + 7] = 1.0f;
			cnt += 1;
		}
	}
	cnt = 0;
	for (int j = 0; j < Nx - 1; j++)
	{
		for (int i = 0; i < Nx - 1; i++)
		{
			int idx = j * Nx + i;
			indices[cnt*3 + 0] = idx;
			indices[cnt*3 + 1] = idx + 1;
			indices[cnt*3 + 2] = idx + Nx;

			indices[cnt*3 + 3] = idx + Nx;
			indices[cnt*3 + 4] = idx + 1;
			indices[cnt*3 + 5] = idx + Nx + 1;

			cnt += 2;
		}
	}
	
	/*
	float vertices[] = {
		// positions          // colors       
		 rect,  rect, nearPlane + 10.0f,   1.0f, 0.0f, 0.0f,  1.0f,1.0f,
		 rect, -rect, nearPlane + 10.0f,   0.0f, 1.0f, 0.0f,  1.0f,0.0f,
		-rect,  rect, nearPlane,   1.0f, 1.0f, 0.0f,               0.0f,1.0f,

		rect, -rect, nearPlane + 10.0f,   0.0f, 1.0f, 0.0f,  1.0f,0.0f,
		-rect, -rect, nearPlane,   0.0f, 0.0f, 1.0f,              0.0f,0.0f,
		-rect,  rect, nearPlane,   1.0f, 1.0f, 0.0f,               0.0f,1.0f
	};
	unsigned int indices[] = {  // note that we start from 0!
		0, 1, 2,  // first Triangle
		3, 4, 5   // second Triangle
	};*/
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	// 这个玩意会使用当前绑定到GL_ARRAY_BUFFER的VBO
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// texture coord attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	unsigned int acount = 1;
	Matrix4f modelMatrix = Matrix4f::Identity();
	unsigned int model_buffer;
	//glGenBuffers(1, &model_buffer);
	//glBindBuffer(GL_ARRAY_BUFFER, model_buffer);
	//glBufferData(GL_ARRAY_BUFFER, acount * sizeof(Matrix4f), &modelMatrix, GL_STATIC_DRAW);

	// load and create a texture 
// -------------------------
	glGenTextures(1, &texture);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	// load image, create texture and generate mipmaps

	// The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
	unsigned char* data = stbi_load("wall.jpg", &texture_width, &texture_height, &nrChannels, 0);

	for (int j = 0; j < texture_height; j++)
	{
		for (int i = 0; i < texture_width; i++)
		{
			int idx = j * texture_width + i;
			data[idx * 3] = 100;
			data[idx * 3 + 1] = 0;
			data[idx * 3 + 2] = 100;
		}
	}

	if (data)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else
	{
		std::cout << "Failed to load texture" << std::endl;
	}
	stbi_image_free(data);

	InitVelocityTexture();
	camera.SetPerspectiveMatrix(400.0f, 400.0f, 0.0f, 1000.0f, 60.0f);

	// 通知开始GLUT的内部循环
	glutMainLoop();

	return 0;
}