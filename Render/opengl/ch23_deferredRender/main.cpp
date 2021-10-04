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
float nearPlane = 10.0f;

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

	ourShader.use();
	glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
		   //glDrawArrays(GL_TRIANGLES, 0, 6);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	// 交换前后缓存
	glutSwapBuffers();
}

/**
 * 主函数
 */
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
	};
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
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// texture coord attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	// load and create a texture 
// -------------------------
	unsigned int texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	// load image, create texture and generate mipmaps
	int width, height, nrChannels;
	// The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
	unsigned char* data = stbi_load("wall.jpg", &width, &height, &nrChannels, 0);
	if (data)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else
	{
		std::cout << "Failed to load texture" << std::endl;
	}
	stbi_image_free(data);


	camera.SetPerspectiveMatrix(400.0f, 400.0f, 0.0f, 1000.0f, 60.0f);

	// 通知开始GLUT的内部循环
	glutMainLoop();

	return 0;
}