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
unsigned int gScaleLocation, gWorldToCameraMatrix, modelLocation;
Camera camera;
float scale = 1.0f;
float farPlane = 1000.0f;
float nearPlane = 0.1;
float SCR_WIDTH = 800, SCR_HEIGHT = 800;
unsigned int texture_wall, texture_disp, texture_normal;

void RenderMainProgram() {
	// 清空颜色缓存
	glClear(GL_COLOR_BUFFER_BIT);
	Matrix4f projection = camera.PerspectiveRH_NO(camera.Zoom * M_PI / 180.0f, SCR_WIDTH / SCR_HEIGHT, 0.1, 1000.0f);
	Matrix4f view = camera.lookAt();
	Matrix4f model = camera.GetRotateMatrix(30.0f, Vector3f(0, 1, 0));
	GLfloat tempgl[16];

	for (int j = 0; j < 4; j++)for (int i = 0; i < 4; i++)tempgl[j*4+i] = model(i, j);

	glUniform1f(gScaleLocation, scale);
	glUniformMatrix4fv(modelLocation, 1,GL_FALSE, (const GLfloat*)tempgl);

	camera.Zoom = 45.0f;
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_wall);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, texture_normal);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D,texture_disp);

	std::string strTextureLocation = "diffuseMap";
	glUniform1i(glGetUniformLocation(ourShader.ID, strTextureLocation.c_str()), 0);
	strTextureLocation = "normalMap";
	glUniform1i(glGetUniformLocation(ourShader.ID, strTextureLocation.c_str()), 1);
	strTextureLocation = "depthMap";
	glUniform1i(glGetUniformLocation(ourShader.ID, strTextureLocation.c_str()), 2);


	ourShader.use();



	glBindVertexArray(VAO); 
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	// 交换前后缓存
	glutSwapBuffers();
}
int loadTexture(char* ImagePath)
{
	unsigned int textureId;
	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	// load image, create texture and generate mipmaps
	int width, height, nrChannels;
	// The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
	unsigned char* data = stbi_load(ImagePath, &width, &height, &nrChannels, 0);
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
	return textureId;
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
	glutInitWindowSize(SCR_WIDTH, SCR_HEIGHT);      // 窗口尺寸
	glutInitWindowPosition(200, 100);  // 窗口位置
	glutCreateWindow("slow opengl learn");   // 窗口标题

	glutDisplayFunc(RenderMainProgram);
	glutIdleFunc(RenderMainProgram);

	// 缓存清空后的颜色值
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glewInit();
	ourShader.ReadShaderFile("simple.vert", "simple.frag");
	//gScaleLocation = glGetUniformLocation(ourShader.ID, "gScale");
	//gWorldToCameraMatrix = glGetUniformLocation(ourShader.ID, "worldToCameraMatrix");
	modelLocation = glGetUniformLocation(ourShader.ID, "model");
	float rect = 1.0f;

	float vertices[] = {
		// positions          // colors       
		 rect,  rect, nearPlane ,   1.0f, 0.0f, 0.0f,  1.0f,1.0f,
		 rect, -rect, nearPlane ,   0.0f, 1.0f, 0.0f,  1.0f,0.0f,
		-rect, -rect, nearPlane,   0.0f, 0.0f, 1.0f,              0.0f,0.0f,
		-rect,  rect, nearPlane,   1.0f, 1.0f, 0.0f,               0.0f,1.0f
	};

	float vertices[] = {
		// positions					// normal				//uv
		 rect,  rect, nearPlane ,	0.0f, 0.0f, 1.0f,		1.0f,1.0f,
		 rect, -rect, nearPlane ,	0.0f, 0.0f, 1.0f,		1.0f,0.0f,
		-rect, -rect, nearPlane,	0.0f, 0.0f, 1.0f,      0.0f,0.0f,
		-rect,  rect, nearPlane,	0.0f, 0.0f, 1.0f,      0.0f,1.0f,
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
	std::string strImage = "bricks2.jpg";
	char* imagePath = const_cast<char*>(strImage.c_str());
	glUniform1i(glGetUniformLocation(ourShader.ID, strImage.c_str()), 0);
	texture_wall = loadTexture(imagePath);
	strImage = "bricks2_disp.jpg";
	texture_disp = loadTexture(const_cast<char*>(strImage.c_str()));
	glUniform1i(glGetUniformLocation(ourShader.ID, strImage.c_str()), 1);
	strImage = "bricks2_normal.jpg";
	texture_normal = loadTexture(const_cast<char*>(strImage.c_str()));
	

	camera.SetPerspectiveMatrix(400.0f, 400.0f, 0.0f, 1000.0f, 60.0f);

	// 通知开始GLUT的内部循环
	glutMainLoop();

	return 0;
}