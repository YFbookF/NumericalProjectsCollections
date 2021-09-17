// ch05 使用freeglut 来操作opengl -- 动态三角形
#include <iostream>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include "Camera.h"
#include "ShaderReader.h"
#define STB_IMAGE_IMPLEMENTATION // 不加上会出现未解析的符号 stb_image_load
#include "stb_image.h"
#include "MeshManager.h"

unsigned int VBO, VAO, EBO;
ShaderReader ourShader, depthShader;
MeshMananger meshManager;
unsigned int gScaleLocation, gWorldToCameraMatrix, modelLocation;
Camera camera;
float scale = 1.0f;
float farPlane = 1000.0f;
float nearPlane = 0.1;
float SCR_WIDTH = 800, SCR_HEIGHT = 800;
unsigned int texture_wall, texture_disp, texture_normal;

const unsigned int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;
unsigned int depthMapFBO;
unsigned int depthMap;


void RenderOneScene()
{
	Matrix4f modelMatrix, translateMatrix, rotateMatrix, scaleMatrix;

	translateMatrix = camera.GetTranslateMatirx(-1.5, 0, 30);
	scaleMatrix = camera.GetScaleMatrix(5, 0.2, 20);
	rotateMatrix = camera.GetRotateMatrix(0, 0, 1, 0);
	modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
	depthShader.setMatrix4("model", modelMatrix);
	meshManager.RenderCube();
}
void RenderAnthorScene()
{
	Matrix4f modelMatrix, translateMatrix, rotateMatrix, scaleMatrix;
	translateMatrix = camera.GetTranslateMatirx(2, 1, 12);
	scaleMatrix = camera.GetScaleMatrix(1, 1, 1);
	rotateMatrix = camera.GetRotateMatrix(0, 0, 1, 0);
	modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
	ourShader.setMatrix4("model", modelMatrix);
	meshManager.RenderCube();
}

void RenderMainProgram() {
	// 清空颜色缓存
	glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	Matrix4f projectionMatrix, viewMatrix;
	projectionMatrix = camera.PerspectiveRH_NO(camera.Zoom * M_PI / 180.0f, SCR_WIDTH / SCR_HEIGHT, 0.1, 1000.0f);
	viewMatrix = camera.lookAt();




	std::string strTextureLocation;
	camera.Zoom = 45.0f;


	
	
	//glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//	glDisable(GL_DEPTH_TEST);
	depthShader.use();
	// 没成功，参数记得都设置上
	depthShader.setMatrix4("projection", projectionMatrix);
	depthShader.setMatrix4("view", viewMatrix);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_wall);
	strTextureLocation = "diffuseMap";
	glUniform1i(glGetUniformLocation(depthShader.ID, strTextureLocation.c_str()), 0);
	// 绑定纹理要在RenderScene之前
	RenderOneScene();
	
	ourShader.use();
		ourShader.setMatrix4("projection", projectionMatrix);
	ourShader.setMatrix4("view", viewMatrix);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_disp);
	glUniform1i(glGetUniformLocation(ourShader.ID, strTextureLocation.c_str()), 0);
	RenderAnthorScene();
	
	//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
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
	depthShader.ReadShaderFile("depth.vert", "depth.frag");
	ourShader.ReadShaderFile("screen.vert", "screen.frag");


	glGenFramebuffers(1, &depthMapFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
	// create depth texture

	glGenTextures(1, &depthMap);
	glBindTexture(GL_TEXTURE_2D, depthMap);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// attach depth texture as FBO's depth buffer

	//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthMap, 0);
	//glDrawBuffer(GL_NONE);
	//glReadBuffer(GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
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