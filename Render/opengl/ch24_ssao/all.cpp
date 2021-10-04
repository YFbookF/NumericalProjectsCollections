// ch05 使用freeglut 来操作opengl -- 动态三角形
#include <iostream>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include "ShaderReader.h"
#include <Eigen\Core>
#include <vector>

#include "Camera.h"
#include "MeshManager.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

unsigned int VBO, VAO, EBO;
ShaderReader shaderGeometryPass, shaderLightingPass, shaderSSAO, shaderSSAOBlur;
Camera  sceneCamera, lightCamera;
MeshMananger meshManager;
unsigned int gScaleLocation;

float scale = 0.1f;
float near_plane = 1, far_plane = 100;
unsigned int texture_wall, texture_disp, texture_normal;
Matrix4f translateMatrix, rotateMatrix, scaleMatrix, lightSpaceMatrix;
std::string strTextureLocation;

unsigned int depthMapFBO, depthMap;
const int SCR_WIDTH = 600, SCR_HEIGHT = 600;
const int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;

std::vector<Vector3f> ssaoKernel, ssaoNoise;
Vector3f lightPositions, lightColors;

unsigned int noiseTexture;
unsigned int gBuffer, gPosition, gNormal, gAlbedo;
unsigned int rboDepth, ssaoFBO, ssaoBlurFBO, ssaoColorBuffer, ssaoColorBufferBlur;

std::vector<Matrix4f> modelMatrix;

void RenderScene(ShaderReader whichShader)
{
	for (unsigned int im = 0; im < modelMatrix.size(); im++)
	{
		whichShader.setMatrix4("model", modelMatrix[im]);
		meshManager.RenderCube();
	}
}

void RenderMainProgram() {


	//========= 第一步：渲染几何场景==================
	glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
	glEnable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	shaderGeometryPass.use();
	shaderGeometryPass.setMatrix4("projection", sceneCamera.PerspectiveMatrix);
	shaderGeometryPass.setMatrix4("view", sceneCamera.lookAt());
	shaderGeometryPass.setInt("invertedNormals", 0);
	RenderScene(shaderGeometryPass);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDisable(GL_DEPTH_TEST);

	//========= 第二步：产生SSAO纹理==================
	glBindFramebuffer(GL_FRAMEBUFFER, ssaoFBO);
	glClear(GL_COLOR_BUFFER_BIT);
	shaderSSAO.use();
	for (unsigned int i = 0; i < 64; ++i)// Send kernel + rotation 
		shaderSSAO.setVec3("samples[" + std::to_string(i) + "]", ssaoKernel[i](0), ssaoKernel[i](1), ssaoKernel[i](2));
	shaderSSAO.setMatrix4("projection", sceneCamera.PerspectiveMatrix);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gPosition);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, gNormal);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, noiseTexture);
	meshManager.RenderQuad();

	//========= 第三步：模糊上步产生的纹理，去除噪声==================
	glBindFramebuffer(GL_FRAMEBUFFER, ssaoBlurFBO);
	glClear(GL_COLOR_BUFFER_BIT);
	shaderSSAOBlur.use();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, ssaoColorBuffer);
	meshManager.RenderQuad();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//========= 第四步：处理光照==================
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	shaderLightingPass.use();
	Vector4f lightPosView = sceneCamera.lookAt() * Vector4f(lightPositions(0), lightPositions(1), lightPositions(2), 1.0);
	shaderLightingPass.setVec3("light.Position", lightPosView(0), lightPosView(1), lightPosView(2));
	shaderLightingPass.setVec3("light.Color", lightColors(0), lightColors(1), lightColors(2));
	// Update attenuation parameters
	const float linear = 0.09;
	const float quadratic = 0.032;
	shaderLightingPass.setFloat("light.Linear", linear);
	shaderLightingPass.setFloat("light.Quadratic", quadratic);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gPosition);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, gNormal);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, gAlbedo);
	glActiveTexture(GL_TEXTURE3); // add extra SSAO texture to lighting pass
	glBindTexture(GL_TEXTURE_2D, ssaoColorBufferBlur);
	meshManager.RenderQuad();

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

void process_Normal_Keys(int key, int x, int y)
{
	switch (key)
	{
	case 27:      break;
	case 100: sceneCamera.Position(0) += 0.1;   break;
	case 102: sceneCamera.Position(0) -= 0.1;  break;
	case 101: sceneCamera.Position(2) += 0.1;  break;
	case 103: sceneCamera.Position(2) -= 0.1;  break;
	}

}

int main(int argc, char** argv) {

	// ================== 初始化glut  ==========================
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(SCR_HEIGHT, SCR_WIDTH);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Tutorial 01");
	glutDisplayFunc(RenderMainProgram);
	glutIdleFunc(RenderMainProgram);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glewInit();

	// ================== 着色器相关 ==========================
	shaderGeometryPass.ReadShaderFile("ssao_geometry.vert", "ssao_geometry.frag");
	shaderLightingPass.ReadShaderFile("ssao.vert", "ssao_lighting.frag");
	shaderSSAO.ReadShaderFile("ssao.vert", "ssao.frag");
	shaderSSAOBlur.ReadShaderFile("ssao.vert", "ssao_blur.frag");



	// ================== 设置摄像机 ==========================
	sceneCamera = Camera(4.0f, 4.0f, 6.0f, 0.0, 0.0, 0.0);
	sceneCamera.SetPerspectiveMatrix2(-0.5, 0.5, 0.5, -0.5, 1, 100);
	lightCamera = Camera(6.0f, 4.0f, 0.0f, 0.0, 0.0, 0.0);
	lightCamera.SetPerspectiveMatrix2(-0.5, 0.5, 0.5, -0.5, 1, 100);

	// ================== 加载图片 ==========================
	std::string strImage = "bricks2.jpg";
	char* imagePath = const_cast<char*>(strImage.c_str());
	texture_wall = loadTexture(imagePath);
	strImage = "bricks2_disp.jpg";
	texture_disp = loadTexture(const_cast<char*>(strImage.c_str()));
	strImage = "bricks2_normal.jpg";
	texture_normal = loadTexture(const_cast<char*>(strImage.c_str()));

	// ================== 阴影的FrameBuffer ==========================
	glGenFramebuffers(1, &depthMapFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
	glGenTextures(1, &depthMap);
	glBindTexture(GL_TEXTURE_2D, depthMap);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthMap, 0);
	//glDrawBuffer(GL_NONE);
	//glReadBuffer(GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	// ================== gBuffer ==========================
	glGenFramebuffers(1, &gBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
	// position color buffer
	glGenTextures(1, &gPosition);
	glBindTexture(GL_TEXTURE_2D, gPosition);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0);
	// normal color buffer
	glGenTextures(1, &gNormal);
	glBindTexture(GL_TEXTURE_2D, gNormal);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormal, 0);
	// color + specular color buffer
	glGenTextures(1, &gAlbedo);
	glBindTexture(GL_TEXTURE_2D, gAlbedo);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gAlbedo, 0);
	// tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	unsigned int attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
	glDrawBuffers(3, attachments);
	//==================深度RenderBuffer==============
	glGenRenderbuffers(1, &rboDepth);
	glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//================== SSAO FBO =================
	glGenFramebuffers(1, &ssaoFBO);  glGenFramebuffers(1, &ssaoBlurFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, ssaoFBO);
	// SSAO color buffer
	glGenTextures(1, &ssaoColorBuffer);
	glBindTexture(GL_TEXTURE_2D, ssaoColorBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssaoColorBuffer, 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "SSAO Framebuffer not complete!" << std::endl;
	// and blur stage
	glBindFramebuffer(GL_FRAMEBUFFER, ssaoBlurFBO);
	glGenTextures(1, &ssaoColorBufferBlur);
	glBindTexture(GL_TEXTURE_2D, ssaoColorBufferBlur);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssaoColorBufferBlur, 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "SSAO Blur Framebuffer not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	// ================= SSAO核心==================
	for (unsigned int i = 0; i < 64; i++)
	{
		Vector3f sample(rand() / double(RAND_MAX) * 2.0 - 1.0,
			rand() / double(RAND_MAX) * 2.0 - 1.0,
			rand() / double(RAND_MAX));
		sample.normalize();
		sample *= rand() / double(RAND_MAX);

		float localScale = float(i) / 64.0;
		localScale = 0.9f * localScale * localScale + 0.1f;
		sample *= localScale;

		ssaoKernel.push_back(sample);
	}
	// ================= SSAO噪声==================
	for (unsigned int i = 0; i < 64; i++)
	{
		Vector3f sample(rand() / double(RAND_MAX) * 2.0 - 1.0,
			rand() / double(RAND_MAX) * 2.0 - 1.0,
			0.0f);
		ssaoNoise.push_back(sample);
	}
	glGenTextures(1, &noiseTexture);
	glBindTexture(GL_TEXTURE_2D, noiseTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssaoNoise[0]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	//================== 灯光 =====================
	lightPositions = Vector3f(0.0, 5.0, -5.0);
	lightColors = Vector3f(0.9, 0.9, 0.7);

	translateMatrix = sceneCamera.GetTranslateMatirx(0, -1, 0);
	scaleMatrix = sceneCamera.GetScaleMatrix(3, 0.1, 3);
	rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
	modelMatrix.push_back(translateMatrix * rotateMatrix * scaleMatrix);

	translateMatrix = sceneCamera.GetTranslateMatirx(1, 0, 2);
	scaleMatrix = sceneCamera.GetScaleMatrix(0.5, 1, 0.5);
	rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
	modelMatrix.push_back(translateMatrix * rotateMatrix * scaleMatrix);

	translateMatrix = sceneCamera.GetTranslateMatirx(0, 0, -2);
	scaleMatrix = sceneCamera.GetScaleMatrix(0.5, 1, 0.5);
	rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
	modelMatrix.push_back(translateMatrix * rotateMatrix * scaleMatrix);

	translateMatrix = sceneCamera.GetTranslateMatirx(-1, 0, -2.5);
	scaleMatrix = sceneCamera.GetScaleMatrix(0.5, 1, 0.5);
	rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
	modelMatrix.push_back(translateMatrix * rotateMatrix * scaleMatrix);

	translateMatrix = sceneCamera.GetTranslateMatirx(1, 0, -2.5);
	scaleMatrix = sceneCamera.GetScaleMatrix(0.5, 1, 0.5);
	rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
	modelMatrix.push_back(translateMatrix * rotateMatrix * scaleMatrix);

	shaderLightingPass.use();
	shaderLightingPass.setInt("gPosition", 0);
	shaderLightingPass.setInt("gNormal", 1);
	shaderLightingPass.setInt("gAlbedo", 2);
	shaderLightingPass.setInt("ssao", 3);
	shaderSSAO.use();
	shaderSSAO.setInt("gPosition", 0);
	shaderSSAO.setInt("gNormal", 1);
	shaderSSAO.setInt("texNoise", 2);
	shaderSSAOBlur.use();
	shaderSSAOBlur.setInt("ssaoInput", 0);

	// ================== glut开始执行  ==========================
	glutSpecialFunc(process_Normal_Keys);
	glutMainLoop();

	return 0;
}