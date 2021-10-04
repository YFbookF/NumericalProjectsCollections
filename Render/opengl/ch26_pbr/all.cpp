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
ShaderReader skyShader, sceneShader;
Camera  sceneCamera, lightCamera;
MeshMananger meshManager;
unsigned int gScaleLocation;

float scale = 0.1f;
float near_plane = 1, far_plane = 100;
unsigned int texture_wall, texture_disp, texture_normal, texture_hdr, texture_cube, texture_sky;
Matrix4f modelMatrix, translateMatrix, rotateMatrix, scaleMatrix, lightSpaceMatrix;
std::string strTextureLocation;

unsigned int depthMapFBO, depthMap, skyVBO, skyVAO, cubeVBO, cubeVAO;
const int SCR_WIDTH = 600, SCR_HEIGHT = 600;
const int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;

bool renderDepth = true;
Matrix4f captureView[6];

float frame_current = 0;

Vector3f lightPositon[] = {
	Vector3f(-10,-10,10),
	Vector3f(-10,10,10),
	Vector3f(10,-10,10),
	Vector3f(10,10,10),
};
Vector3f lightColor[] = {
	Vector3f(200,200,200),
    Vector3f(200,200,200),
    Vector3f(200,200,200),
	Vector3f(200,200,200),
};
void RenderScene(ShaderReader whichShader)
{
	translateMatrix = sceneCamera.GetTranslateMatirx(0, -1, 0);
	scaleMatrix = sceneCamera.GetScaleMatrix(3, 0.1, 3);
	rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
	modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
	whichShader.setMatrix4("model", modelMatrix);
	meshManager.RenderCube();

	translateMatrix = sceneCamera.GetTranslateMatirx(1, 0, 2);
	scaleMatrix = sceneCamera.GetScaleMatrix(0.5, 1, 0.5);
	rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
	modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
	whichShader.setMatrix4("model", modelMatrix);
	meshManager.RenderSphere();

	translateMatrix = sceneCamera.GetTranslateMatirx(1, 0, -2);
	scaleMatrix = sceneCamera.GetScaleMatrix(0.5, 1, 0.5);
	rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
	modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
	whichShader.setMatrix4("model", modelMatrix);
	meshManager.RenderSphere();
}

int nrRows = 7, nrColumns = 7;
float spacing = 2.5;

void RenderMainProgram() {
	frame_current += 0.001f;
	//=======绘制正常场景================
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_DEPTH_TEST);
	//glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
	Matrix4f view = sceneCamera.lookAt();
	sceneShader.use();
	sceneShader.setMatrix4("projection", sceneCamera.PerspectiveMatrix);
	sceneShader.setMatrix4("view", view);
	sceneShader.setVec3("camPos", sceneCamera.Position(0), sceneCamera.Position(1), sceneCamera.Position(2));
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_sky);

	for (unsigned int i = 0; i < sizeof(lightPositon) / sizeof(lightPositon[0]); i++)
	{
		Vector3f newPos = lightPositon[i] + Vector3f(sin(frame_current), 0, sin(frame_current));
		sceneShader.setVec3("lightPositions[" + std::to_string(i) + "]", newPos(0), newPos(1), newPos(2));
		sceneShader.setVec3("lightColors[" + std::to_string(i) + "]", lightColor[i](0), lightColor[i](1), lightColor[i](2));

		/*
		translateMatrix = sceneCamera.GetTranslateMatirx(newPos(0), newPos(1), newPos(2));
		scaleMatrix = sceneCamera.GetScaleMatrix(0.2, 0.2, 0.2);
		rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
		modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
		sceneShader.setMatrix4("model", modelMatrix);
		meshManager.RenderSphere();
		*/
	}

	for (int row = 0; row < nrRows; row++)
	{
		sceneShader.setFloat("metallic", (float)row / (float)nrRows);
		for (int col = 0; col < nrColumns; col++)
		{
			sceneShader.setFloat("roughness", (float)col / (float)nrColumns + 0.1);
			translateMatrix = sceneCamera.GetTranslateMatirx(row * spacing, col * spacing, 0);
			scaleMatrix = sceneCamera.GetScaleMatrix(1, 1, 1);
			rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
			modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
			sceneShader.setMatrix4("model", modelMatrix);
			meshManager.RenderSphere();
		}
	}


	//RenderScene(sceneShader);
	//============天空盒=================
	glDepthFunc(GL_LEQUAL);
	skyShader.use();
	view(0, 3) = view(1, 3) = view(2, 3) = 0;
	view(3, 0) = view(3, 1) = view(3, 2) = 0;
	view(3, 3) = 1;
	skyShader.setMatrix4("projection", sceneCamera.PerspectiveMatrix);
	skyShader.setMatrix4("view", view);
	glBindVertexArray(skyVAO);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture_sky);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);
	glDepthFunc(GL_LESS);

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

unsigned int loadCubemap(std::vector<std::string> faces)
{
	unsigned int textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
	int width, height, nrChannels;
	for (unsigned int i = 0; i < faces.size(); i++)
	{
		unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
		if (data)
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
			stbi_image_free(data);
		}
		else
		{
			std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
			stbi_image_free(data);
		}
	}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	return textureID;
}

void process_Normal_Keys(int key, int x, int y)
{
	switch (key)
	{
	case 27:      break;
	case 100: printf("lightdirx %f\n", sceneCamera.Position(0)); sceneCamera.Position(0) += 0.1;   break;
	case 102: printf("lightdirx %f\n", sceneCamera.Position(0)); sceneCamera.Position(0) -= 0.1;  break;
	case 101: printf("lightdiry %f\n", sceneCamera.Position(2)); sceneCamera.Position(2) += 0.1;  break;
	case 103: printf("lightdiry %f\n", sceneCamera.Position(2)); sceneCamera.Position(2) -= 0.1;  break;
	}

}

int main(int argc, char** argv) {

	// 初始化GLUT
	glutInit(&argc, argv);
	// 显示模式：双缓冲、RGBA
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

	// 窗口设置
	glutInitWindowSize(SCR_HEIGHT, SCR_WIDTH);      // 窗口尺寸
	glutInitWindowPosition(100, 100);  // 窗口位置
	glutCreateWindow("CubeMap");   // 窗口标题

	glutDisplayFunc(RenderMainProgram);
	glutIdleFunc(RenderMainProgram);

	// 缓存清空后的颜色值
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glewInit();

	sceneShader.ReadShaderFile("simple.vert", "simple.frag");
	skyShader.ReadShaderFile("skybox.vert", "skybox.frag");

	sceneShader.use();
	sceneShader.setVec3("albedo", 0.5f, 0.0f, 0.0f);
	sceneShader.setFloat("ao", 1.0f);

	skyShader.use();
	skyShader.setInt("skybox", 0);
	sceneCamera = Camera(4.0f, 4.0f, 6.0f, 0.0, 0.0, 0.0);
	sceneCamera.SetPerspectiveMatrix2(-0.5, 0.5, 0.5, -0.5, 1, 100);
	lightCamera = Camera(6.0f, 4.0f, 0.0f, 0.0, 0.0, 0.0);
	lightCamera.SetPerspectiveMatrix2(-0.5, 0.5, 0.5, -0.5, 1, 100);

	// ================= 读取图片====================
	std::string strImage = "bricks2.jpg";
	char* imagePath = const_cast<char*>(strImage.c_str());
	texture_wall = loadTexture(imagePath);
	strImage = "bricks2_disp.jpg";
	texture_disp = loadTexture(const_cast<char*>(strImage.c_str()));
	strImage = "bricks2_normal.jpg";
	texture_normal = loadTexture(const_cast<char*>(strImage.c_str()));
	strImage = "hdr/newport_loft.hdr";
	texture_hdr = loadTexture(const_cast<char*>(strImage.c_str()));
	std::vector<std::string> faces{
		"skybox/right.jpg",
		"skybox/left.jpg",
		"skybox/top.jpg",
		"skybox/bottom.jpg",
		"skybox/front.jpg",
		"skybox/back.jpg",
	};
	texture_sky = loadCubemap(faces);
	//=================天空盒====================
	float skyboxVertices[] = {
		// positions          
		-1.0f,  1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		-1.0f,  1.0f, -1.0f,
		 1.0f,  1.0f, -1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		 1.0f, -1.0f,  1.0f
	};
	glGenVertexArrays(1, &skyVAO);
	glGenBuffers(1, &skyVAO);
	glBindVertexArray(skyVAO);
	glBindBuffer(GL_ARRAY_BUFFER, skyVAO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	//=================设置摄像机=============================
	sceneCamera = Camera(4.0f, 4.0f, 6.0f, 0.0, 0.0, 0.0);
	sceneCamera.SetPerspectiveMatrix2(-0.5, 0.5, 0.5, -0.5, 1, 100);

	glutSpecialFunc(process_Normal_Keys);
	glutMainLoop();

	return 0;
}