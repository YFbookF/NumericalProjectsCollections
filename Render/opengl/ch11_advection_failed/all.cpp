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
float nearPlane = 1.732f;
unsigned int texture, velocityTexture;
int texture_width = 128, texture_height = 128, nrChannels;
Vector3f* velocity;
Vector3f* velocityTemp;
Vector3f* density;
Vector3f* densityTemp;
float SolverDomainLength = 1.0f;
float dx = SolverDomainLength / texture_width;
float velocityScale = .1f;

enum class RungeKuttaType { NONE, first, second, third, fourth };
RungeKuttaType rungeType = RungeKuttaType::first;

enum class SemiLagrangeType { NONE, Semi, MidPoint, BEFFC };
SemiLagrangeType semiType = SemiLagrangeType::Semi;

enum class InterpolationVelocityType { none, fixed, linear, quadratic, cubic };
InterpolationVelocityType interVelType = InterpolationVelocityType::fixed;

enum class InterpolationDensityType { none, linear, quadratic, cubic };
InterpolationDensityType interDenType = InterpolationDensityType::linear;

void AdvectionVelocity();
void AdvectionDensity();
void RungeKutta4th();

void DensityToTexture()
{
	glActiveTexture(GL_TEXTURE0);
	unsigned char* newdata = (unsigned char*)malloc(texture_width * texture_height * 3 * sizeof(unsigned char));

	for (int j = 0; j < texture_height; j++)
	{
		for (int i = 0; i < texture_width; i++)
		{
			int idx = j * texture_width + i;
			newdata[idx * 3 + 0] = density[idx](0);
			newdata[idx * 3 + 1] = density[idx](1);
			newdata[idx * 3 + 2] = density[idx](2);
		}
	}

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, newdata);
	free(newdata);
}

void subStep()
{
	RungeKutta4th();
	DensityToTexture();
}


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
	glUniform1i(glGetUniformLocation(ourShader.ID, "texture0"), 0);
	glBindTexture(GL_TEXTURE_2D, texture);
	ourShader.use();
	glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
		   //glDrawArrays(GL_TRIANGLES, 0, 6);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	// 交换前后缓存
	glutSwapBuffers();
}

Vector3f sampleStatic(Vector3f pos)
{
	Vector3f res(0.0, 0.0, 0.0);
	float dis2 = (pos(0) - 0.5) * (pos(0) - 0.5) + (pos(1) - 0.5) * (pos(1) - 0.5) - 0.2;
	if (dis2 > 0)return res;

	// pos 的范围就是0~1
	res(0) = (pos(1) - 0.5f) * velocityScale * 2.0;
	res(1) = -(pos(0) - 0.5f) * velocityScale * 2.0;
	res(2) = 0;
	return res;
}

Vector3f sampleLinear(Vector3f pos, int level)
{
	Vector3f res = Vector3f(0.0f, 0.0f, 0.0f);
	Vector3f base = pos / dx;
	int basex = floor(base(0)), basey = floor(base(1));
	if (basex < 0)basex = 0;
	if (basex > texture_width - 2)basex = texture_width - 2;
	if (basey < 0)basey = 0;
	if (basey > texture_width - 2)basey = texture_width - 2;
	Vector2f f = Vector2f(base(0) - basex, base(1) - basey);

	Vector2f weight[2];
	weight[0](0) = 1.0f - f(0);
	weight[0](1) = 1.0f - f(1);
	weight[1](0) = f(0);
	weight[1](1) = f(1);


	for (int j = basey; j < basey + 2; j++)
	{
		for (int i = basex; i < basex + 2; i++)
		{
			if (i < 0 || i > texture_width - 1 || j < 0 || j > texture_height - 1)continue;
			int idx = j * texture_width + i;
			float w = weight[i - basex](0) * weight[j - basey](1);
			if (level == 1)
			{
				res += density[idx] * w;
			}
			else
			{
				res += velocity[idx] * w;
			}
		}
	}
	return res;
}
Vector3f sampleQuadratic(Vector3f pos, int level)
{
	Vector3f res = Vector3f(0.0f, 0.0f, 0.0f);
	Vector3f base = pos / dx;// --Vector3f(0.5f, 0.5f, 0.5f);
	int basex = floor(base(0)), basey = floor(base(1));
	Vector2f f = Vector2f(base(0) - basex, base(1) - basey);
	/*
	if (basex < 0)basex = 0,f(0) = 0;
	if (basex > texture_width - 3)basex = texture_width - 3,f(0) = 1;
	if (basey < 0)basey = 0,f(1) = 0;
	if (basey > texture_width - 3)basey = texture_width - 3,f(1) = 1;*/

	Vector2f weight[3];

	/*
	* 
	* 感觉下面这些代码有问题？并不精确，不如直接从Taylor公式推导而来
	weight[0](0) = 0.5f * (1.5f - f(0)) * (1.5f - f(0));
	weight[0](1) = 0.5f * (1.5f - f(1)) * (1.5f - f(1));

	weight[1](0) = 0.75f - (f(0) - 1.0f) * (f(0) - 1.0f);
	weight[1](1) = 0.75f - (f(1) - 1.0f) * (f(1) - 1.0f);

	weight[2](0) = 0.5f * (f(0) - 0.5f) * (f(0) - 0.5f);
	weight[2](1) = 0.5f * (f(1) - 0.5f) * (f(1) - 0.5f);
	*/

	weight[0](0) = 1.0f-1.5f * f(0) + f(0) * f(0) / 2.0f;
	weight[0](1) = 1.0f -1.5f * f(1) + f(1) * f(1) / 2.0f;

	weight[1](0) = 2.0 * f(0) - f(0) * f(0);
	weight[1](1) = 2.0 * f(1) - f(1) * f(1);

	weight[2](0) = -f(0) / 2.0 + f(0) * f(0) / 2.0f;
	weight[2](1) = -f(1) / 2.0 + f(1) * f(1) / 2.0f;

	for (int j = basey; j < basey + 3; j++)
	{
		for (int i = basex; i < basex + 3; i++)
		{
			int bx = i, by = j;
			float w = weight[bx - basex](0) * weight[by - basey](1);
			if (bx < 0)bx = 0;
			if (bx > texture_width - 1)bx = texture_width - 1;
			if (by < 0)by = 0;
			if (by > texture_height - 1)by = texture_height - 1;
			int idx = by * texture_width + bx;
			if (level == 1)res += density[idx] * w;
			else res += velocity[idx] * w;
		}
	}
	return res;
}

Vector3f sampleCubic(Vector3f pos, int level)
{
	Vector3f res = Vector3f(0.0f, 0.0f, 0.0f);
	Vector3f base = pos / dx - Vector3f(1.0f, 1.0f, 1.0f);
	int basex = floor(base(0)), basey = floor(base(1));
	if (basex < 0)basex = 0;
	if (basex > texture_width - 2)basex = texture_width - 2;
	if (basey < 0)basey = 0;
	if (basey > texture_width - 2)basey = texture_width - 2;
	Vector2f f = Vector2f(base(0) - basex, base(1) - basey);

	Vector2f weight[4];
	weight[0](0) = (2.0f - f(0)) * (2.0f - f(0)) * (2.0f - f(0)) * 0.16666666f;
	weight[0](1) = (2.0f - f(1)) * (2.0f - f(1)) * (2.0f - f(1)) * 0.16666666f;
	weight[1](0) = 0.5f * (f(0) - 1.0) * (f(0) - 1.0f) * (f(0) - 1.0f) - (f(0) - 1.0f) * (f(0) - 1.0f) + 0.66666666f;
	weight[1](1) = 0.5f * (f(1) - 1.0) * (f(1) - 1.0f) * (f(1) - 1.0f) - (f(1) - 1.0f) * (f(1) - 1.0f) + 0.66666666f;
	weight[2](0) = 0.5f * (f(0) - 1.0) * (f(0) - 1.0f) * (1.0f - f(0)) - (f(0) - 1.0f) * (f(0) - 1.0f) + 0.66666666f;
	weight[2](1) = 0.5f * (f(1) - 1.0) * (f(1) - 1.0f) * (1.0f - f(1)) - (f(1) - 1.0f) * (f(1) - 1.0f) + 0.66666666f;
	weight[3](0) = (f(0) + 1.0f) * (f(0) + 1.0f) * (f(0) + 1.0f) * 0.16666666;
	weight[3](1) = (f(1) + 1.0f) * (f(1) + 1.0f) * (f(1) + 1.0f) * 0.16666666;

	for (int j = basey; j < basey + 4; j++)
	{
		for (int i = basex; i < basex + 4; i++)
		{
			int bx = i, by = j;
			if (bx < 0)bx = 0;
			if (bx > texture_width - 1)bx = texture_width - 1;
			if (by < 0)by = 0;
			if (by > texture_height - 1)by = texture_height - 1;
			int idx = by * texture_width + bx;
			float w = weight[bx - basex](0) * weight[by - basey](1);
			if (level == 1)res += density[idx] * w;
			else res += velocity[idx] * w;
		}
	}
	return res;
}
Vector3f getVelocity(Vector3f pos)
{
	Vector3f res;
	switch (interVelType)
	{
	case InterpolationVelocityType::none:
		break;
	case InterpolationVelocityType::fixed:
		res = sampleStatic(pos);
		break;
	case InterpolationVelocityType::linear:
		res = sampleLinear(pos, 0);
		break;
	case InterpolationVelocityType::quadratic:
		res = sampleQuadratic(pos, 0);
		break;
	case InterpolationVelocityType::cubic:
		res = sampleCubic(pos, 0);
		break;
	default:
		break;
	}
	return res;
}
Vector3f getDensity(Vector3f pos)
{
	Vector3f res;
	switch (interDenType)
	{
	case InterpolationDensityType::none:
		break;
	case InterpolationDensityType::linear:
		res = sampleLinear(pos, 1);
		break;
	case InterpolationDensityType::quadratic:
		res = sampleQuadratic(pos, 1);
		break;
	case InterpolationDensityType::cubic:
		res = sampleCubic(pos, 1);
		break;
	default:
		break;
	}
	return res;
}
float dt = 0.1f;
/*
* BACK AND FORTH ERROR COMPENSATION AND CORRECTION
* https://www.shadertoy.com/view/wt33z2
*
*/

void RungeKutta4th()
{
    Vector3f* 	density0 = (Vector3f*)malloc(texture_width * texture_height * sizeof(Vector3f));
	Vector3f* density1 = (Vector3f*)malloc(texture_width * texture_height * sizeof(Vector3f));
	Vector3f* density2 = (Vector3f*)malloc(texture_width * texture_height * sizeof(Vector3f));
	Vector3f* density3 = (Vector3f*)malloc(texture_width * texture_height * sizeof(Vector3f));
	for (int i = 0; i < texture_width * texture_height; i++)density0[i] = density[i];
	AdvectionDensity();
	for (int i = 0; i < texture_width * texture_height; i++)density1[i] = density[i];
	AdvectionDensity();
	for (int i = 0; i < texture_width * texture_height; i++)density2[i] = density[i];
	AdvectionDensity();
	for (int i = 0; i < texture_width * texture_height; i++)density3[i] = density[i];
	AdvectionDensity();
	for (int i = 0; i < texture_width * texture_height; i++)
	{
		density[i] = density0[i] * 9.0 / 24.0 + density1[i] / 3.0 + density2[i] / 4.0 + densityTemp[i] / 24.0;
	}
}

void AdvectionVelocity()
{
	for (int j = 0; j < texture_height; j++)
	{
		for (int i = 0; i < texture_width; i++)
		{
			int idx = j * texture_width + i;
			Vector3f vel = velocity[idx];
			Vector3f pos = Vector3f(i * dx, j * dx, 0.0f);
			Vector3f result;
			switch (semiType)
			{
			case SemiLagrangeType::Semi:
			{

				// y = L(v, x)
				Vector3f pos_forward = pos - dt * vel;
				Vector3f vel_forward = getVelocity(pos_forward);
				result = vel_forward;
				break;
			}
			case SemiLagrangeType::BEFFC:
			{
				// y = L(v, x)
				Vector3f pos_forward = pos - dt * vel;
				Vector3f vel_forward = getVelocity(pos_forward);
				// z = L(-v, y)   注意dt 之前的符号
				Vector3f pos_backward = pos_forward + dt * vel_forward;
				// x = L(v, x + 0.5 * (x - z))
				Vector3f pos_corrected = pos + 0.5 * (pos - pos_backward);
				Vector3f vel_corrected = getVelocity(pos_corrected);

				Vector3f pos_final = pos_corrected - 0.5 * vel_corrected;
				Vector3f vel_final = getVelocity(pos_final);
				result = vel_final;
				break;
			}

			default:
				break;
			}

			velocityTemp[idx] = result;
		}
	}
	for (int j = 0; j < texture_height; j++)
	{
		for (int i = 0; i < texture_width; i++)
		{
			int idx = j * texture_width + i;
			velocity[idx] = velocityTemp[idx];
		}
	}
}

void AdvectionDensity()
{
	for (int j = 0; j < texture_height; j++)
	{
		for (int i = 0; i < texture_width; i++)
		{
			int idx = j * texture_width + i;
			Vector3f vel = velocity[idx];
			Vector3f pos = Vector3f(i * dx, j * dx, 0.0f);
			Vector3f result;
			switch (semiType)
			{
			case SemiLagrangeType::Semi:
			{

				// y = L(v, x)
				Vector3f pos_forward = pos - dt * vel;
				Vector3f den_forward = getDensity(pos_forward);
				result = den_forward;
				break;
			}
			case SemiLagrangeType::BEFFC:
			{
				// y = L(v, x)
				Vector3f pos_forward = pos - dt * vel;
				Vector3f vel_forward = getVelocity(pos_forward);
				// z = L(-v, y)   注意dt 之前的符号
				Vector3f pos_backward = pos_forward + dt * vel_forward;
				// x = L(v, x + 0.5 * (x - z))
				Vector3f pos_corrected = pos +0.5 * (pos - pos_backward);
				Vector3f vel_corrected = getVelocity(pos_corrected);

				Vector3f pos_final = pos_corrected - dt * vel_corrected;
				Vector3f den_final = getDensity(pos_final);
				result = den_final;
				break;
			}

			default:
				break;
			}

			densityTemp[idx] = result;
		}
	}
	for (int j = 0; j < texture_height; j++)
	{
		for (int i = 0; i < texture_width; i++)
		{
			int idx = j * texture_width + i;
			density[idx](0) = densityTemp[idx](0);
			density[idx](1) = densityTemp[idx](1);
			density[idx](2) = densityTemp[idx](2);
		}
	}
}

void InitVelocityTexture()
{
	velocity = (Vector3f*)malloc(texture_width * texture_height * sizeof(Vector3f));
	velocityTemp = (Vector3f*)malloc(texture_width * texture_height * sizeof(Vector3f));
	density = (Vector3f*)malloc(texture_width * texture_height * sizeof(Vector3f));
	densityTemp = (Vector3f*)malloc(texture_width * texture_height * sizeof(Vector3f));

	for (int j = 0; j < texture_height; j++)
	{
		for (int i = 0; i < texture_width; i++)
		{
			int idx = j * texture_width + i;
			float dis2 = (i * dx - 0.5) * (i * dx - 0.5) + (j * dx - 0.5) * (j * dx - 0.5) - 0.2;
			if(dis2 > 0)
			{
				velocity[idx](0) = 0;
				velocity[idx](1) = -0;
			}
			else
			{

			velocity[idx](0) = ((float)j / (float)texture_height * 2.0f - 1.0f) * velocityScale;
			velocity[idx](1) = -((float)i / (float)texture_width * 2.0f - 1.0f) * velocityScale;
			}
			velocity[idx](2) = 0;
			velocityTemp[idx](0) = 0;
			velocityTemp[idx](1) = 0;
			velocityTemp[idx](2) = 0;
		}
	}

	unsigned char* data = stbi_load("wall.jpg", &texture_width, &texture_height, &nrChannels, 0);

	for (int j = 0; j < texture_height; j++)
	{
		for (int i = 0; i < texture_width; i++)
		{
			int idx = j * texture_width + i;
			density[idx](0) = data[idx * 3 + 0];
			density[idx](1) = data[idx * 3 + 1];
			density[idx](2) = data[idx * 3 + 2];
		}
	}

	unsigned char* newdata = (unsigned char*)malloc(texture_width * texture_height * 3 * sizeof(unsigned char));

	for (int j = 0; j < texture_height; j++)
	{
		for (int i = 0; i < texture_width; i++)
		{
			int idx = j * texture_width + i;
			newdata[idx * 3 + 0] = density[idx](0);
			newdata[idx * 3 + 1] = density[idx](1);
			newdata[idx * 3 + 2] = density[idx](2);
		}
	}
	glActiveTexture(GL_TEXTURE0);
	if (density)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, newdata);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else
	{
		std::cout << "Failed to load texture" << std::endl;
	}
	stbi_image_free(data);
	free(newdata);

}

void timer(int junk)
{
	subStep();
	glutPostRedisplay();
	glutTimerFunc(30, timer, 0);
}

int main(int argc, char** argv) {

	// 初始化GLUT
	glutInit(&argc, argv);
	// 显示模式：双缓冲、RGBA
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

	// 窗口设置
	glutInitWindowSize(512, 512);      // 窗口尺寸
	glutInitWindowPosition(300, 300);  // 窗口位置
	glutCreateWindow("Tutorial 01");   // 窗口标题

	glutTimerFunc(1000, timer, 0);
	glutDisplayFunc(RenderMainProgram);

	// 缓存清空后的颜色值
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glewInit();
	ourShader.ReadShaderFile("simple.vert", "simple.frag");
	gScaleLocation = glGetUniformLocation(ourShader.ID, "gScale");
	gWorldToCameraMatrix = glGetUniformLocation(ourShader.ID, "worldToCameraMatrix");



	float rect = 1.0f;

	float vertices[] = {
		// positions          // colors       
		 rect,  rect, nearPlane,   1.0f, 0.0f, 0.0f,  1.0f,1.0f,
		 rect, -rect, nearPlane,   0.0f, 1.0f, 0.0f,  1.0f,0.0f,
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




	// 初始化纹理
	glGenTextures(1, &texture);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// 初始化速度
	InitVelocityTexture();
	// 摄像机设置投影矩阵
	camera.SetPerspectiveMatrix(400.0f, 400.0f, 0.0f, 1000.0f, 60.0f);

	// glut主循环
	glutMainLoop();

	return 0;
}