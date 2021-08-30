// ch05 使用freeglut 来操作opengl -- 动态三角形
#include <iostream>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include <Eigen\Core>

float static depth = 1.0f;

void DrawGround()
{
	glBegin(GL_TRIANGLES);
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f(-1.0, -1.0, 0.0);
	glVertex3f(-1.0, -1.0, -depth);
	glVertex3f(1.0, -1.0, -2.0);
	
	glVertex3f(-1.0, -1.0, 0.0);
	glVertex3f(1.0, -1.0, -2.0);
	glVertex3f(1.0, -1.0, 0.0);
	glEnd();
}

void RenderMainProgram() {

	// 清空颜色缓存
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//gluLookAt(-1,0,0,0,0,0,0,0,1);
	//glFrustum(-1, 1, -1, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	gluPerspective(60.0f, 1.0f, 1.0f, 100.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glRotated(20.0, 1, 0, 0);


	DrawGround();
	glPopMatrix();

	glutSwapBuffers();
}



void subStep()
{
	depth += 0.01f;
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

	// glut主循环
	glutMainLoop();

	return 0;
}