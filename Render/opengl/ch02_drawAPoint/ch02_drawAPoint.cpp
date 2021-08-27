// ch02 使用freeglut 来操作opengl -- 绘制一个点
#include <iostream>
#include <GL\glew.h>
#include <GL\freeglut.h>

GLuint VBO, VAO;
/**
 * 渲染回调函数
 */
void RenderMainProgram() {
    // 清空颜色缓存
    glClear(GL_COLOR_BUFFER_BIT);


    glEnableVertexAttribArray(0); // 我们目前只有顶点的位置这一个属性，所以激活0号索引就行了
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_POINTS, 0, 1);
    glDisableVertexAttribArray(0);

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
    glutInitWindowSize(480, 320);      // 窗口尺寸
    glutInitWindowPosition(100, 100);  // 窗口位置
    glutCreateWindow("Tutorial 01");   // 窗口标题

    // 开始渲染
    glutDisplayFunc(RenderMainProgram);

    // 缓存清空后的颜色值
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glewInit();
    
    float vertices[] = { 0.0f,0.0f,0.0f };
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // 通知开始GLUT的内部循环
    glutMainLoop();

    return 0;
}